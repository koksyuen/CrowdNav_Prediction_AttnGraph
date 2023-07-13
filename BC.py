from crowd_sim.envs.crowd_sim_sgan import CrowdSimSgan
from crowd_sim.envs.crowd_sim_sgan_apf import CrowdSimSganApf
from crowd_sim.envs.crowd_sim_no_pred import CrowdSimNoPred
from crowd_sim.envs.crowd_sim_raw import CrowdSimRaw
from sb3.feature_extractor import Preprocessor, ApfFeaturesExtractor
import gym
import os
import time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataset import Dataset, random_split
from torch.utils.tensorboard import SummaryWriter

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO, A2C

from arguments import get_args
from crowd_nav.configs.config import Config
config = Config()

training = True
dataset_name = "apf_raw"


class ExpertDataSet(Dataset):
    def __init__(self, expert_observations, expert_actions):
        self.observations = expert_observations
        self.actions = expert_actions

    def __getitem__(self, index):
        return (self.observations[index], self.actions[index])

    def __len__(self):
        return len(self.observations)


class BehaviourCloning():
    def __init__(self,
                 student,
                 env,
                 batch_size=64,
                 scheduler_gamma=0.7,
                 learning_rate=1.0,
                 log_interval=100,
                 device='cuda',
                 seed=1,
                 test_batch_size=64,
                 train_expert_dataset=None,
                 test_expert_dataset=None,
                 tensorboard_log=None
                 ):
        torch.manual_seed(seed)
        if isinstance(env.action_space, gym.spaces.Box):
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.env = env
        self.student = student
        self.model = student.policy.to(device)
        self.device = device
        self.log_interval = log_interval
        self.writer = SummaryWriter(tensorboard_log)

        kwargs = {"num_workers": 1, "pin_memory": True}
        self.train_loader = torch.utils.data.DataLoader(
            dataset=train_expert_dataset, batch_size=batch_size, shuffle=True, **kwargs
        )
        self.test_loader = torch.utils.data.DataLoader(
            dataset=test_expert_dataset,
            batch_size=test_batch_size,
            shuffle=True,
            **kwargs,
        )

        # Define an Optimizer and a learning rate schedule.
        self.optimizer = optim.Adadelta(self.model.parameters(), lr=learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=scheduler_gamma)

    def train(self, epoch):
        self.model.train()

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            if isinstance(self.env.action_space, gym.spaces.Box):
                # A2C/PPO policy outputs actions, values, log_prob
                # SAC/TD3 policy outputs actions only
                if isinstance(self.student, (A2C, PPO)):
                    action, _, _ = self.model(data)
                else:
                    # SAC/TD3:
                    action = self.model(data)
                action_prediction = action.double()
            else:
                # Retrieve the logits for A2C/PPO when using discrete actions
                dist = self.model.get_distribution(data)
                action_prediction = dist.distribution.logits
                target = target.long()

            loss = self.criterion(action_prediction, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(self.train_loader.dataset),
                        100.0 * batch_idx / len(self.train_loader),
                        loss.item(),
                    )
                )
                self.writer.add_scalar('training loss',
                                       scalar_value=loss.item(),
                                       global_step=batch_idx * len(data))

    def test(self, epoch):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)

                if isinstance(self.env.action_space, gym.spaces.Box):
                    # A2C/PPO policy outputs actions, values, log_prob
                    # SAC/TD3 policy outputs actions only
                    if isinstance(self.student, (A2C, PPO)):
                        action, _, _ = self.model(data)
                    else:
                        # SAC/TD3:
                        action = self.model(data)
                    action_prediction = action.double()
                else:
                    # Retrieve the logits for A2C/PPO when using discrete actions
                    dist = self.model.get_distribution(data)
                    action_prediction = dist.distribution.logits
                    target = target.long()

                test_loss = self.criterion(action_prediction, target)
        test_loss /= len(self.test_loader.dataset)
        print(f"Test set: Average loss: {test_loss:.4f}")
        self.writer.add_scalar('testing loss',
                               scalar_value=test_loss,
                               global_step=epoch)

    def learn(self, epochs, save_interval, checkpoint_dir):
        for epoch in range(1, epochs + 1):
            self.train(epoch)
            self.test(epoch)
            self.scheduler.step()
            if epoch % save_interval== 0:
                self.student.policy = self.model
                model_path = os.path.join(checkpoint_dir, 'best_model_{}'.format(epoch))
                self.student.save(model_path)


def collect_dataset():
    env = CrowdSimRaw()
    env.configure(config)
    env.setup(seed=0, num_of_env=1, ax=None)

    num_interactions = int(6e6)

    if isinstance(env.action_space, gym.spaces.Box):
        expert_observations = np.empty((num_interactions,) + env.observation_space.shape)
        expert_actions = np.empty((num_interactions,) + (env.action_space.shape[0],))

    else:
        expert_observations = np.empty((num_interactions,) + env.observation_space.shape)
        expert_actions = np.empty((num_interactions,) + env.action_space.shape)

    obs = env.reset()

    for i in tqdm(range(num_interactions)):
        env.render()
        action = np.array(env.calculate_orca())
        expert_observations[i] = obs
        expert_actions[i] = action
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()

    np.savez_compressed(
        dataset_name,
        expert_actions=expert_actions,
        expert_observations=expert_observations,
    )


def train_bc():
    env = CrowdSimRaw()
    env.configure(config)
    env.setup(seed=0, num_of_env=1, ax=None)

    CHECKPOINT_DIR = './train/BC_APF_RAW/'
    LOG_DIR = './logs/BC_APF_RAW/'
    DATASET_NAME = 'apf_raw'

    np_data = np.load(DATASET_NAME)

    expert_dataset = ExpertDataSet(np_data['expert_observations'], np_data['expert_actions'])

    train_size = int(0.8 * len(expert_dataset))

    test_size = len(expert_dataset) - train_size

    train_expert_dataset, test_expert_dataset = random_split(expert_dataset,
                                                             [train_size, test_size])

    print("test_expert_dataset: ", len(test_expert_dataset))
    print("train_expert_dataset: ", len(train_expert_dataset))

    policy_kwargs = dict(
        features_extractor_class=ApfFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=512),
    )
    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, device='cuda', batch_size=64)

    bc = BehaviourCloning(student=model,
                          env=env,
                          train_expert_dataset=train_expert_dataset,
                          test_expert_dataset=test_expert_dataset,
                          tensorboard_log=LOG_DIR)
    bc.learn(epochs=int(1e4), save_interval=1e3, checkpoint_dir=CHECKPOINT_DIR)


def main():
    if not training:
        collect_dataset()
    else:
        train_bc()


if __name__ == '__main__':
    main()
