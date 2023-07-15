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
from torch.utils.data.dataset import Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold

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
                 kfold=5,
                 test_batch_size=64,
                 expert_dataset=None,
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
        self.splits = KFold(n_splits=kfold, shuffle=True, random_state=42)
        self.dataset = expert_dataset
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size

        # Define an Optimizer and a learning rate schedule.
        self.optimizer = optim.Adadelta(self.model.parameters(), lr=learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=scheduler_gamma)

    def train(self, epoch, train_loader):
        self.model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
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
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
                total_step = ((epoch-1) * len(train_loader.dataset)) + (batch_idx * len(data))
                self.writer.add_scalar('training loss',
                                       scalar_value=loss.item(),
                                       global_step=total_step)

    def test(self, epoch, test_loader):
        self.model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for data, target in test_loader:
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
                total_test_loss = total_test_loss + test_loss.item()
        print(f"Test set: Average loss: {total_test_loss:.4f}")
        self.writer.add_scalar('total test loss',
                               scalar_value=total_test_loss,
                               global_step=epoch)

    def learn(self, epochs, save_interval, checkpoint_dir):
        fold_epochs = int(epochs / self.splits.get_n_splits())

        for fold, (train_idx, test_idx) in enumerate(self.splits.split(np.arange(len(self.dataset)))):

            train_dataset = Subset(self.dataset, train_idx)
            test_dataset = Subset(self.dataset, test_idx)
            print("train_expert_dataset: ", len(train_dataset))
            print("test_expert_dataset: ", len(test_dataset))
            kwargs = {"num_workers": 1, "pin_memory": True}
            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset, batch_size=self.batch_size, shuffle=True, **kwargs
            )
            test_loader = torch.utils.data.DataLoader(
                dataset=test_dataset,
                batch_size=self.test_batch_size,
                shuffle=True,
                **kwargs,
            )
            for epoch in range(1, fold_epochs + 1):
                total_epoch = epoch + (fold * fold_epochs)
                self.train(total_epoch, train_loader)
                self.test(total_epoch, test_loader)
                self.scheduler.step()
                if total_epoch % save_interval == 0:
                    self.student.policy = self.model
                    model_path = os.path.join(checkpoint_dir, 'best_model_{}'.format(total_epoch))
                    print('Saving best_model_{} to {}'.format(total_epoch, model_path))
                    self.student.save(model_path)
            self.writer.close()


def collect_dataset():
    env = CrowdSimRaw()
    env.configure(config)
    env.setup(seed=0, num_of_env=1, ax=None)

    DATASET_NAME = 'BC_dataset_nofov_test'

    num_interactions = int(6e6)

    if isinstance(env.action_space, gym.spaces.Box):
        expert_observations = np.empty((num_interactions,) + env.observation_space.shape)
        expert_actions = np.empty((num_interactions,) + (env.action_space.shape[0],))

    else:
        expert_observations = np.empty((num_interactions,) + env.observation_space.shape)
        expert_actions = np.empty((num_interactions,) + env.action_space.shape)

    obs = env.reset()

    for i in tqdm(range(num_interactions)):
        action = np.array(env.calculate_orca())
        expert_observations[i] = obs
        expert_actions[i] = action
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()

    np.savez_compressed(
        DATASET_NAME,
        expert_actions=expert_actions,
        expert_observations=expert_observations,
    )


def train_bc():
    env = CrowdSimRaw()
    env.configure(config)
    env.setup(seed=0, num_of_env=1, ax=None)

    CHECKPOINT_DIR = './train/BC_APF_RAW/'
    LOG_DIR = './logs/BC_APF_RAW/'
    DATASET_NAME = 'BC_dataset_nofov_small.npz'

    np_data = np.load(DATASET_NAME)

    expert_dataset = ExpertDataSet(np_data['expert_observations'], np_data['expert_actions'])
    print("expert_dataset: ", len(expert_dataset))

    policy_kwargs = dict(
        features_extractor_class=ApfFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=512),
    )
    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, device='cuda', batch_size=64)

    # for name, param in model.policy.named_parameters():
    #     if param.requires_grad:
    #         print("GRAD: {}".format(name))
    #         print(param)
    #     else:
    #         print("NO GRAD: {}".format(name))

    bc = BehaviourCloning(student=model,
                          env=env,
                          batch_size=64,
                          scheduler_gamma=0.7,
                          learning_rate=1.0,
                          log_interval=1000,
                          device='cuda',
                          seed=1,
                          test_batch_size=64,
                          expert_dataset=expert_dataset,
                          tensorboard_log=LOG_DIR)
    bc.learn(epochs=int(1e3), save_interval=int(1e2), checkpoint_dir=CHECKPOINT_DIR)


def main():
    if not training:
        collect_dataset()
    else:
        train_bc()


if __name__ == '__main__':
    main()
