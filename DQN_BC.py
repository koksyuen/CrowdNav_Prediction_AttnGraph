from crowd_sim.envs.crowd_sim_sgan import CrowdSimSgan
from crowd_sim.envs.crowd_sim_sgan_apf import CrowdSimSganApf
from crowd_sim.envs.crowd_sim_no_pred import CrowdSimNoPred
from crowd_sim.envs.crowd_sim_raw import CrowdSimRaw
from sb3.feature_extractor import Preprocessor, ApfFeaturesExtractor
import gym
from gym.spaces import Discrete
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

from sb3.dqn.dqn import DQN

from arguments import get_args
from crowd_nav.configs.config import Config

config = Config()

# Discrete Actions
U_A = [-1.0, -0.5, 0.0, 0.5, 1.0]
u_a = np.array(U_A)
Y, X = np.meshgrid(u_a, u_a)
discrete_actions = np.stack((X, Y), axis=-1)
discrete_actions = discrete_actions.reshape((-1, 2))


class DiscreteActions(gym.ActionWrapper):
    def __init__(self, env, disc_to_cont):
        super().__init__(env)
        self.disc_to_cont = disc_to_cont
        self.action_space = Discrete(len(disc_to_cont))
        # self.action_space = Discrete(disc_to_cont.shape[0])

    def action(self, act):
        return self.disc_to_cont[act]

class ExpertDataSet(Dataset):
    def __init__(self, expert_observations, expert_actions, expert_rewards,
                 expert_episodes, expert_dones, episode_record):
        self.observations = expert_observations
        self.actions = expert_actions
        self.rewards = expert_rewards
        self.episodes = expert_episodes
        self.dones = expert_dones
        self.record = episode_record

    def __getitem__(self, index):
        return self.observations[index], self.actions[index]

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
        self.model = student.policy.q_net.to(device)
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
                        len(train_loader.sampler),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
                total_step = ((epoch-1) * len(train_loader.sampler)) + (batch_idx * len(data))
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
        print(f"Test set: Total loss: {total_test_loss:.4f}")
        self.writer.add_scalar('total test loss',
                               scalar_value=total_test_loss,
                               global_step=epoch)

    def learn(self, epochs, save_interval, checkpoint_dir, previous_epoch=0):
        fold_epochs = int((epochs - previous_epoch) / self.splits.get_n_splits())

        for fold, (train_idx, test_idx) in enumerate(self.splits.split(np.arange(len(self.dataset)))):

            print('Fold {}'.format(fold + 1))

            train_sampler = SubsetRandomSampler(train_idx)
            test_sampler = SubsetRandomSampler(test_idx)

            kwargs = {"num_workers": 1, "pin_memory": True}
            train_loader = torch.utils.data.DataLoader(
                dataset=self.dataset, batch_size=self.batch_size, sampler=train_sampler, **kwargs
            )
            test_loader = torch.utils.data.DataLoader(
                dataset=self.dataset,
                batch_size=self.test_batch_size,
                sampler=test_sampler,
                **kwargs,
            )

            print("train_expert_dataset: ", len(train_loader.sampler))
            print("test_expert_dataset: ", len(test_loader.sampler))

            for epoch in range(1, fold_epochs + 1):
                total_epoch = epoch + (fold * fold_epochs) + previous_epoch
                self.train(total_epoch, train_loader)
                self.test(total_epoch, test_loader)
                self.scheduler.step()
                if total_epoch % save_interval == 0:
                    model_path = os.path.join(checkpoint_dir, 'best_dict_{}.pth'.format(total_epoch))
                    print('Saving best_model_{} to {}'.format(total_epoch, model_path))
                    torch.save(self.model.state_dict(), model_path)
            self.writer.close()


def collect_dataset():
    env = CrowdSimRaw()
    env.configure(config)
    env.setup(seed=0, num_of_env=1, ax=None)

    DATASET_NAME = 'DQN_BC_dataset_nofov_test'

    num_interactions = 10000
    # num_interactions = int(2e6)

    # record the start index (step) and final index (step) of every episode
    episode_record = []

    if isinstance(env.action_space, gym.spaces.Box):
        expert_observations = np.empty((num_interactions,) + env.observation_space.shape)
        expert_actions = np.empty((num_interactions,) + (env.action_space.shape[0],))
        expert_rewards = np.empty((num_interactions,))
        expert_episodes = np.empty((num_interactions,), dtype='int')
        expert_dones = np.empty((num_interactions,), dtype='int8')
    else:
        expert_observations = np.empty((num_interactions,) + env.observation_space.shape)
        expert_actions = np.empty((num_interactions,) + env.action_space.shape)
        expert_rewards = np.empty((num_interactions,))
        expert_episodes = np.empty((num_interactions,))
        expert_dones = np.empty((num_interactions,))

    episode = 1
    previous_i = 0
    obs = env.reset()

    for i in tqdm(range(num_interactions)):
        action = np.array(env.calculate_orca())
        expert_observations[i] = obs
        expert_actions[i] = action
        obs, reward, done, info = env.step(action)
        expert_rewards[i] = reward
        expert_episodes[i] = episode
        if done:
            expert_dones[i] = 0   # 0: at the last step
            episode_record.append([previous_i, i])
            episode += 1
            previous_i = i + 1
            obs = env.reset()
        else:
            expert_dones[i] = 1   # 1: before last step

    # exclude the steps (at the end) that isn't a complete episode
    last_step = episode_record[-1][-1]

    np.savez_compressed(
        DATASET_NAME,
        expert_actions=expert_actions[:last_step],
        expert_observations=expert_observations[:last_step],
        expert_rewards=expert_rewards[:last_step],
        expert_episodes=expert_episodes[:last_step],
        expert_dones=expert_dones[:last_step],
        episode_record=np.array(episode_record, dtype='int')
    )


def train_bc():
    env = CrowdSimRaw()
    env.configure(config)
    env.setup(seed=0, num_of_env=1, ax=None)
    env = DiscreteActions(env, discrete_actions)

    CHECKPOINT_DIR = './train/DQN_BC_APF_RAW/'
    LOG_DIR = './logs/DQN_BC_APF_RAW/'
    DATASET_NAME = 'DQN_BC_dataset_nofov_test.npz'

    np_data = np.load(DATASET_NAME)

    expert_dataset = ExpertDataSet(np_data['expert_observations'], np_data['expert_actions'],
                                   np_data['expert_rewards'], np_data['expert_episodes'],
                                   np_data['expert_dones'], np_data['episode_record'])
    print("expert_dataset: ", len(expert_dataset))

    policy_kwargs = dict(
        features_extractor_class=ApfFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=512),
    )
    model = DQN("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, device='cuda', tensorboard_log=LOG_DIR,
                batch_size=128)

    print(model.policy.q_net)

    # OLD_MODEL_PATH = CHECKPOINT_DIR + 'best_dict_17.pth'
    # old_dict = torch.load(OLD_MODEL_PATH)
    # model.policy.load_state_dict(old_dict)

    # for name, param in model.policy.named_parameters():
    #     if param.requires_grad:
    #         print("GRAD: {}".format(name))
    #         print(param)
    #     else:
    #         print("NO GRAD: {}".format(name))

    bc = BehaviourCloning(student=model,
                          env=env,
                          batch_size=512,
                          scheduler_gamma=0.7,
                          learning_rate=0.00233,
                          log_interval=300,
                          device='cuda',
                          seed=1000,
                          test_batch_size=512,
                          expert_dataset=expert_dataset,
                          tensorboard_log=LOG_DIR)

    bc.learn(epochs=20, save_interval=1, checkpoint_dir=CHECKPOINT_DIR, previous_epoch=0)


def main():
    training = True
    if not training:
        collect_dataset()
    else:
        train_bc()


if __name__ == '__main__':
    main()
