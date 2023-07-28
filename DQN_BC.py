from crowd_sim.envs.crowd_sim_sgan import CrowdSimSgan
from crowd_sim.envs.crowd_sim_sgan_apf import CrowdSimSganApf
from crowd_sim.envs.crowd_sim_no_pred import CrowdSimNoPred
from crowd_sim.envs.crowd_sim_raw import CrowdSimRaw
from sb3.feature_extractor import Preprocessor, ApfFeaturesExtractor
from stable_baselines3.common.utils import polyak_update
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
from torch.utils.data.dataset import Dataset
from torch.utils.data import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold

from sb3.dqn.dqn import DQN

from arguments import get_args
from crowd_nav.configs.config import Config

config = Config()


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
                 expert_episodes, episode_record, dqfd_n_step=3):
        self.observations = expert_observations
        self.actions = expert_actions
        self.rewards = expert_rewards
        self.episodes = expert_episodes
        self.record = episode_record
        self.dqfd_n_step = dqfd_n_step
        self.r_n = np.zeros((dqfd_n_step - 1,))

    '''
    Return: s_0, a, r_arr, s_1, s_n, before_done, present_qn
    s_0: state_0
    a: a_0
    r_arr: r_0 ~ r_(n-1)
    s_1: state_1
    s_n: state_n
    before_done: 0 (episode_last_step), otherwise 1
    present_qn: 0 (if episode_last_step < n), otherwise 1
    '''

    def __getitem__(self, index):
        episode_last_index = self.record[self.episodes[index], -1]
        # if n < episode_last_step
        if episode_last_index - index >= self.dqfd_n_step:
            return (self.observations[index], self.actions[index],
                    self.rewards[index:index + self.dqfd_n_step], self.observations[index + 1],
                    self.observations[index + self.dqfd_n_step],
                    1, 1)
        # if episode_last_step
        elif index == episode_last_index:
            r_n = np.zeros((self.dqfd_n_step,))  # r_0 ~ r_(n-1)
            r_n[0] = self.rewards[index]
            return (self.observations[index], self.actions[index],
                    r_n, self.observations[index],
                    self.observations[index],
                    0, 0)
        # if episode_last_step < n
        elif episode_last_index - index < self.dqfd_n_step:
            n = episode_last_index - index
            r_n = np.zeros((self.dqfd_n_step,))  # r_0 ~ r_(n-1)
            r_n[:n + 1] = self.rewards[index:episode_last_index + 1]
            return (self.observations[index], self.actions[index],
                    r_n, self.observations[index + 1],
                    self.observations[index],
                    1, 0)

    def __len__(self):
        return len(self.observations)

    def discrete_actions(self, discrete_actions):
        self.actions = np.expand_dims(self.actions, axis=1)
        discrete_actions = np.expand_dims(discrete_actions, axis=0)
        diff = np.abs(self.actions - discrete_actions)
        total_diff = np.sum(diff, axis=-1)
        self.actions = np.argmin(total_diff, axis=-1).astype('int8')


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
                 gamma=0.99,
                 dqfd_n_step=3,
                 margin_fn=0.8,
                 lambda_nstep=1.0,
                 lambda_supervise=1.0,
                 lambdaL2=1e-5,
                 double_dqn=True,
                 expert_dataset=None,
                 tensorboard_log=None
                 ):
        torch.manual_seed(seed)
        self.criterion = nn.SmoothL1Loss()
        self.env = env
        self.student = student
        self.q_net = student.policy.q_net.to(device)
        self.q_net_target = student.policy.q_net.to(device)
        self.device = device
        self.log_interval = log_interval
        self.writer = SummaryWriter(tensorboard_log)
        self.splits = KFold(n_splits=kfold, shuffle=True, random_state=42)
        self.dataset = expert_dataset
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size

        # DQN from Demonstration (DQfD) related parameters
        self.gamma = gamma
        self.n_gamma = gamma ** dqfd_n_step
        gammas = []
        for i in range(dqfd_n_step):
            gammas.append(gamma ** i)
        self.gammas = torch.tensor(gammas, device=device)
        self.margin_fn = margin_fn
        self.lambda_nstep = lambda_nstep
        self.lambda_supervise = lambda_supervise
        self.double_dqn = True

        # Define an Optimizer and a learning rate schedule.
        self.optimizer = optim.Adadelta(self.q_net.parameters(), lr=learning_rate, weight_decay=lambdaL2)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=scheduler_gamma)

    def update_target_net(self):
        self.q_net_target.load_state_dict(self.q_net.state_dict())

    def cal_target_q_values(self, s_1, s_n, r_arr, before_done, present_qn):
        """
        compute: target Q(s,a)
        Q(s,a) = r + gamma * maxQ(s_1, a)
        Q(s,a) = r + gamma * r_1 + ... gamma**(n-1) * r_(n-1) + gamma**n * maxQ(s_n, a)
        """
        '''
        1-step TD
        '''
        # Compute the next Q-values using the target network
        next_q_values = self.q_net_target(s_1)
        # Follow greedy policy: use the one with the highest value
        if self.double_dqn:
            # use current model to select the action with maximal q value
            max_actions = torch.argmax(self.q_net(s_1), dim=1)
            # evaluate q value of that action using fixed target network
            next_q_values = torch.gather(next_q_values, dim=1, index=max_actions.unsqueeze(-1))
        else:
            # Follow greedy policy: use the one with the highest value
            next_q_values, _ = next_q_values.max(dim=1)
        # Avoid potential broadcast issue
        next_q_values = next_q_values.reshape(-1, 1)
        # 1-step TD target
        target_q_values = r_arr[:, 0].unsqueeze(-1) + self.gamma * torch.mul(before_done.unsqueeze(-1), next_q_values)

        '''
        n-step TD
        '''
        # Compute the next Q-values using the target network
        next_n_q_values = self.q_net_target(s_n)
        # Follow greedy policy: use the one with the highest value
        if self.double_dqn:
            # use current model to select the action with maximal q value
            max_actions = torch.argmax(self.q_net(s_n), dim=1)
            # evaluate q value of that action using fixed target network
            next_n_q_values = torch.gather(next_n_q_values, dim=1, index=max_actions.unsqueeze(-1))
        else:
            # Follow greedy policy: use the one with the highest value
            next_n_q_values, _ = next_q_values.max(dim=1)
        # Avoid potential broadcast issue
        next_n_q_values = next_n_q_values.reshape(-1, 1)
        # n-step TD target
        target_n_q_values = torch.matmul(r_arr.type(torch.float32), self.gammas).unsqueeze(-1) + \
                            self.n_gamma * torch.mul(present_qn.unsqueeze(-1), next_n_q_values)

        return target_q_values, target_n_q_values

    def train(self, epoch, train_loader, target_update_batch):
        self.q_net.train()
        self.q_net_target.eval()

        for batch_idx, (s_0, a, r_arr, s_1, s_n, before_done, present_qn) in enumerate(train_loader):
            # update Q target network
            if batch_idx % target_update_batch == 0:
                self.update_target_net()

            s_0, a, r_arr, s_1, s_n, before_done, present_qn = s_0.to(self.device), a.to(self.device), \
                                                               r_arr.to(self.device), s_1.to(self.device), \
                                                               s_n.to(self.device), before_done.to(self.device), \
                                                               present_qn.to(self.device)
            self.optimizer.zero_grad()

            # calculate target Q-values
            with torch.no_grad():
                target_q_values, target_n_q_values = self.cal_target_q_values(s_1, s_n, r_arr, before_done, present_qn)

            # Get current Q-values estimates
            q_s = self.q_net(s_0)
            # Retrieve the q-values for the actions from the dataset
            q_s_a = torch.gather(q_s, dim=1, index=a.long().unsqueeze(-1))
            # Avoid potential broadcast issue
            q_s_a = q_s_a.reshape(-1, 1)

            '''
            Supervised loss
            '''
            l_margin = self.margin_fn * torch.ones_like(q_s, device=self.device)
            l_margin.scatter_(1, a.long().unsqueeze(-1), torch.zeros_like(q_s, device=self.device))
            J_E = (torch.max(q_s + l_margin.detach(), dim=1)[0].unsqueeze(-1) - q_s_a).mean()

            '''
            Q learning loss
            '''
            # Compute Huber loss (less sensitive to outliers)
            J_DQ = self.criterion(q_s_a, target_q_values.detach())
            J_N = self.criterion(q_s_a, target_n_q_values.detach())

            '''
            Total loss
            '''
            loss = J_DQ + self.lambda_nstep * J_N + self.lambda_supervise * J_E

            loss.backward()
            self.optimizer.step()
            if batch_idx % self.log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(s_0),
                        len(train_loader.sampler),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
                # print(
                #     "J_DQ: {:.6f}   J_N: {:.6f}   J_E: {:.6f}".format(
                #         J_DQ.item(), J_N.item(), J_E.item()
                #     )
                # )
                total_step = ((epoch - 1) * len(train_loader.sampler)) + (batch_idx * len(s_0))
                self.writer.add_scalar('training loss (J_DQ+J_N+J_E)',
                                       scalar_value=loss.item(),
                                       global_step=total_step)
                self.writer.add_scalar('J_DQ training loss',
                                       scalar_value=J_DQ.item(),
                                       global_step=total_step)
                self.writer.add_scalar('J_N training loss',
                                       scalar_value=J_N.item(),
                                       global_step=total_step)
                self.writer.add_scalar('J_E training loss',
                                       scalar_value=J_E.item(),
                                       global_step=total_step)

    def test(self, epoch, test_loader):
        self.update_target_net()
        self.q_net.eval()
        self.q_net_target.eval()

        total_test_loss = 0.0
        total_J_DQ = 0.0
        total_J_N = 0.0
        total_J_E = 0.0
        with torch.no_grad():
            for s_0, a, r_arr, s_1, s_n, before_done, present_qn in test_loader:
                s_0, a, r_arr, s_1, s_n, before_done, present_qn = s_0.to(self.device), a.to(self.device), \
                                                                   r_arr.to(self.device), s_1.to(self.device), \
                                                                   s_n.to(self.device), before_done.to(self.device), \
                                                                   present_qn.to(self.device)

                # calculate target Q-values
                target_q_values, target_n_q_values = self.cal_target_q_values(s_1, s_n, r_arr, before_done, present_qn)

                # Get current Q-values estimates
                q_s = self.q_net(s_0)
                # Retrieve the q-values for the actions from the dataset
                q_s_a = torch.gather(q_s, dim=1, index=a.long().unsqueeze(-1))
                # Avoid potential broadcast issue
                q_s_a = q_s_a.reshape(-1, 1)

                '''
                Supervised loss
                '''
                l_margin = self.margin_fn * torch.ones_like(q_s, device=self.device)
                l_margin.scatter_(1, a.long().unsqueeze(-1), torch.zeros_like(q_s, device=self.device))
                J_E = (torch.max(q_s + l_margin, dim=1)[0].unsqueeze(-1) - q_s_a).mean()

                '''
                Q learning loss
                '''
                # Compute Huber loss (less sensitive to outliers)
                J_DQ = self.criterion(q_s_a, target_q_values)
                J_N = self.criterion(q_s_a, target_n_q_values)

                '''
                Total loss
                '''
                test_loss = J_DQ + self.lambda_nstep * J_N + self.lambda_supervise * J_E
                total_test_loss = total_test_loss + test_loss.item()
                total_J_DQ += J_DQ.item()
                total_J_N += J_N.item()
                total_J_E += J_E.item()
        print(f"Test set: Total loss: {total_test_loss:.4f}")
        self.writer.add_scalar('total test loss',
                               scalar_value=total_test_loss,
                               global_step=epoch)
        self.writer.add_scalar('J_DQ test loss',
                               scalar_value=total_J_DQ,
                               global_step=epoch)
        self.writer.add_scalar('J_N test loss',
                               scalar_value=total_J_N,
                               global_step=epoch)
        self.writer.add_scalar('J_E test loss',
                               scalar_value=total_J_E,
                               global_step=epoch)

    def learn(self, epochs, save_interval, checkpoint_dir, previous_epoch=0, target_update_batch=500):
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
                self.train(total_epoch, train_loader, target_update_batch)
                self.test(total_epoch, test_loader)
                self.scheduler.step()
                if total_epoch % save_interval == 0:
                    model_path = os.path.join(checkpoint_dir, 'best_dict_{}.pth'.format(total_epoch))
                    print('Saving best_model_{} to {}'.format(total_epoch, model_path))
                    torch.save(self.q_net.state_dict(), model_path)
            self.writer.close()


def collect_dataset():
    env = CrowdSimRaw()
    env.configure(config)
    env.setup(seed=100, num_of_env=1, ax=None)

    num_interactions = 20000
    # num_interactions = int(2e6)

    DATASET_NAME = 'DQN_BC_dataset_nofov_{}'.format(num_interactions)

    # record the start index (step) and final index (step) of every episode
    episode_record = []

    if isinstance(env.action_space, gym.spaces.Box):
        expert_observations = np.empty((num_interactions,) + env.observation_space.shape)
        expert_actions = np.empty((num_interactions,) + (env.action_space.shape[0],))
        expert_rewards = np.empty((num_interactions,))
        expert_episodes = np.empty((num_interactions,), dtype='int')
    else:
        expert_observations = np.empty((num_interactions,) + env.observation_space.shape)
        expert_actions = np.empty((num_interactions,) + env.action_space.shape)
        expert_rewards = np.empty((num_interactions,))
        expert_episodes = np.empty((num_interactions,), dtype='int')

    episode = 0
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
            episode_record.append([previous_i, i])
            episode += 1
            previous_i = i + 1
            obs = env.reset()

    # exclude the steps (at the end) that isn't a complete episode
    last_step = episode_record[-1][-1]

    np.savez_compressed(
        DATASET_NAME,
        expert_actions=expert_actions[:last_step + 1],
        expert_observations=expert_observations[:last_step + 1],
        expert_rewards=expert_rewards[:last_step + 1],
        expert_episodes=expert_episodes[:last_step + 1],
        episode_record=np.array(episode_record, dtype='int')
    )


def train_bc():
    # Discrete Actions
    U_A = [-1.0, -0.5, 0.0, 0.5, 1.0]
    u_a = np.array(U_A)
    Y, X = np.meshgrid(u_a, u_a)
    discrete_actions = np.stack((X, Y), axis=-1)
    discrete_actions = discrete_actions.reshape((-1, 2))

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
                                   np_data['episode_record'])
    expert_dataset.discrete_actions(discrete_actions)
    print("expert_dataset: ", len(expert_dataset))

    policy_kwargs = dict(
        features_extractor_class=ApfFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=512),
    )
    model = DQN("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, device='cuda', tensorboard_log=LOG_DIR,
                batch_size=128)
    # print(model.policy.q_net)

    # OLD_MODEL_PATH = CHECKPOINT_DIR + 'best_dict_17.pth'
    # old_dict = torch.load(OLD_MODEL_PATH)
    # model.policy.q_net.load_state_dict(old_dict)
    # model.policy.q_net_target.load_state_dict(old_dict)

    # for name, param in model.policy.q_net.named_parameters():
    #     if param.requires_grad:
    #         print("GRAD: {}".format(name))
    #         print(param)
    #     else:
    #         print("NO GRAD: {}".format(name))

    bc = BehaviourCloning(student=model,
                          env=env,
                          batch_size=16,
                          scheduler_gamma=0.7,
                          learning_rate=0.00233,
                          log_interval=100,
                          device='cuda',
                          seed=1000,
                          test_batch_size=16,
                          expert_dataset=expert_dataset,
                          tensorboard_log=LOG_DIR)

    bc.learn(epochs=20, save_interval=1, checkpoint_dir=CHECKPOINT_DIR,
             previous_epoch=0, target_update_batch=50)


def main():
    training = True
    if not training:
        collect_dataset()
    else:
        train_bc()


if __name__ == '__main__':
    main()
