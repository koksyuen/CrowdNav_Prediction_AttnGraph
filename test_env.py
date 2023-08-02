import gym
from gym.spaces import Discrete
import numpy as np
from crowd_sim.envs.crowd_sim_sgan import CrowdSimSgan
from crowd_sim.envs.crowd_sim_sgan_apf import CrowdSimSganApf
from crowd_sim.envs.crowd_sim_no_pred import CrowdSimNoPred
from crowd_sim.envs.crowd_sim_raw import CrowdSimRaw
from sb3.feature_extractor import Preprocessor, ApfFeaturesExtractor
import time
import os
import matplotlib.pyplot as plt
import torch

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
# from stable_baselines3 import DQN, PPO, A2C
from sb3.dqn.dqn import DQN

from arguments import get_args
from crowd_nav.configs.config import Config

# '''
# 0: Vx=0, Vy=0
# 1: Vx=1, Vy=0
# 2: Vx=0, Vy=1
# 3: Vx=-1, Vy=0
# 4: Vx=0, Vy=-1
# 5: Vx=1, Vy=1
# 6: Vx=-1, Vy=-1
# 7: Vx=1, Vy=-1
# 8: Vx=-1, Vy=1
# '''
# discrete_actions = [np.array([0, 0]), np.array([1, 0]),
#                     np.array([0, 1]), np.array([-1, 0]),
#                     np.array([0, -1]), np.array([1, 1]),
#                     np.array([-1, -1]), np.array([1, -1]),
#                     np.array([-1, 1])]

U_A = [-1.0, -0.5, 0.0, 0.5, 1.0]
u_a = np.array(U_A)
Y, X = np.meshgrid(u_a, u_a)
discrete_actions = np.stack((X, Y), axis=-1)
discrete_actions = discrete_actions.reshape((-1, 2))
print(discrete_actions)


class DiscreteActions(gym.ActionWrapper):
    def __init__(self, env, disc_to_cont):
        super().__init__(env)
        self.disc_to_cont = disc_to_cont
        self.action_space = Discrete(len(disc_to_cont))

    def action(self, act):
        return self.disc_to_cont[act]


class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True

def make_env(seed, rank, env_config, envNum=1):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        env = CrowdSimRaw()
        # use a seed for reproducibility
        # Important: use a different seed for each environment
        # otherwise they would generate the same experiences
        env.configure(env_config)
        env.seed(seed + rank)
        env.setup(seed=seed + rank, num_of_env=envNum)
        env = Monitor(env)
        return env

    return _init

def make_discrete_env(seed, rank, env_config, envNum=1):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        env = CrowdSimRaw()
        # use a seed for reproducibility
        # Important: use a different seed for each environment
        # otherwise they would generate the same experiences
        env.configure(env_config)
        env.seed(seed + rank)
        env.setup(seed=seed + rank, num_of_env=envNum)
        env = DiscreteActions(env, discrete_actions)
        env = Monitor(env)
        return env

    return _init


def main():
    plt.figure(1, figsize=(7, 7))
    ax1 = plt.subplot()
    ax1.set_xlim(-10, 10)
    ax1.set_ylim(-10, 10)
    ax1.set_xlabel('x(m)', fontsize=16)
    ax1.set_ylabel('y(m)', fontsize=16)

    # plt.figure(2)
    # ax2 = plt.subplot()
    # ax2.set_xlabel('x (grid)', fontsize=16)
    # ax2.set_ylabel('y (grid)', fontsize=16)

    config = Config()

    env = CrowdSimRaw()
    env.configure(config)
    env.setup(seed=45000, num_of_env=1, ax=ax1)

    denv = DiscreteActions(env, discrete_actions)

    policy_kwargs = dict(
        features_extractor_class=ApfFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=512),
    )

    #DQfN
    PRETRAIN_MODEL_PATH = './train/DQN_BC_APF_RAW/best_dict_15.pth'
    policy_dict = torch.load(PRETRAIN_MODEL_PATH)
    print(policy_dict)
    model = DQN("CnnPolicy", denv, policy_kwargs=policy_kwargs, verbose=1, device='cuda', batch_size=32)
    model.policy.q_net.load_state_dict(policy_dict)
    model.policy.q_net_target.load_state_dict(policy_dict)

    # DQN
    # MODEL_PATH = './train/D3QN_RAW_APF3/best_model'
    # model = DQN.load(MODEL_PATH, denv)


    episodes = 10
    for episode in range(1, episodes + 1):
        obs = denv.reset()
        done = False
        score = 0
        avg_time = 0
        step = 0

        while not done:
            # plt.figure(1)
            env.render()
            # action = env.action_space.sample()
            # vx, vy = env.calculate_orca()
            # action = np.array([vx, vy])
            start_time = time.time()
            action_rl = model.predict(obs, deterministic=True)
            # print(action_rl[0])
            end_time = time.time()
            # print("action_shape".format(action_rl.shape))
            # print("vx: {}   vy: {}".format(action_rl[0], action_rl[1]))
            # obs, reward, done, info = env.step(action_rl)
            obs, reward, done, info = denv.step(action_rl[0])
            # plt.figure(2)
            # plt.imshow(np.rot90(obs.reshape(obs.shape[1], obs.shape[2]), -1), cmap='gray')
            # plt.pause(0.01)
            avg_time += (end_time - start_time)
            step += 1
            score += reward
        print('Episode:{} Score:{}'.format(episode, score))
        # print('average inference time ({} steps): {}s'.format(step, avg_time / step))
    env.close()


if __name__ == '__main__':
    main()
