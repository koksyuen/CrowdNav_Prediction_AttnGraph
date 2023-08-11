import gym
from gym.spaces import Discrete
import numpy as np
from crowd_sim.envs.crowd_sim_goal import CrowdSimBasic
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
from stable_baselines3 import PPO

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
        env = CrowdSimBasic()
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

    plt.figure(3)
    ax3 = plt.subplot()
    ax3.set_xlabel('step', fontsize=16)
    ax3.set_ylabel('reward', fontsize=16)

    plt.figure(4)
    ax3 = plt.subplot()
    ax3.set_xlabel('step', fontsize=16)
    ax3.set_ylabel('cummulative_reward', fontsize=16)

    config = Config()

    env = CrowdSimBasic()
    env.configure(config)
    env.setup(seed=0, num_of_env=1, ax=ax1)

    denv = DiscreteActions(env, discrete_actions)
    MODEL_PATH = './train/D3QN_GOAL/POT/best_model.zip'
    model = DQN.load(MODEL_PATH, denv)

    # MODEL_PATH = './train/PPO_GOAL/POT/E01/best_model_500000.zip'
    # model = PPO.load(MODEL_PATH, env)

    episodes = 10
    for episode in range(1, episodes + 1):
        # obs = env.reset()
        obs = denv.reset()
        done = False
        score = 0
        avg_time = 0
        step = 0
        rewards = []
        cum_rewards = []

        while not done:
            # print(obs)
            plt.figure(1)
            denv.render()
            # env.render()
            # action = denv.action_space.sample()
            # vx, vy = env.calculate_orca()
            # action = np.array([vx, vy])
            # start_time = time.time()
            action_rl = model.predict(obs, deterministic=True)
            # print(action_rl[0])
            # end_time = time.time()
            # print("action_shape".format(action_rl.shape))
            # print("vx: {}   vy: {}".format(action_rl[0], action_rl[1]))
            # obs, reward, done, info = env.step(action_rl)
            obs, reward, done, info = denv.step(action_rl[0])
            # obs, reward, done, info = env.step(action_rl[0])
            # plt.figure(2)
            # plt.imshow(np.rot90(obs.reshape(obs.shape[1], obs.shape[2]), -1), cmap='gray')
            # plt.pause(0.01)
            # avg_time += (end_time - start_time)
            step += 1
            score += reward
            rewards.append(reward)
            cum_rewards.append(score)
            # rewards.append(score)
        print('Episode:{} Score:{}'.format(episode, score))
        plt.figure(3)
        plt.plot(rewards)
        plt.pause(0.01)
        plt.figure(4)
        plt.plot(cum_rewards)
        plt.pause(0.01)
        # print('average inference time ({} steps): {}s'.format(step, avg_time / step))
    env.close()


if __name__ == '__main__':
    main()
