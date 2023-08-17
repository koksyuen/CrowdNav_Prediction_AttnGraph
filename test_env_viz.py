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


def main():
    plt.figure(1, figsize=(7, 7))
    ax1 = plt.subplot()
    ax1.set_xlim(-10, 10)
    ax1.set_ylim(-10, 10)
    ax1.set_xlabel('x(m)', fontsize=16)
    ax1.set_ylabel('y(m)', fontsize=16)

    config = Config()

    env = CrowdSimSganApf()
    env.configure(config)
    env.setup(seed=int(1e8), num_of_env=1, ax=ax1)

    episodes = 10
    for episode in range(1, episodes + 1):
        obs = env.reset()
        done = False
        score = 0
        avg_time = 0
        step = 0
        rewards = []

        while not done:
            plt.figure(1)
            env.render()
            vx, vy = env.calculate_orca()
            action = np.array([vx, vy])
            start_time = time.time()
            obs, reward, done, info = env.step(action)
            end_time = time.time()
            avg_time += (end_time - start_time)
            step += 1
            score += reward
            rewards.append(reward)
        print('Episode:{} Score:{}'.format(episode, score))
        plt.figure(2)
        plt.plot(rewards)
        plt.pause(0.01)
        # print('average inference time ({} steps): {}s'.format(step, avg_time / step))
    env.close()


if __name__ == '__main__':
    main()
