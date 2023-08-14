import gym
from gym.spaces import Discrete
import numpy as np
from crowd_sim.envs.crowd_sim_cl import CrowdSimCL
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
    ax4 = plt.subplot()
    ax4.set_xlabel('step', fontsize=16)
    ax4.set_ylabel('cummulative_reward', fontsize=16)

    plt.figure(5)
    ax5 = plt.subplot()
    ax5.set_xlabel('step', fontsize=16)
    ax5.set_ylabel('comfort distance', fontsize=16)

    config = Config()
    human_num = config.sim.human_num

    env = CrowdSimCL()
    env.configure(config)
    env.setup(seed=90000, num_of_env=1, ax=ax1)

    # denv = DiscreteActions(env, discrete_actions)
    # MODEL_PATH = './train/D3QN_GOAL/POT/best_model.zip'
    # model = DQN.load(MODEL_PATH, denv)

    MODEL_PATH = './train/PPO_NEW_APF/H10/E001_L0003/best_model.zip'
    old_model = PPO.load(MODEL_PATH)
    policy_dict = old_model.policy.state_dict()

    # FIRST TIME TRAINING
    policy_kwargs = dict(
        features_extractor_class=ApfFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=512),
    )
    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, learning_rate=0.001,
                device='cuda', batch_size=64, ent_coef=0.01)
    model.policy.load_state_dict(policy_dict)

    episodes = 10
    for episode in range(1, episodes + 1):
        obs = env.reset()
        done = False
        score = 0
        avg_time = 0
        step = 0
        rewards = []
        cum_rewards = []
        comfort_distances = obs[-2, :human_num, 0]
        comfort_distances = comfort_distances.reshape(comfort_distances.shape[0], 1)

        while not done:
            # print(obs)
            plt.figure(1)
            env.render()
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
            obs, reward, done, info = env.step(action_rl[0])
            x = obs[-2, :human_num, 0]
            x = x.reshape(x.shape[0], 1)
            comfort_distances = np.concatenate((comfort_distances, x), axis=1)
            # obs, reward, done, info = env.step(action_rl[0])
            # plt.figure(2)
            # plt.imshow(np.rot90(obs.reshape(obs.shape[1], obs.shape[2]), -1), cmap='gray')
            # plt.pause(0.01)
            # avg_time += (end_time - start_time)
            if info['type'] == 'discomfort':
                print('discomfort discount: {}'.format(info['discomfort']))
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
        plt.figure(5)
        plt.clf()
        for i in range(human_num):
            plt.plot(comfort_distances[i], label='Human {}'.format(i + 1))
        plt.legend()
        plt.pause(0.01)
        # print('average inference time ({} steps): {}s'.format(step, avg_time / step))
    env.close()


if __name__ == '__main__':
    main()
