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
from sb3.dqn.dqn import DQN
# from stable_baselines3 import PPO
from sb3.ppo.ppo import PPO

from arguments import get_args
from crowd_nav.configs.config import Config


def main():
    plt.figure(1, figsize=(7, 7))
    ax1 = plt.subplot()
    ax1.set_xlim(-11, 11)
    ax1.set_ylim(-11, 11)
    ax1.set_xlabel('x(m)', fontsize=16)
    ax1.set_ylabel('y(m)', fontsize=16)

    # plt.figure(2)
    # ax2 = plt.subplot()
    # ax2.set_xlim(-5, 5)
    # ax2.set_ylim(-5, 5)
    # ax2.set_xlabel('x (m)', fontsize=16)
    # ax2.set_ylabel('y (m)', fontsize=16)

    # plt.figure(3)
    # ax3 = plt.subplot()
    # ax3.set_xlabel('step', fontsize=16)
    # ax3.set_ylabel('reward', fontsize=16)
    #
    # plt.figure(4)
    # ax4 = plt.subplot()
    # ax4.set_xlabel('step', fontsize=16)
    # ax4.set_ylabel('cummulative_reward', fontsize=16)
    #
    # plt.figure(5)
    # ax5 = plt.subplot()
    # ax5.set_xlabel('step', fontsize=16)
    # ax5.set_ylabel('comfort distance', fontsize=16)

    # fig6 = plt.figure(6, figsize=(8, 6))
    # ax6 = fig6.add_subplot(111, projection='3d')
    # ax6.set_xlim(-5, 5)
    # ax6.set_ylim(-5, 5)
    # ax6.set_zlim(0, 1)
    # ax6.set_xlabel('x (m)', fontsize=16)
    # ax6.set_ylabel('y (m)', fontsize=16)
    # ax6.set_zlabel('potential', fontsize=16)
    # x, y = np.meshgrid(np.arange(-50, 50),
    #                    np.arange(-50, 50))
    # x = x / 10
    # y = y / 10

    config = Config()
    human_num = config.sim.human_num

    env = CrowdSimCL()
    env.configure(config)
    env.setup(seed=0, num_of_env=1, ax=ax1, emotion=False)
    # env.setup(seed=0, num_of_env=1, ax=ax1, emotion=True)

    # MODEL_PATH = 'train/PPO_FINAL/H20_EMO_D30_G075/E001_L0003/latest_model.zip'
    MODEL_PATH = 'train/PPO_FINAL/H20_D30_G075/E001_L0003/latest_model.zip'
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
    print(model.policy)

    test_case = 30000138
    episodes = 1
    for episode in range(1, episodes + 1):
        # obs = env.reset(test_case=test_case, emotion=True)
        obs = env.reset(test_case=test_case, emotion=False)
        done = False
        score = 0
        avg_time = 0
        step = 0
        rewards = []
        cum_rewards = []
        comfort_distances = obs[-2, :human_num, 0]
        comfort_distances = comfort_distances.reshape(comfort_distances.shape[0], 1)

        while not done:
            # vx, vy = env.calculate_orca()
            # action = np.array([vx, vy])
            # start_time = time.time()
            # print(obs[-1, 0])
            action_rl = model.predict(obs, deterministic=True)
            apf, obs_traj, pred_traj = model.get_hidden_info()
            # map_name = './apf/{}.npy'.format(step)
            # np.save(map_name, apf)
            # end_time = time.time()
            # print("vx: {}   vy: {}".format(action_rl[0], action_rl[1]))

            # plt.figure(2)
            # print('max: {}, min: {}'.format(np.max(apf), np.min(apf)))
            # plt.imshow(np.rot90(apf, -1), cmap='gray', extent=[-5, 5, -5, 5])
            # plt.pause(0.001)

            # plt.figure(6)
            # plt.clf()
            # surf = ax6.plot_surface(x, y, apf, cmap='ocean', linewidth=0, antialiased=True)
            # cset = ax6.contour(x, y, apf, zdir='z', offset=np.min(apf), cmap='ocean')
            # fig6.colorbar(surf, shrink=0.75, aspect=30, pad=0.1)
            # plt.pause(0.001)

            plt.figure(1)
            env.render_traj(obs_traj=obs_traj, pred_traj=pred_traj)
            env.render()

            obs, reward, done, info = env.step(action_rl[0])
            # obs, reward, done, info = env.step(action)
            # obs, reward, done, info = env.step(np.array([0.0, 0.0]))

            x = obs[-2, :human_num, 0]
            x = x.reshape(x.shape[0], 1)
            comfort_distances = np.concatenate((comfort_distances, x), axis=1)

            # avg_time += (end_time - start_time)
            if info['type'] == 'discomfort':
                print('discomfort discount: {}'.format(info['discomfort']))
            step += 1
            score += reward
            rewards.append(reward)
            cum_rewards.append(score)
        result_record = env.report_recording()
        np.save('./test_case_result/15_20000068/H15.npy', result_record)
        print('Episode:{} Score:{}'.format(episode, score))

        # plt.figure(3)
        # plt.plot(rewards, label='Episode {}'.format(episode + 1))
        # plt.legend()
        # plt.pause(0.01)
        # plt.figure(4)
        # plt.plot(cum_rewards, label='Episode {}'.format(episode + 1))
        # plt.legend()
        # plt.pause(0.01)
        # plt.figure(5)
        # plt.clf()
        # for i in range(human_num):
        #     plt.plot(comfort_distances[i], label='Human {}'.format(i + 1))
        # plt.legend()
        # plt.pause(0.01)
        # print('average inference time ({} steps): {}s'.format(step, avg_time / step))
    env.close()


if __name__ == '__main__':
    main()
    # x = [[1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
    # y = np.array(x)
    # plt.figure(2)
    # plt.imshow(y, cmap='gray')
    # plt.show()
