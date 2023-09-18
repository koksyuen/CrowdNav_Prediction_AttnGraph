import gym
from gym.spaces import Discrete
import numpy as np
from crowd_sim.envs.crowd_sim_cl import CrowdSimCL
from sb3.feature_extractor import Preprocessor, ApfFeaturesExtractor
from tqdm import tqdm
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
    config = Config()

    env = CrowdSimCL()
    env.configure(config)
    env.setup(seed=0, num_of_env=1, ax=None, emotion=False)

    MODEL_PATH = 'train/PPO_FINAL/H20_D30_G075/E001_L0003/latest_model.zip'
    model_emotionless = PPO.load(MODEL_PATH)

    MODEL_PATH_EMO = 'train/PPO_FINAL/H20_EMO_D30_G075/E001_L0003/latest_model.zip'
    model_emotion = PPO.load(MODEL_PATH_EMO)

    # FIRST TIME TRAINING
    # policy_kwargs = dict(
    #     features_extractor_class=ApfFeaturesExtractor,
    #     features_extractor_kwargs=dict(features_dim=512),
    # )
    # model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, learning_rate=0.001,
    #             device='cuda', batch_size=64, ent_coef=0.01)
    # model.policy.load_state_dict(policy_dict)

    max_reward_diff = 0
    best_test_case = 0
    verification_num = 10
    episodes = 2000
    test_case = int(3e7)
    for episode in tqdm(range(episodes)):

        goals_emotion = []
        goals_emotionless = []
        avg_discomfort_reward_emotion = 0
        avg_discomfort_reward_emotionless = 0

        for i in range(verification_num):

            # with emotion
            tem_goal_emotion = False
            obs = env.reset(test_case=test_case, emotion=True)
            done = False
            discomfort_reward_emotion = 0

            while not done:
                action_rl = model_emotion.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action_rl[0])

                if info['type'] == 'discomfort':
                    # print('discomfort discount: {}'.format(info['discomfort']))
                    discomfort_reward_emotion += info['discomfort']

                if info['type'] == 'goal':
                    tem_goal_emotion = True

            goals_emotion.append(tem_goal_emotion)
            avg_discomfort_reward_emotion = (avg_discomfort_reward_emotion + discomfort_reward_emotion)/2

            # without emotion
            tem_goal_emotionless = False
            obs = env.reset(test_case=test_case, emotion=False)
            done = False
            discomfort_reward_emotionless = 0

            while not done:
                action_rl = model_emotionless.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action_rl[0])

                if info['type'] == 'discomfort':
                    # print('discomfort discount: {}'.format(info['discomfort']))
                    discomfort_reward_emotionless += info['discomfort']

                if info['type'] == 'goal':
                    tem_goal_emotionless = True

            goals_emotionless.append(tem_goal_emotionless)
            avg_discomfort_reward_emotionless = (avg_discomfort_reward_emotionless + discomfort_reward_emotionless) / 2

        goal_emotion = all(goals_emotion)
        goal_emotionless = all(goals_emotionless)

        if goal_emotion and goal_emotionless:
            if (avg_discomfort_reward_emotion - avg_discomfort_reward_emotionless) > max_reward_diff:
                max_reward_diff = avg_discomfort_reward_emotion - avg_discomfort_reward_emotionless
                best_test_case = test_case
                print('{} test case: {}'.format(best_test_case, max_reward_diff))

        test_case += 1

    print('best test case: {}'.format(best_test_case))


if __name__ == '__main__':
    main()
