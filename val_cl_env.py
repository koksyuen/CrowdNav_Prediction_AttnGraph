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
    env.setup(seed=int(2e7), num_of_env=1, ax=None, emotion=False)

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
    result = []
    success = []

    episodes = 3000
    for episode in tqdm(range(episodes)):
        goal = False
        obs = env.reset()
        # obs = env.reset(phase='test')
        done = False
        discomfort_reward = 0

        while not done:
            action_rl = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action_rl[0])

            if info['type'] == 'discomfort':
                # print('discomfort discount: {}'.format(info['discomfort']))
                discomfort_reward += info['discomfort']

            if info['type'] == 'goal':
                reach_time = info['goal']
                result.append([reach_time, discomfort_reward])
                goal = True

        success.append(goal)

    result_np = np.array(result)
    np.save('./result/H20.npy', result_np)
    success_np = np.array(success)
    np.save('./result/Success_H20.npy', success_np)
    env.close()


if __name__ == '__main__':
    main()