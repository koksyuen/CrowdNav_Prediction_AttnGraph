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
from tqdm import tqdm

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


def main():
    config = Config()

    env = CrowdSimRaw()
    env.configure(config)
    env.setup(seed=45000, num_of_env=1, ax=None)

    denv = DiscreteActions(env, discrete_actions)

    policy_kwargs = dict(
        features_extractor_class=ApfFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=512),
    )

    # DQfN
    PRETRAIN_MODEL_PATH = './train/DQN_BC_APF_RAW/best_dict_15.pth'
    policy_dict = torch.load(PRETRAIN_MODEL_PATH)
    print(policy_dict)
    model = DQN("CnnPolicy", denv, learning_rate=0.0001, policy_kwargs=policy_kwargs,
                exploration_fraction=1.0, exploration_initial_eps=0.5, exploration_final_eps=0.5,
                verbose=1, device='cuda', batch_size=512)
    model.policy.q_net.load_state_dict(policy_dict)
    model.policy.q_net_target.load_state_dict(policy_dict)

    # DQN
    # MODEL_PATH = './train/D3QN_RAW_APF3/best_model'
    # model = DQN.load(MODEL_PATH, denv)

    num_interactions = 1000
    obs = denv.reset()
    potential_reward = []
    discomfort_reward = []

    for i in tqdm(range(num_interactions)):
        action_rl = model.predict(obs, deterministic=False)
        obs, reward, done, info = denv.step(action_rl[0])
        if info['info']['type'] == 'potential':
            potential_reward.append(info['info']['potential'])
        if info['info']['type'] == 'discomfort':
            discomfort_reward.append(info['info']['discomfort'])
        if done:
            obs = env.reset()
    env.close()

    np.savez_compressed('reward_record.npz', discomfort_reward=discomfort_reward, potential_reward=potential_reward)


if __name__ == '__main__':
    main()
