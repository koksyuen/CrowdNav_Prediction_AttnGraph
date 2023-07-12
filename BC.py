from crowd_sim.envs.crowd_sim_sgan import CrowdSimSgan
from crowd_sim.envs.crowd_sim_sgan_apf import CrowdSimSganApf
from crowd_sim.envs.crowd_sim_no_pred import CrowdSimNoPred
from crowd_sim.envs.crowd_sim_raw import CrowdSimRaw
import gym
import time
from tqdm import tqdm
import numpy as np

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO, A2C

from arguments import get_args
from crowd_nav.configs.config import Config


def main():
    config = Config()

    env = CrowdSimRaw()
    env.configure(config)
    env.setup(seed=0, num_of_env=1, ax=None)

    num_interactions = int(1e6)

    if isinstance(env.action_space, gym.spaces.Box):
        expert_observations = np.empty((num_interactions,) + env.observation_space.shape)
        expert_actions = np.empty((num_interactions,) + (env.action_space.shape[0],))

    else:
        expert_observations = np.empty((num_interactions,) + env.observation_space.shape)
        expert_actions = np.empty((num_interactions,) + env.action_space.shape)

    obs = env.reset()

    for i in tqdm(range(num_interactions)):
        env.render()
        action = np.array(env.calculate_orca())
        expert_observations[i] = obs
        expert_actions[i] = action
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()

    np.savez_compressed(
        ".train/BC/dataset",
        expert_actions=expert_actions,
        expert_observations=expert_observations,
    )


if __name__ == '__main__':
    main()
