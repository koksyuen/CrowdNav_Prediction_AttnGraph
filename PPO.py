import gym
import numpy as np
from crowd_sim.envs.crowd_sim_cl import CrowdSimCL
from sb3.feature_extractor import Preprocessor, ApfFeaturesExtractor
from torch.utils.tensorboard import SummaryWriter
import torch
import time
import os

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO

from arguments import get_args
from crowd_nav.configs.config import Config

def make_env(seed, rank, env_config, envNum=1):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        env = CrowdSimCL()
        # use a seed for reproducibility
        # Important: use a different seed for each environment
        # otherwise they would generate the same experiences
        env.configure(env_config)
        env.seed(seed + rank)
        env.setup(seed=seed + rank, num_of_env=envNum)
        env = Monitor(env)
        return env

    return _init


class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, start_step=0, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.offset = start_step

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls + self.offset))
            self.model.save(model_path)

        return True


def main():
    config = Config()
    num_cpu = 12  # Number of processes to use
    seed = 1
    venv = SubprocVecEnv([make_env(seed, i, config, num_cpu) for i in range(num_cpu)])

    # PRETRAIN_MODEL_PATH = './train/BC/best_dict_27.pth'
    # policy_dict = torch.load(PRETRAIN_MODEL_PATH)
    # print(policy_dict)

    CHECKPOINT_DIR = './train/PPO_NEW_APF/test/'
    LOG_DIR = './logs/PPO_NEW_APF/test/'

    # FIRST TIME TRAINING
    callback = TrainAndLoggingCallback(check_freq=int(1e5), save_path=CHECKPOINT_DIR)
    policy_kwargs = dict(
        features_extractor_class=ApfFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=512),
    )
    model = PPO("CnnPolicy", venv, policy_kwargs=policy_kwargs, verbose=1, learning_rate=0.001,
                device='cuda', tensorboard_log=LOG_DIR, batch_size=64, ent_coef=0.01)
    print(model.policy)
    # model.policy.load_state_dict(policy_dict)

    # CONTINUAL TRAINING
    # MODEL_PATH = './train/PPO_SGAN/best_model_4000000'
    # callback = TrainAndLoggingCallback(check_freq=100000, save_path=CHECKPOINT_DIR, start_step=4000000)
    # model = PPO.load(MODEL_PATH, env, tensorboard_log=LOG_DIR)

    model.learn(total_timesteps=int(1e7), callback=callback)
    model_path = os.path.join(CHECKPOINT_DIR, 'latest_model')
    model.save(model_path)


if __name__ == '__main__':
    main()