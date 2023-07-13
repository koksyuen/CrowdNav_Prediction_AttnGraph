import gym
from gym.spaces import Discrete
import numpy as np
from crowd_sim.envs.crowd_sim_sgan import CrowdSimSgan
from crowd_sim.envs.crowd_sim_no_pred import CrowdSimNoPred
from crowd_sim.envs.crowd_sim_sgan_apf import CrowdSimSganApf
from crowd_sim.envs.crowd_sim_raw import CrowdSimRaw
from sb3.feature_extractor import Preprocessor, ApfFeaturesExtractor
from torch.utils.tensorboard import SummaryWriter
import torch
import time
import os

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
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
        env = CrowdSimRaw()
        # use a seed for reproducibility
        # Important: use a different seed for each environment
        # otherwise they would generate the same experiences
        env.configure(env_config)
        env.seed(seed + rank)
        env.setup(seed=seed + rank, num_of_env=envNum)
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
    num_cpu = 10  # Number of processes to use
    seed = 100
    venv = SubprocVecEnv([make_env(seed, i, config, num_cpu) for i in range(num_cpu)])

    obs = venv.reset()

    writer = SummaryWriter("./logs/ppo_apf_raw")

    CHECKPOINT_DIR = './train/PPO_RAW/'
    LOG_DIR = './logs/APF_RAW/'

    # FIRST TIME TRAINING
    callback = TrainAndLoggingCallback(check_freq=5000, save_path=CHECKPOINT_DIR)
    policy_kwargs = dict(
        features_extractor_class=ApfFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=512),
    )
    model = PPO("CnnPolicy", venv, policy_kwargs=policy_kwargs, verbose=1, device='cuda', tensorboard_log=LOG_DIR, batch_size=64)

    observations = torch.from_numpy(obs).cuda().float()
    writer.add_graph(model.policy, observations)

    # CONTINUAL TRAINING
    # MODEL_PATH = './train/PPO_SGAN/best_model_4000000'
    # callback = TrainAndLoggingCallback(check_freq=100000, save_path=CHECKPOINT_DIR, start_step=4000000)
    # model = PPO.load(MODEL_PATH, env, tensorboard_log=LOG_DIR)

    model.learn(total_timesteps=int(100000), callback=callback)
    model.save('testingmodel')


if __name__ == '__main__':
    main()