import gym
from gym.spaces import Discrete
import numpy as np
from crowd_sim.envs.crowd_sim_sgan import CrowdSimSgan
from crowd_sim.envs.crowd_sim_no_pred import CrowdSimNoPred
from crowd_sim.envs.crowd_sim_sgan_apf import CrowdSimSganApf
import time
import os

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO

from arguments import get_args
from crowd_nav.configs.config import Config


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

    env = CrowdSimSganApf()
    env.configure(config)
    env.setup(seed=0, num_of_env=1)

    CHECKPOINT_DIR = './train/PPO_APF/'
    LOG_DIR = './logs/APF/'

    # FIRST TIME TRAINING
    callback = TrainAndLoggingCallback(check_freq=500000, save_path=CHECKPOINT_DIR)
    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR)

    # CONTINUAL TRAINING
    # MODEL_PATH = './train/PPO_SGAN/best_model_4000000'
    # callback = TrainAndLoggingCallback(check_freq=100000, save_path=CHECKPOINT_DIR, start_step=4000000)
    # model = PPO.load(MODEL_PATH, env, tensorboard_log=LOG_DIR)

    # model.learn(total_timesteps=int(1e7), callback=callback)
    # model.save('latestmodel')


if __name__ == '__main__':
    main()