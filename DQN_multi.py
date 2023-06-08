import torch

import gym
from gym.spaces import Discrete
import numpy as np
from crowd_sim.envs.crowd_sim_sgan import CrowdSimSgan
from crowd_sim.envs.crowd_sim_no_pred import CrowdSimNoPred
import time
import os

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import DQN

from arguments import get_args
from crowd_nav.configs.config import Config

import sys
sys.path.append('/home/koksyuen/python_project/sgan')
# from predictor import socialGAN
from predictor2 import get_generator, socialGAN

# traj_predictor = socialGAN(model_path='/home/koksyuen/python_project/sgan/models/sgan-p-models/eth_8_model.pt')
# print('general obj id: {}'.format(id(traj_predictor)))

'''
0: Vx=0, Vy=0
1: Vx=1, Vy=0
2: Vx=0, Vy=1
3: Vx=-1, Vy=0
4: Vx=0, Vy=-1
5: Vx=1, Vy=1
6: Vx=-1, Vy=-1
7: Vx=1, Vy=-1
8: Vx=-1, Vy=1
'''
discrete_actions = [np.array([0, 0]), np.array([1, 0]),
                    np.array([0, 1]), np.array([-1, 0]),
                    np.array([0, -1]), np.array([1, 1]),
                    np.array([-1, -1]), np.array([1, -1]),
                    np.array([-1, 1])]


class DiscreteActions(gym.ActionWrapper):
    def __init__(self, env, disc_to_cont):
        super().__init__(env)
        self.disc_to_cont = disc_to_cont
        self.action_space = Discrete(len(disc_to_cont))

    def action(self, act):
        return self.disc_to_cont[act]


class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, offset=0, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.offset = offset

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls + self.offset))
            self.model.save(model_path)

        return True


def make_env(seed, rank, env_config, traj_predictor, envNum=1):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        # print('first received object: {}'.format(id(traj_predictor)))
        # env = CrowdSimNoPred()
        env = CrowdSimSgan()
        # use a seed for reproducibility
        # Important: use a different seed for each environment
        # otherwise they would generate the same experiences
        env.seed(seed + rank)
        env.setup(seed=seed + rank, num_of_env=envNum, traj_predictor=traj_predictor)
        env.configure(env_config)
        env = DiscreteActions(env, discrete_actions)
        return env

    return _init


def main():
    config = Config()

    num_cpu = 1  # Number of processes/threads to use
    seed = 0

    CHECKPOINT_DIR = './train/DQN_SGAN/'
    LOG_DIR = './logs/SGAN/'

    checkpoint = torch.load('/home/koksyuen/python_project/sgan/models/sgan-p-models/eth_8_model.pt')
    generator = get_generator(checkpoint)
    print('general generator id: {}'.format(id(generator)))

    traj_predictor = socialGAN(generator)

    # multi_env = [make_env(seed, i, config, traj_predictor, num_cpu) for i in range(num_cpu)]
    envs = SubprocVecEnv([make_env(seed, i, config, traj_predictor, num_cpu) for i in range(num_cpu)])

    callback = TrainAndLoggingCallback(check_freq=100000, save_path=CHECKPOINT_DIR)

    model = DQN('MultiInputPolicy', envs, verbose=1, tensorboard_log=LOG_DIR)

    # model.learn(total_timesteps=2000000, callback=callback)
    # model.save('latestmodel')


if __name__ == '__main__':
    main()
