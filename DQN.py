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

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True


def main():
    config = Config()

    env = CrowdSimSgan()
    env.configure(config)
    env.setup(seed=0, num_of_env=1)

    env = DiscreteActions(env, discrete_actions)

    CHECKPOINT_DIR = './train/DQN_SGAN/'
    LOG_DIR = './logs/SGAN/'

    callback = TrainAndLoggingCallback(check_freq=100000, save_path=CHECKPOINT_DIR)

    model = DQN('MultiInputPolicy', env, verbose=1, tensorboard_log=LOG_DIR)

    # model.learn(total_timesteps=2000000, callback=callback)
    # model.save('latestmodel')


if __name__ == '__main__':
    main()
