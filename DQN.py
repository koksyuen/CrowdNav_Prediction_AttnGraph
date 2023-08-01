import gym
from gym.spaces import Discrete
import numpy as np
from crowd_sim.envs.crowd_sim_raw import CrowdSimRaw
from sb3.feature_extractor import Preprocessor, ApfFeaturesExtractor
import os
import torch

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
# from stable_baselines3 import DQN
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
# print(len(discrete_actions))



class DiscreteActions(gym.ActionWrapper):
    def __init__(self, env, disc_to_cont):
        super().__init__(env)
        self.disc_to_cont = disc_to_cont
        self.action_space = Discrete(len(disc_to_cont))
        # self.action_space = Discrete(disc_to_cont.shape[0])

    def action(self, act):
        return self.disc_to_cont[act]


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
        env = DiscreteActions(env, discrete_actions)
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
    num_cpu = 20  # Number of processes to use
    seed = 1
    venv = SubprocVecEnv([make_env(seed, i, config, num_cpu) for i in range(num_cpu)])

    # obs = venv.reset()
    # writer = SummaryWriter("./logs/dqn_apf_raw")

    PRETRAIN_MODEL_PATH = './train/DQN_BC_APF_RAW/best_dict_15.pth'
    policy_dict = torch.load(PRETRAIN_MODEL_PATH)
    print(policy_dict)

    CHECKPOINT_DIR = './train/D3QN_RAW_APF/'
    LOG_DIR = './logs/D3QN_RAW_APF/'

    # FIRST TIME TRAINING
    policy_kwargs = dict(
        features_extractor_class=ApfFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=512),
    )
    model = DQN("CnnPolicy", venv, learning_rate=0.0001, policy_kwargs=policy_kwargs,
                verbose=1, device='cuda', tensorboard_log=LOG_DIR,
                batch_size=512)
    print(model.policy)
    model.policy.q_net.load_state_dict(policy_dict)
    model.policy.q_net_target.load_state_dict(policy_dict)

    # observations = torch.from_numpy(obs).cuda().float()
    # writer.add_graph(model.policy, observations)

    # CONTINUAL TRAINING
    # MODEL_PATH = './train/PPO_SGAN/best_model_4000000'
    # callback = TrainAndLoggingCallback(check_freq=100000, save_path=CHECKPOINT_DIR, start_step=4000000)
    # model = PPO.load(MODEL_PATH, env, tensorboard_log=LOG_DIR)

    callback = TrainAndLoggingCallback(check_freq=int(1e5), save_path=CHECKPOINT_DIR)
    model.learn(total_timesteps=int(1e6), callback=callback, progress_bar=True)
    model_path = os.path.join(CHECKPOINT_DIR, 'best_model')
    model.save(model_path)


if __name__ == '__main__':
    main()
