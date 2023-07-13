from crowd_sim.envs.crowd_sim_raw import CrowdSimRaw
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from sb3.feature_extractor import Preprocessor
import gym
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from arguments import get_args
from crowd_nav.configs.config import Config

import sys

sys.path.append('/home/koksyuen/python_project/sgan')

from attrdict import AttrDict
from predictor import get_generator
from sgan.models import TrajectoryGenerator
from sgan.utils import relative_to_abs, torch_abs_to_relative

# Artificial Potential Field Parameters
KP = 9.0  # attractive potential gain
ETA = 1.0  # repulsive potential gain
# decay rate with respect to time    0.9^(t-t0)
DECAY = [1.0, 0.9, 0.81, 0.73, 0.66, 0.59, 0.53, 0.48]


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


def main():
    config = Config()

    writer = SummaryWriter("./logs/apf")

    num_cpu = 2  # Number of processes to use
    seed = 100
    batch_size = 2
    human_num = 12
    map_size = 12.0
    map_resolution = 0.05

    center = map_size / 2
    width = int(round((map_size / map_resolution)))

    pmap_y, pmap_x = torch.meshgrid(torch.linspace(center, -center, width, device='cuda'),
                                    torch.linspace(center, -center, width, device='cuda'),
                                    indexing='ij')

    decay = torch.tensor(DECAY, device='cuda')


    seq_start_end = []
    for i in range(batch_size):
        xx = i * human_num
        seq_start_end.append([xx, xx + human_num])
    seq_start_end = torch.tensor(seq_start_end)

    model_path = '/home/koksyuen/python_project/sgan/models/sgan-p-models/eth_8_model.pt'
    checkpoint = torch.load(model_path)
    sgan = get_generator(checkpoint)

    venv = SubprocVecEnv([make_env(seed, i, config, num_cpu) for i in range(num_cpu)])

    # obs: (batch, obs_len+2, human_num, 2)
    obs = venv.reset()

    observations = torch.from_numpy(obs).cuda().float()
    # observations = torch.from_numpy(np.random.random((2, 10, 10, 2))).cuda().float()

    # goal: (batch_size, 2)
    goal = observations[:, -1, 0]

    # num_human: (batch_size, 1)
    # num_human = observations[:, -1, 1, 0]
    # num_human = num_human.reshape(num_human.shape[0], 1)

    # human_visibility: (batch_size, human_num)
    # human_visibility = observations[:, -2, :, 0].bool()
    # human_visibility_stack = torch.flatten(human_visibility)

    # radius: (batch_size, human_num)
    radius = observations[:, -2, :, 1]

    # obs_traj_stack: (obs_len, human_num * batch size, 2)
    obs_traj = observations[:, :-2, :, :].permute(0, 2, 1, 3)
    obs_traj_stack = obs_traj.reshape(obs_traj.shape[0] * obs_traj.shape[1], obs_traj.shape[2], obs_traj.shape[3])
    obs_traj_stack = obs_traj_stack.permute(1, 0, 2)

    obs_traj_rel = torch_abs_to_relative(obs_traj_stack)

    pred_traj_rel = sgan(obs_traj_stack, obs_traj_rel, seq_start_end)

    pred_traj = relative_to_abs(pred_traj_rel, obs_traj_stack[-1])

    pred_traj = pred_traj.permute(1, 0, 2)
    pred_traj_batch = pred_traj.reshape(batch_size, int(pred_traj.shape[0]/batch_size), pred_traj.shape[1],
                                        pred_traj.shape[2])
    pred_traj_batch = pred_traj_batch.permute(0, 2, 1, 3)
    pred_traj_batch = pred_traj_batch.reshape(pred_traj_batch.shape[0], 1, 1, pred_traj_batch.shape[1],
                                              pred_traj_batch.shape[2], pred_traj_batch.shape[3])

    """ goal attractive force """
    ug = 0.5 * KP * torch.hypot(torch.unsqueeze(pmap_x, dim=0) - goal[:, 0].reshape(goal.shape[0], 1, 1),
                                torch.unsqueeze(pmap_y, dim=0) - goal[:, 1].reshape(goal.shape[0], 1, 1))

    pmap_x_temp = pmap_x.reshape(1, pmap_x.shape[0], pmap_x.shape[1], 1, 1)
    pmap_y_temp = pmap_y.reshape(1, pmap_y.shape[0], pmap_y.shape[1], 1, 1)

    dq_x = pmap_x_temp - pred_traj_batch[:, :, :, :, :, 0]
    dq_y = pmap_y_temp - pred_traj_batch[:, :, :, :, :, 1]

    dq = torch.hypot(dq_x, dq_y)

    dq[dq <= 0.1] = 0.1

    radius = radius.reshape(radius.shape[0], 1, 1, 1, radius.shape[1])
    uo = 0.5 * ETA * (1.0 / dq - 1.0 / radius) ** 2

    decay = decay.reshape(1, 1, 1, 1, decay.shape[0])
    uo = uo.permute(0, 1, 2, 4, 3)
    uo = decay * uo

    uo = torch.amax(uo, dim=(-2, -1))

    """ total potential force """
    u_total = torch.add(ug, uo)

    u_min = torch.amin(u_total, dim=(-2, -1), keepdim=True)
    u_max = torch.amax(u_total, dim=(-2, -1), keepdim=True)
    pmap_norm = (u_total - u_min) / (u_max - u_min)
    pmap_norm = torch.unsqueeze(pmap_norm, dim=1)

    writer.add_images("pmap", img_tensor=pmap_norm, global_step=1, dataformats='NCHW')
    # writer.close()
    print('finish')


if __name__ == '__main__':
    main()
