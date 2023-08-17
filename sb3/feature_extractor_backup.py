import torch
import torch.nn as nn
from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import sys

sys.path.append('../sgan')

from attrdict import AttrDict
from sgan.models import TrajectoryGenerator
from sgan.utils import relative_to_abs, torch_abs_to_relative

# Artificial Potential Field Parameters
KP = 1.0  # attractive potential gain
ETA = 4.0  # repulsive potential gain
# KP = 50.0  # attractive potential gain
# ETA = 100.0  # repulsive potential gain
# decay rate with respect to time    0.9^(t-t0)
DECAY = [1.0, 0.9, 0.81, 0.73, 0.66, 0.59, 0.53, 0.48]


class Preprocessor(nn.Module):
    def __init__(self, map_size=10.0, map_resolution=0.05,
                 model_path='../sgan/models/sgan-p-models/eth_8_model.pt'):
        super(Preprocessor, self).__init__()

        with torch.no_grad():
            """ initialisation """
            self.batch_size = None
            self.trial = True
            # center of map
            center = map_size / 2
            # number of grid
            width = int(round((map_size / map_resolution)))
            # x and y coordinates of potential map
            self.pmap_x, self.pmap_y = torch.meshgrid(torch.linspace(center, -center, width, device='cuda'),
                                                      torch.linspace(center, -center, width, device='cuda'),
                                                      indexing='ij')
            # Decay Rate with respect to time
            self.decay = torch.tensor(DECAY, device='cuda')
            # reshape to (batch_size, map_height, map_width, human_num, obs_len)
            self.decay = self.decay.reshape(1, 1, 1, 1, self.decay.shape[0])

            """ Social GAN initialisation """
            checkpoint = torch.load(model_path)
            args = AttrDict(checkpoint['args'])
            self.traj_predictor = TrajectoryGenerator(
                obs_len=args.obs_len,
                pred_len=args.pred_len,
                embedding_dim=args.embedding_dim,
                encoder_h_dim=args.encoder_h_dim_g,
                decoder_h_dim=args.decoder_h_dim_g,
                mlp_dim=args.mlp_dim,
                num_layers=args.num_layers,
                noise_dim=args.noise_dim,
                noise_type=args.noise_type,
                noise_mix_type=args.noise_mix_type,
                pooling_type=args.pooling_type,
                pool_every_timestep=args.pool_every_timestep,
                dropout=args.dropout,
                bottleneck_dim=args.bottleneck_dim,
                neighborhood_size=args.neighborhood_size,
                grid_size=args.grid_size,
                batch_norm=args.batch_norm)
            self.traj_predictor.load_state_dict(checkpoint['g_state'])
            self.traj_predictor.cuda()
            self.traj_predictor.eval()

            for param in self.traj_predictor.parameters():
                param.requires_grad = False

    def forward(self, observations):
        with torch.no_grad():
            # observations shape: (batch size, obs_len + 2, num_human, 2)
            observations = observations.cuda()

            """ initialisation based on observations """
            if self.batch_size != observations.shape[0]:
                if self.trial:
                    self.trial = False
                    self.human_num = 0  # dummy
                else:
                    self.batch_size = observations.shape[0]
                    # seq of humans' index (for Social GAN)
                    seq_start_end = []
                    self.human_num = int(observations[0, -1, 1, 0])
                    for i in range(self.batch_size):
                        start = i * self.human_num
                        seq_start_end.append([start, start + self.human_num])
                    # seq_start_end: (batch_size, 2)
                    self.seq_start_end = torch.tensor(seq_start_end, device='cuda')
                    print('human_num: {}'.format(self.human_num))

            """ data extraction """
            # goal: (batch_size, 2)
            goal = observations[:, -1, 0]

            if self.human_num > 0 and not self.trial:
                # radius: (batch_size, human_num)
                radius = observations[:, -2, :self.human_num, 1]

                # comfort_distance: (batch_size, human_num)
                comfort_distance = observations[:, -2, :self.human_num, 0]

                # obs_traj_stack: (obs_len, human_num * batch size, 2)
                obs_traj = observations[:, :-2, :self.human_num, :].permute(0, 2, 1, 3)
                obs_traj_stack = obs_traj.reshape(obs_traj.shape[0] * obs_traj.shape[1],
                                                  obs_traj.shape[2], obs_traj.shape[3])
                obs_traj_stack = obs_traj_stack.permute(1, 0, 2)

                """ Social GAN: pedestrians' trajectories prediction """
                # obs_traj_rel: (obs_len, human_num * batch size, 2)
                obs_traj_rel = torch_abs_to_relative(obs_traj_stack)

                # pred_traj_rel: (pred_len, human_num * batch size, 2)
                pred_traj_rel = self.traj_predictor(obs_traj_stack, obs_traj_rel, self.seq_start_end)

                # pred_traj: (pred_len, human_num * batch size, 2)
                pred_traj = relative_to_abs(rel_traj=pred_traj_rel, start_pos=obs_traj_stack[-1])

                # pred_traj_batch: (batch_size, map_height, map_width, pred_len, human_num, 2)
                pred_traj = pred_traj.permute(1, 0, 2)
                pred_traj_batch = pred_traj.reshape(self.batch_size,
                                                    int(pred_traj.shape[0] / self.batch_size),
                                                    pred_traj.shape[1], pred_traj.shape[2])
                pred_traj_batch = pred_traj_batch.permute(0, 2, 1, 3)
                pred_traj_batch = pred_traj_batch.reshape(pred_traj_batch.shape[0], 1, 1, pred_traj_batch.shape[1],
                                                          pred_traj_batch.shape[2], pred_traj_batch.shape[3])

            """ goal attractive force """
            # ug: (batch size, map_height, map_width)
            ug = 0.5 * KP * torch.hypot(torch.unsqueeze(self.pmap_x, dim=0) - goal[:, 0].reshape(goal.shape[0], 1, 1),
                                        torch.unsqueeze(self.pmap_y, dim=0) - goal[:, 1].reshape(goal.shape[0], 1, 1))

            if self.human_num > 0 and not self.trial:
                """ obstacle repulsive force """
                pmap_x_temp = self.pmap_x.reshape(1, self.pmap_x.shape[0], self.pmap_x.shape[1], 1, 1)
                pmap_y_temp = self.pmap_y.reshape(1, self.pmap_y.shape[0], self.pmap_y.shape[1], 1, 1)
                dq_x = pmap_x_temp - pred_traj_batch[:, :, :, :, :, 0]
                dq_y = pmap_y_temp - pred_traj_batch[:, :, :, :, :, 1]

                # distance to obstacle, dq: (batch size, map_height, map_width, obs_len, human_num)
                dq = torch.hypot(dq_x, dq_y)
                # dq[dq <= 0.1] = 0.1  # minimum distance to obstacle

                # radius: (batch size, map_height, map_width, obs_len, human_num)
                radius = radius.reshape(radius.shape[0], 1, 1, 1, radius.shape[1])
                comfort_distance = comfort_distance.reshape(comfort_distance.shape[0], 1, 1, 1, comfort_distance.shape[1])

                # obstacle repulsive force, uo: (batch size, map_height, map_width, obs_len, human_num)
                # uo = 0.5 * ETA * (10000000.0 / dq - 1.0 / radius)
                # uo = 0.5 * ETA * (1.0 / dq - 1.0 / radius) ** 2
                # uo = 0.5 * ETA * (1.0 / (dq + radius)) ** 2
                # uo = 0.5 * ETA * (-dq / radius + 1)
                # uo = 0.5 * ETA * (- dq / radius + 1)
                # uo_comfort_zone = 0.5 * ETA
                # uo_human = ETA

                uo = torch.where(dq <= comfort_distance, 0.5 * ETA, 0.0)
                uo = torch.where(dq <= radius, ETA, uo)

                # uo: (batch size, map_height, map_width, human_num, obs_len)
                uo = uo.permute(0, 1, 2, 4, 3)
                uo = self.decay * uo

                # uo: (batch size, map_height, map_width)
                uo = torch.amax(uo, dim=(-2, -1))

                """ total potential force """
                # mask = torch.where(uo <= 0.01 * ETA, 1.0, 0.0)
                # ug = torch.mul(mask, ug)
                # artificial potential map: (batch size, map_height, map_width)
                u_total = torch.add(ug, uo)
                # u_total = uo
            else:
                u_total = ug

            """ normalization of artificial potential map (0 ~ 1) """
            # pmap_norm = (batch size, map_channel, map_height, map_width)
            u_min = torch.amin(u_total, dim=(-2, -1), keepdim=True)
            u_max = torch.amax(u_total, dim=(-2, -1), keepdim=True)
            pmap_norm = (u_total - u_min) / (u_max - u_min)
            pmap_norm = torch.unsqueeze(pmap_norm, dim=1)

            return pmap_norm


class ApfFeaturesExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # print(observation_space.shape)
        self.apf_generator = Preprocessor(map_size=10.0, map_resolution=0.1)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = 1
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        ).cuda()

        # Compute shape by doing one forward pass
        with torch.no_grad():
            xxx = self.cnn(self.apf_generator(
                torch.as_tensor(observation_space.sample()[None]).float()
            ))
            n_flatten = xxx.shape[1]
            # print(xxx.shape)

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.apf_map = self.apf_generator(observations)
        return self.linear(self.cnn(self.apf_map))

    def get_current_apf(self):
        return self.apf_map

    # def get_current_pred_traj(self):
    #     return self.apf_generator.pred_traj

