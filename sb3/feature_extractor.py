import torch
import torch.nn as nn
from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import sys
sys.path.append('/home/koksyuen/python_project/sgan')

from attrdict import AttrDict
from sgan.models import TrajectoryGenerator
from sgan.utils import relative_to_abs, abs_to_relative

# Artificial Potential Field Parameters
KP = 9.0  # attractive potential gain
ETA = 1.0  # repulsive potential gain
# decay rate with respect to time    0.9^(t-t0)
DECAY = [1.0, 0.9, 0.81, 0.73, 0.66, 0.59, 0.53, 0.48]


class Preprocessor(nn.Module):
    def __init__(self, map_size=None, map_resolution=None,
                 model_path='/home/koksyuen/python_project/sgan/models/sgan-p-models/eth_8_model.pt'):
        super(Preprocessor, self).__init__()
        self.map_size = map_size
        self.resolution = map_resolution
        self.decay = torch.tensor(DECAY, device='cuda')

        with torch.no_grad():
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

    def apf(self, no_human=True, gx=None, gy=None, radius=None, obstacle_traj=None):

        # center of map
        center = self.map_size / 2
        # number of grid
        width = int(round((self.map_size / self.resolution)))

        # Create an empty map
        # 0: x-axis    1: y-axis
        pmap_y, pmap_x = torch.meshgrid(torch.linspace(center, -center, width, device='cuda'),
                                        torch.linspace(center, -center, width, device='cuda'),
                                        indexing='ij')

        if no_human:
            # only calculate goal attractive force
            u_total = 0.5 * KP * torch.hypot(pmap_x - gx, pmap_y - gy)

        else:
            """ goal attractive force """
            ug = 0.5 * KP * torch.hypot(pmap_x - gx, pmap_y - gy)

            # dq shape = (pmap.shape[0], pmap.shape[1], traj_len, num_of_human)
            dq = torch.hypot(pmap_x.reshape(pmap_x.shape[0], pmap_x.shape[1], 1, 1) - obstacle_traj[:, :, 0],
                             pmap_y.reshape(pmap_y.shape[0], pmap_y.shape[1], 1, 1) - obstacle_traj[:, :, 1])

            dq[dq <= 0.1] = 0.1

            # radius shape = (num_of_human,)
            # uo shape = (pmap.shape[0], pmap.shape[1], traj_len, num_of_human)
            uo = 0.5 * ETA * (1.0 / dq - 1.0 / radius) ** 2

            # uo shape = (pmap.shape[0], pmap.shape[1], num_of_human, traj_len)
            uo = uo.permute(0, 1, 3, 2)

            # decay shape = (traj_len,)
            # uo shape = (pmap.shape[0], pmap.shape[1], num_of_human, traj_len)
            uo = self.decay * uo

            # find the maximum over the last two axes
            # uo shape = (pmap.shape[0], pmap.shape[1])
            uo = torch.amax(uo, dim=(-2, -1))

            """ total potential force """
            u_total = torch.add(ug, uo)

        # normalization (0.0 - 1.0)
        pmap_norm = (u_total - torch.min(u_total)) / (torch.max(u_total) - torch.min(u_total))
        return pmap_norm.reshape(1, pmap_norm.shape[0], pmap_norm.shape[1])

    def forward(self, observations):
        with torch.no_grad():
            # observations shape: (batch size, obs_len + 2, num_human, 2)
            observations = observations.cuda()
            # print(observations.shape, observations.dtype, observations.device)
            # observations = observations.reshape(observations.shape[1], observations.shape[2], observations.shape[3])
            human_visibility = observations[-2, :, 0].bool()
            radius = observations[-2, human_visibility, 1]
            gx, gy = observations[-1, 0]
            num_human = observations[-1, 1, 0]
            obs_traj = observations[:-2, human_visibility, :]
            obs_traj_rel = abs_to_relative(obs_traj)
            seq_start_end = torch.tensor([[0, obs_traj.shape[1]]], device='cuda')
            pred_traj_rel = self.traj_predictor(
                obs_traj, obs_traj_rel, seq_start_end
            )
            pred_traj = relative_to_abs(
                pred_traj_rel, obs_traj[-1]
            )
            if num_human > 0:
                return self.apf(no_human=False, gx=gx, gy=gy, radius=radius, obstacle_traj=pred_traj)
            else:
                return self.apf(no_human=True, gx=gx, gy=gy, radius=radius, obstacle_traj=pred_traj)


class ApfFeaturesExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        self.apf_generator = Preprocessor(map_size=12.0, map_resolution=0.1)
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
            print(xxx.shape)

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            apf_map = self.apf_generator(observations)
        return self.linear(self.cnn(apf_map))
