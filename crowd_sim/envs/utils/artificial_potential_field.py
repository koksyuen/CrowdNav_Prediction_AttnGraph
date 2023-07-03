"""

Potential Field based path planner

author: Atsushi Sakai (@Atsushi_twi)

Ref:
https://www.cs.cmu.edu/~motionplanning/lecture/Chap4-Potential-Field_howie.pdf

"""

import numpy as np

# Parameters
KP = 9.0  # attractive potential gain
ETA = 1.0  # repulsive potential gain
# decay rate with respect to time    0.9^(t-t0)
DECAY = [1.0, 0.9, 0.81, 0.73, 0.66, 0.59, 0.53, 0.48]


class ArtificialPotentialField():

    def __init__(self, map_size, reso, comfort_dist_database):
        """
        map_size: width and height of map (unit: meter)
        reso: resolution of map (unit: meter per grid)
        comfort_dist: dictionary of comfort distance boundary (meter) based on emotion
        """
        self.map_size = map_size
        self.resolution = reso
        self.comfort_dist = comfort_dist_database

    def calc_potential_field(self, gx, gy, humans_emotion, pred_humans_traj, rr, hr, detected_human):
        """
        construct artificial potential field (APF) based on the trajectories and goal
        gx: x-coordinate of goal
        gy: y-coordinate of goal
        humans_emotion: array of humans' emotion (numpy array: num_of_human)
        comfort_dist: dictionary of comfort distance boundary (meter) based on emotion
        pred_humans_traj: array of predicted humans' trajectories (traj_len, num_of_human, 2)

        rr: robot_radius (unit: meter)
        hr: array of predicted humans' trajectories (numpy array: num_of_human)
        detected_human: True if detected human
        :return: pmap (numpy array with shape of: map_size, map_size, 1)
        """

        # check if the goal is within local map
        # threshold = self.map_size / 2
        # if gx > threshold or gx < -threshold or gy > threshold or gy < -threshold:
        #     # calculate direction to the goal
        #     angle = np.arctan2(gy, gx)
        #     gx = np.clip(gx, -threshold, threshold) * np.cos(angle)
        #     gy = np.clip(gy, -threshold, threshold) * np.sin(angle)

        # center of map
        center = self.map_size / 2
        # number of grid
        width = int(round((self.map_size / self.resolution)))

        # Generate empty map
        # 0: x-axis    1: y-axis
        pmap_y, pmap_x = np.meshgrid(np.linspace(center, -center, width), np.linspace(center, -center, width))

        if detected_human:

            comfort_radius = []
            for human_id in range(humans_emotion.shape[0]):
                comfort_radius.append(self.comfort_dist[humans_emotion[human_id]])
            comfort_radius = np.array(comfort_radius)

            radius = rr + hr + comfort_radius
            decay = np.array(DECAY)

            """ goal attractive force """
            ug = 0.5 * KP * np.hypot(pmap_x - gx, pmap_y - gy)

            """ obstacle repulsive force """
            # dq shape = (pmap.shape[0], pmap.shape[1], traj_len, num_of_human)
            dq = np.hypot(pmap_x.reshape(pmap_x.shape[0], pmap_x.shape[1], 1, 1) - pred_humans_traj[:, :, 0],
                          pmap_y.reshape(pmap_y.shape[0], pmap_y.shape[1], 1, 1) - pred_humans_traj[:, :, 1])

            dq[dq <= 0.1] = 0.1

            # radius shape = (num_of_human,)
            # uo shape = (pmap.shape[0], pmap.shape[1], traj_len, num_of_human)
            uo = 0.5 * ETA * (1.0 / dq - 1.0 / radius) ** 2

            # uo shape = (pmap.shape[0], pmap.shape[1], num_of_human, traj_len)
            uo = uo.transpose(0, 1, 3, 2)

            # decay shape = (traj_len,)
            # uo shape = (pmap.shape[0], pmap.shape[1], num_of_human, traj_len)
            uo = decay * uo

            # find the maximum over the last two axes
            # uo shape = (pmap.shape[0], pmap.shape[1])
            uo = np.max(uo, axis=(-2, -1))

            """ total potential force """
            pmap = ug + uo

        else:
            # only calculate goal attractive force
            pmap = 0.5 * KP * np.hypot(pmap_x - gx, pmap_y - gy)

        pmap_norm = (pmap-np.min(pmap))/(np.max(pmap)-np.min(pmap)) * np.iinfo(np.uint8).max
        pmap_norm = np.round(pmap_norm).astype(np.uint8)
        return pmap_norm.reshape(pmap.shape[0], pmap.shape[1], 1)
