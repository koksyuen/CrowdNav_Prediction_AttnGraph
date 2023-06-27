"""

Potential Field based path planner

author: Atsushi Sakai (@Atsushi_twi)

Ref:
https://www.cs.cmu.edu/~motionplanning/lecture/Chap4-Potential-Field_howie.pdf

"""

import numpy as np

# Parameters
KP = 5.0  # attractive potential gain
ETA = 100.0  # repulsive potential gain
DECAY = 0.9  # decay rate with respect to time


def calc_potential_field(gx, gy, humans_emotion, comfort_dist, pred_humans_traj, map_size, reso, rr, hr, detected_human):
    """
    construct artificial potential field (APF) based on the trajectories and goal
    gx: x-coordinate of goal
    gy: y-coordinate of goal
    humans_emotion: array of humans' emotion (num_of_human)
    comfort_dist: dictionary of comfort distance boundary (meter) based on emotion
    pred_humans_traj: array of predicted humans' trajectories (traj_len, num_of_human, 2)
    map_size: width and height of map (unit: meter)
    reso: resolution of map (unit: meter per grid)
    rr: robot_radius (unit: meter)
    hr: array of predicted humans' trajectories (num_of_human)
    detected_human: True if detected human
    :return: pmap (numpy array with shape of: map_size, map_size, 1)
    """

    # check if the goal is within local map
    threshold = map_size / 2
    if gx > threshold or gx < -threshold or gy > threshold or gy < -threshold:
        # calculate direction to the goal
        angle = np.arctan2(gy, gx)
        gx = np.clip(gx, -threshold, threshold) * np.cos(angle)
        gy = np.clip(gy, -threshold, threshold) * np.sin(angle)

    center = map_size / 2
    index = int(round((map_size / reso)))

    # empty map
    pmap = np.zeros((index, index, 1), dtype=np.uint8)

    if detected_human:

        # calc each potential

        for ix in range(index):
            x = -(ix * reso - center)

            for iy in range(index):
                y = -(iy * reso + center)
                ug = calc_attractive_potential(x, y, gx, gy)
                uo = calc_repulsive_potential(x, y, humans_emotion, comfort_dist, pred_humans_traj, rr, hr)
                uf = ug + uo
                pmap[ix][iy] = uf
    else:
        # calc each potential

        for ix in range(index):
            x = -(ix * reso - center)

            for iy in range(index):
                y = -(iy * reso + center)
                ug = calc_attractive_potential(x, y, gx, gy)
                pmap[ix][iy] = ug

    return pmap.reshape(pmap.shape[0], pmap.shape[1], 1)


def calc_attractive_potential(x, y, gx, gy):
    return 0.5 * KP * np.hypot(x - gx, y - gy)


def calc_repulsive_potential(x, y, humans_emotion, comfort_dist, pred_humans_traj, rr, hr):

    uo_max = -float("inf")
    pred_humans_traj = pred_humans_traj.transpose(1, 0, 2)

    for human_id in range(pred_humans_traj.shape[0]):

        radius = rr + hr[human_id] + comfort_dist[humans_emotion[human_id]]
        for seq_num in range(pred_humans_traj.shape[1]):
            ox = pred_humans_traj[human_id, seq_num, 0]
            oy = pred_humans_traj[human_id, seq_num, 1]
            dq = np.hypot(x - ox, y - oy)
            if dq <= radius:
                if dq <= 0.1:
                    dq = 0.1
                uo = (DECAY ** seq_num) * (0.5 * ETA * (1.0 / dq - 1.0 / radius) ** 2)
            else:
                uo = 0.0

            if uo > uo_max:
                uo_max = uo
    return uo_max
