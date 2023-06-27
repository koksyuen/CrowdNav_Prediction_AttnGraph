"""

Potential Field based path planner

author: Atsushi Sakai (@Atsushi_twi)

Ref:
https://www.cs.cmu.edu/~motionplanning/lecture/Chap4-Potential-Field_howie.pdf

"""

import numpy as np

# Parameters
KP = 5.0  # attractive potential gain
ETA = 1.0  # repulsive potential gain
# decay rate with respect to time    0.9^(t-t0)
DECAY = [1.0, 0.9, 0.81, 0.73, 0.66, 0.59, 0.53, 0.48]


def calc_potential_field(gx, gy, humans_emotion, comfort_dist, pred_humans_traj, map_size, reso, rr, hr, detected_human):
    """
    construct artificial potential field (APF) based on the trajectories and goal
    gx: x-coordinate of goal
    gy: y-coordinate of goal
    humans_emotion: array of humans' emotion (numpy array: num_of_human)
    comfort_dist: dictionary of comfort distance boundary (meter) based on emotion
    pred_humans_traj: array of predicted humans' trajectories (traj_len, num_of_human, 2)
    map_size: width and height of map (unit: meter)
    reso: resolution of map (unit: meter per grid)
    rr: robot_radius (unit: meter)
    hr: array of predicted humans' trajectories (numpy array: num_of_human)
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

    comfort_radius = []
    for human_id in range(humans_emotion.shape[0]):
        comfort_radius.append(comfort_dist[humans_emotion[human_id]])
    comfort_radius = np.array(comfort_radius)

    radius = rr + hr + comfort_radius

    decay = np.array(DECAY)

    # empty map
    pmap = np.zeros((index, index, 1), dtype=np.float32)

    if detected_human:

        # calc each potential

        for ix in range(index):
            x = -(ix * reso - center)

            for iy in range(index):
                y = -(iy * reso - center)
                ug = calc_attractive_potential(x, y, gx, gy)
                uo = calc_repulsive_potential(x, y, pred_humans_traj, radius, decay)
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

    pmap_norm = (pmap / np.linalg.norm(pmap)) * 255
    return pmap_norm.reshape(pmap.shape[0], pmap.shape[1], 1)


def calc_attractive_potential(x, y, gx, gy):
    return 0.5 * KP * np.hypot(x - gx, y - gy)


def calc_repulsive_potential(x, y, pred_humans_traj, radius, decay):

    # pred_humans_traj: array of predicted humans' trajectories (traj_len, num_of_human, 2)
    dq = np.hypot(x-pred_humans_traj[:,:,0], y-pred_humans_traj[:,:,1])
    # dq (traj_len, num_of_human)
    dq[dq<=0.1] = 0.1
    # radius (num_of_human, )

    # dq[dq > radius] = 0.0
    uo = 0.5 * ETA * (1.0 / dq - 1.0 / radius) ** 2

    # transpose to from (traj_len, num_of_human) to (num_of_human,traj_len)
    uo = uo.transpose(1, 0)

    uo = decay * uo

    return np.max(uo)
