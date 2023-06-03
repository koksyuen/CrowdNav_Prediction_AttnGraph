import logging
import gym
from gym.spaces import Box, Dict
import numpy as np
import rvo2
import random
import copy
import time

from numpy.linalg import norm
from crowd_sim.envs import *
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.info import *
from crowd_nav.policy.orca import ORCA
from crowd_sim.envs.utils.state import *
from crowd_sim.envs.utils.action import ActionRot, ActionXY

import sys

sys.path.append('/home/koksyuen/python_project/sgan')
from predictor import socialGAN


class CrowdSimSgan(CrowdSim):
    """
        Added Social GAN for pedestrians' trajectory prediction
    """

    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.
        """
        super().__init__()

    def configure(self, config):
        """ read the config to the environment variables """

        super().configure(config)

        self.emotion_coeff = config.sgan.emotions  # database of emotion category

        self.obs_len = config.sgan.obs_len
        self.pred_len = config.sgan.pred_len

        """" Human states database (include history) for SocialGAN"""
        # queue data structure
        # include all humans (visible & non-visible)
        # observable state of a human = px, py (2 variables)
        self.human_states_record = np.zeros((self.obs_len, self.human_num, 2), dtype=np.float32)
        """"""

        self.traj_predictor = socialGAN(model_path=config.sgan.model_path)

        self.map_resolution = config.sgan.map_resolution
        self.local_map_size = int(2 * (self.robot.sensor_range + 1) / config.sgan.map_resolution)

        # For unicycle action
        self.robot.vx_max = config.robot.vx_max
        self.robot.dtheta_max = config.robot.dtheta_max

        self.observation_space = Dict({"local_goal": Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
                                       "local_map": Box(low=0, high=255,
                                                        shape=(self.local_map_size, self.local_map_size),
                                                        dtype=np.uint8)})

        self.action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def set_robot(self, robot):
        self.robot = robot

    def setup(self, seed, num_of_env, ax=None):
        self.thisSeed = seed
        self.nenv = num_of_env
        self.render_axis = ax

    def reset(self, phase='train', test_case=None):
        """
        Reset the environment
        :return: observation
        """

        if self.robot is None:
            raise AttributeError('robot has to be set!')

        """ SEED FOR RANDOMISATION """

        if self.phase is not None:
            phase = self.phase
        if self.test_case is not None:
            test_case = self.test_case

        assert phase in ['train', 'val', 'test']

        # test case is passed in to calculate specific seed to generate case
        # by default, case counter start from 0 without specifying test case
        if test_case is not None:
            self.case_counter[phase] = test_case

        # train, val, and test phase should start with different seed.
        # seed made up of 32bit int, thus the seed used for:
        #     - validation: 0 - 999
        #     - testing:    1000 - 1999
        #     - training:   2000 - (max size of 32bit int)
        counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                          'val': 0, 'test': self.case_capacity['val']}

        # print('phase: {}   counter_offset: {}   case_counter: {}   seed: {}'.format(phase, counter_offset[phase], self.case_counter[phase], self.thisSeed))

        np.random.seed(counter_offset[phase] + self.case_counter[phase] + self.thisSeed)

        # case size is used to make sure that the case_counter is always between 0 and case_size[phase]
        # The % operator ensures the counter variable loops to beginning once it reaches the end
        # To prevent errors due to boundary conditions of the seed (32bit int)
        self.case_counter[phase] = (self.case_counter[phase] + int(1 * self.nenv)) % self.case_size[phase]

        """"""

        """ Initialisation """

        self.humans = []  # remove all humans at the beginning of an episode
        self.humans_emotion = []
        self.global_time = 0
        self.step_counter = 0

        # default time step: 0.25s
        for agent in [self.robot] + self.humans:
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step

        self.generate_robot_humans(phase)

        """"""

        # record humans' emotion
        self.record_humans_emotion()

        # get first observation
        ob = self.generate_ob(reset=True)

        # initialize potential (distance to goal) reward
        self.potential = -abs(
            np.linalg.norm(np.array([self.robot.px, self.robot.py]) - np.array([self.robot.gx, self.robot.gy])))

        return ob

    # calculate the reward at current timestep R(s, a)
    def calc_reward(self, action):
        # collision detection
        dmin = float('inf')  # for discomfort distance penalty

        danger_dists = []
        collision = False

        for i, human in enumerate(self.humans):
            dx = human.px - self.robot.px
            dy = human.py - self.robot.py
            closest_dist = (dx ** 2 + dy ** 2) ** (1 / 2) - human.radius - self.robot.radius

            if closest_dist < self.discomfort_dist:
                danger_dists.append(closest_dist)
            if closest_dist < 0:
                collision = True
                # logging.debug("Collision: distance between robot and p{} is {:.2E}".format(i, closest_dist))
                break
            elif closest_dist < dmin:
                dmin = closest_dist

        # check if reached the goal
        reaching_goal = norm(
            np.array(self.robot.get_position()) - np.array(self.robot.get_goal_position())) < self.robot.radius

        if self.global_time >= self.time_limit - 1:  # reached termination state (time limit for one episode)
            reward = 0
            done = True
            episode_info = Timeout()
        elif collision:  # termination state (collision)
            reward = self.collision_penalty
            done = True
            episode_info = Collision()
        elif reaching_goal:  # termination state (reached goal)
            reward = self.success_reward
            done = True
            episode_info = ReachGoal()

        elif dmin < self.discomfort_dist:
            # only penalize agent for getting too close if it's visible
            # only take one pedestrian (minimum distance) into account
            # adjust the reward based on FPS
            # print(dmin)
            reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
            done = False
            episode_info = Danger(dmin)

        else:
            # potential reward
            potential_cur = np.linalg.norm(
                np.array([self.robot.px, self.robot.py]) - np.array(self.robot.get_goal_position()))
            reward = 2 * (-abs(potential_cur) - self.potential)
            self.potential = -abs(potential_cur)

            done = False
            episode_info = Nothing()

        # if the robot is near collision/arrival, it should be able to turn a large angle
        if self.robot.kinematics == 'unicycle':
            # add a rotational penalty
            # if action.r is w, factor = -0.02 if w in [-1.5, 1.5], factor = -0.045 if w in [-1, 1];
            # if action.r is delta theta, factor = -2 if r in [-0.15, 0.15], factor = -4.5 if r in [-0.1, 0.1]
            r_spin = -5 * action.r ** 2

            # add a penalty for going backwards
            if action.v < 0:
                r_back = -2 * abs(action.v)
            else:
                r_back = 0.

            reward = reward + r_spin + r_back

        return reward, done, episode_info


    def generate_ob(self, reset):
        """
        Compute the observation
        :return: a dictionary
            - local_goal: goal coordinate in robot (local) frame
            - local_map: costmap in robot (local) frame
        """

        # human's state
        visible_human_states, num_visible_humans, human_visibility = self.get_num_human_in_fov()

        self.update_last_human_states(human_visibility, reset=reset)

        self.update_human_states_record(reset=reset)

        self.update_visible_human_states_record(human_visibility, reset=reset)

        ''' Generate local map (includes trajectory prediction) '''
        if num_visible_humans > 0:  # implement socialGAN only if detected human(s)
            self.update_visible_last_human_emotion(human_visibility)
            # SocialGAN: pedestrians' trajectory prediction in global frame
            self.predicted_human_states = self.traj_predictor(self.visible_human_states_record)
            # convert predicted trajectory from global frame to local (robot) frame
            local_predicted_human_states = self.global_to_local(self.predicted_human_states)
            # start_time = time.time()
            local_map = self.generate_costmap(local_predicted_human_states)
            # end_time = time.time()
            # print('map construction time: {}s'.format(end_time-start_time))
        else:
            local_map = np.zeros((self.local_map_size, self.local_map_size), dtype=np.uint8)

        ### Calculates goal coordinate in local frame
        gx, gy = self.calculate_local_goal()

        ### Normalise observation (doesn't include local map)
        nor_gx, nor_gy = self.normalize_goal(gx, gy)

        # record for next step
        self.previous_human_visibility = np.array(human_visibility)

        ob = {'local_goal': np.array([nor_gx, nor_gy], dtype=np.float32),
              'local_map': local_map
              }
        return ob

    def normalize_goal(self, gx, gy):
        """
        normalize goal coordinates to (-1.0 ~ 1.0)
        :return: normalized x and y coordinate of goal
        """
        # normalized_value = original_value / original_max_value
        # since goal is symmetrical (-max ~ max)
        nor_gx = gx / (2 * self.circle_radius)
        nor_gy = gy / (2 * self.circle_radius)

        return nor_gx, nor_gy


    def calculate_local_goal(self):
        """
        calculate robot's goal in robot (local) frame
        :return: x and y coordinate of robot's goal in robot (local) frame
        """
        dx = self.robot.px
        dy = self.robot.py

        if self.robot.kinematics == 'holonomic':
            # T ^ w _ r
            T_w_r = np.array([[1.0, 0.0, dx],
                              [0.0, 1.0, dy],
                              [0, 0, 1]])  # 2D transformation matrix
        else:  # unicyle
            theta = self.robot.theta
            # T ^ w _ r
            T_w_r = np.array([[np.cos(theta), - np.sin(theta), dx],
                              [np.sin(theta), np.cos(theta), dy],
                              [0, 0, 1]])  # 2D transformation matrix

        # T ^ r _ w
        T_r_w = np.linalg.inv(T_w_r)

        # goal in global frame
        global_goal = np.array([self.robot.gx, self.robot.gy])
        # goal in robot (local) frame
        local_goal = np.dot(T_r_w, np.hstack((global_goal, 1)))

        return local_goal[0], local_goal[1]


    def generate_costmap(self, trajectories):
        """
        construct costmap based on the trajectories
        trajectories: array of humans' trajectories (traj_len, num_of_human, 2)
        :return: costmap (n-by-n uint8 numpy array)
        """
        local_map = np.zeros((self.local_map_size, self.local_map_size), dtype=np.uint8)

        # transform coordinates (x,y) [unit: meter] of the trajectories to indexes (i,j) of the local map
        trajectories_index = (1/self.map_resolution) * trajectories

        # offset the indexes (i,j) of the local map, so that center of local map is located at center point
        offset = int(self.local_map_size / 2)
        offset_trajectories_index = offset - np.ceil(trajectories_index).astype(int)

        # clip indexes (i,j) outside of local map
        offset_trajectories_index = np.clip(offset_trajectories_index, 0, self.local_map_size - 1)

        # change the shape of array to: (num_of_human, traj_len, 2)
        offset_trajectories_index = offset_trajectories_index.transpose(1,0,2)

        # construct local map
        for human_id in range(offset_trajectories_index.shape[0]):
            # decay rate based on the emotion of pedestrian
            decay_rate = self.emotion_coeff[self.visible_humans_emotion[human_id]]
            map_value = 255
            for seq_num in range(offset_trajectories_index.shape[1]):
                x = offset_trajectories_index[human_id, seq_num, 0]
                y = offset_trajectories_index[human_id, seq_num, 1]
                map_value = int(map_value * decay_rate)
                if local_map[x, y] < map_value:
                    local_map[x, y] = map_value

        return local_map

    def global_to_local(self, traj_global):
        """
        convert coordinates from global frame to robot (local) frame
        traj_global: array of humans' trajectories (traj_len, num_of_human, 2) in the global frame
        :return: array of humans' trajectories (traj_len, num_of_human, 2) in the local (robot) frame
        """
        dx = self.robot.px
        dy = self.robot.py

        if self.robot.kinematics == 'holonomic':
            # T ^ w _ r
            T_w_r = np.array([[1.0, 0.0, dx],
                              [0.0, 1.0, dy],
                              [0, 0, 1]])  # 2D transformation matrix
        else:  # unicyle
            theta = self.robot.theta
            # T ^ w _ r
            T_w_r = np.array([[np.cos(theta), - np.sin(theta), dx],
                              [np.sin(theta), np.cos(theta), dy],
                              [0, 0, 1]])  # 2D transformation matrix

        # T ^ r _ w
        T_r_w = np.linalg.inv(T_w_r)
        # add the homogeneous coordinate
        traj_global_hom = np.concatenate([traj_global, np.ones((traj_global.shape[0], traj_global.shape[1], 1))],
                                         axis=2)
        traj_local_hom = np.tensordot(traj_global_hom, T_r_w, axes=([2], [1]))
        traj_local = traj_local_hom[:, :, :-1]  # remove the homogeneous coordinate
        return traj_local

    # extract visible_human_states_record from human_states_record
    def update_visible_human_states_record(self, human_visibility, reset):
        """
        update self.visible_human_states_record (input to SocialGAN)
        dimension of visible_human_states_record array: obs_len * num_of_visible_humans * 2
        reset: True if this function is called by reset, False if called by step
        :return:
        """
        current_human_visibility = np.array(human_visibility)

        if not reset:
            # previous frame is not visible, but current frame is visible
            # (current_visibility XOR previous_visibility) AND current_visibility
            human_first_visibility = np.bitwise_and(
                np.bitwise_xor(self.previous_human_visibility, current_human_visibility),
                current_human_visibility)

            # since: the previous frame is not visible, but current frame is visible
            # thus: the states of previous frames are set to the states observed in the current frame
            self.human_states_record[:, human_first_visibility, :] = self.last_human_states[human_first_visibility, :2]

        # print('human_states_record: {}, current_human_visibility: {}'.format(self.human_states_record.shape, current_human_visibility.shape))

        # extract visible humans' states
        self.visible_human_states_record = self.human_states_record[:, current_human_visibility, :]

    # update the human_states_record (queue)
    def update_human_states_record(self, reset):
        """
        update the self.human_states_record array (queue)
        dimension of human_states_record array: obs_len * num_of_humans * 2
        reset: True if this function is called by reset, False if called by step
        :return:
        """

        if reset:
            # at the first frame, the whole trajectory is recorded as the starting point of human
            self.human_states_record[:] = self.last_human_states[:, :2]
        else:
            # push latest humans' states into the queue
            self.human_states_record[:-1] = self.human_states_record[1:]
            self.human_states_record[-1] = self.last_human_states[:, :2]

    def record_humans_emotion(self):
        for i in range(self.human_num):
            self.humans_emotion.append(self.humans[i].emotion)


    def update_visible_last_human_emotion(self, human_visibility):
        self.visible_humans_emotion = np.array(self.humans_emotion)[np.array(human_visibility)]


    def rescale_action(self, normalized_action):
        """
        rescale the normalized action to usable action
        normalized_action: action that range from -1.0 to 1.0
        :return: usable_action that range from -max to max
        """
        if self.robot.kinematics == 'holonomic':
            vx = normalized_action.vx * self.robot.v_pref
            vy = normalized_action.vy * self.robot.v_pref
            return ActionXY(vx, vy)
        else:   # unicycle
            vx = normalized_action.v * self.robot.vx_max
            dtheta = normalized_action.r * self.robot.dtheta_max
            return ActionRot(vx, dtheta)


    def step(self, raw_action, update=True):
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)
        """

        # different action format
        # rescale action
        if self.robot.kinematics == 'holonomic':
            normalized_action = ActionXY(raw_action[0], raw_action[1])   # vx, vy
            action = self.rescale_action(normalized_action)
        else:   # unicycle
            normalized_action = ActionRot(raw_action[0], raw_action[1])   # v, r
            action = self.rescale_action(normalized_action)

        # humans perform action first
        human_actions = self.get_human_actions()

        # compute reward and episode info
        reward, done, episode_info = self.calc_reward(action)

        # apply action and update all agents
        self.robot.step(action)
        for i, human_action in enumerate(human_actions):
            self.humans[i].step(human_action)
        self.global_time += self.time_step  # max episode length=time_limit/time_step
        self.step_counter = self.step_counter + 1

        ##### compute_ob goes here!!!!!
        ob = self.generate_ob(reset=False)

        if self.robot.policy.name in ['srnn']:
            info = {'info': episode_info}
        else:  # for orca and sf
            info = episode_info

        # Update all humans' goals randomly midway through episode
        if self.random_goal_changing:
            if self.global_time % 5 == 0:  # every 5 seconds
                self.update_human_goals_randomly()

        # Update a specific human's goal once its reached its original goal
        if self.end_goal_changing:
            for human in self.humans:
                if not human.isObstacle and human.v_pref != 0 and norm(
                        (human.gx - human.px, human.gy - human.py)) < human.radius:
                    self.update_human_goal(human)

        return ob, reward, done, info

    def render(self, mode='human'):
        """ Render the current status of the environment using matplotlib """
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines
        from matplotlib import patches

        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

        robot_color = 'gold'
        goal_color = 'red'
        arrow_color = 'red'
        emotion_color = {'happy': 'blue',
                         'normal': 'black',
                         'angry': 'purple'}
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)

        def calcFOVLineEndPoint(ang, point, extendFactor):
            # choose the extendFactor big enough
            # so that the endPoints of the FOVLine is out of xlim and ylim of the figure
            FOVLineRot = np.array([[np.cos(ang), -np.sin(ang), 0],
                                   [np.sin(ang), np.cos(ang), 0],
                                   [0, 0, 1]])
            point.extend([1])
            # apply rotation matrix
            newPoint = np.matmul(FOVLineRot, np.reshape(point, [3, 1]))
            # increase the distance between the line start point and the end point
            newPoint = [extendFactor * newPoint[0, 0], extendFactor * newPoint[1, 0], 1]
            return newPoint

        ax = self.render_axis
        artists = []

        # add goal
        goal = mlines.Line2D([self.robot.gx], [self.robot.gy], color=goal_color, marker='*', linestyle='None',
                             markersize=15, label='Goal')
        ax.add_artist(goal)
        artists.append(goal)

        # add robot
        robotX, robotY = self.robot.get_position()

        robot = plt.Circle((robotX, robotY), self.robot.radius, fill=True, color=robot_color)
        ax.add_artist(robot)
        artists.append(robot)

        plt.legend([robot, goal], ['Robot', 'Goal'], fontsize=16)

        # compute orientation in each step and add arrow to show the direction
        radius = self.robot.radius
        arrowStartEnd = []

        robot_theta = self.robot.theta if self.robot.kinematics == 'unicycle' else np.arctan2(self.robot.vy,
                                                                                              self.robot.vx)

        arrowStartEnd.append(
            ((robotX, robotY), (robotX + radius * np.cos(robot_theta), robotY + radius * np.sin(robot_theta))))

        for i, human in enumerate(self.humans):
            theta = np.arctan2(human.vy, human.vx)
            arrowStartEnd.append(
                ((human.px, human.py), (human.px + radius * np.cos(theta), human.py + radius * np.sin(theta))))

        arrows = [patches.FancyArrowPatch(*arrow, color=arrow_color, arrowstyle=arrow_style)
                  for arrow in arrowStartEnd]
        for arrow in arrows:
            ax.add_artist(arrow)
            artists.append(arrow)

        # draw FOV for the robot
        # add robot FOV

        if self.robot.FOV < 2 * np.pi:
            FOVAng = self.robot_fov / 2
            FOVLine1 = mlines.Line2D([0, 0], [0, 0], linestyle='--')
            FOVLine2 = mlines.Line2D([0, 0], [0, 0], linestyle='--')

            startPointX = robotX
            startPointY = robotY
            endPointX = robotX + radius * np.cos(robot_theta)
            endPointY = robotY + radius * np.sin(robot_theta)

            # transform the vector back to world frame origin, apply rotation matrix, and get end point of FOVLine
            # the start point of the FOVLine is the center of the robot
            FOVEndPoint1 = calcFOVLineEndPoint(FOVAng, [endPointX - startPointX, endPointY - startPointY],
                                               20. / self.robot.radius)
            FOVLine1.set_xdata(np.array([startPointX, startPointX + FOVEndPoint1[0]]))
            FOVLine1.set_ydata(np.array([startPointY, startPointY + FOVEndPoint1[1]]))
            FOVEndPoint2 = calcFOVLineEndPoint(-FOVAng, [endPointX - startPointX, endPointY - startPointY],
                                               20. / self.robot.radius)
            FOVLine2.set_xdata(np.array([startPointX, startPointX + FOVEndPoint2[0]]))
            FOVLine2.set_ydata(np.array([startPointY, startPointY + FOVEndPoint2[1]]))

            ax.add_artist(FOVLine1)
            ax.add_artist(FOVLine2)
            artists.append(FOVLine1)
            artists.append(FOVLine2)

        # add an arc of robot's sensor range
        sensor_range = plt.Circle(self.robot.get_position(), self.robot.sensor_range, fill=False, linestyle='--')
        ax.add_artist(sensor_range)
        artists.append(sensor_range)
        # add humans and change the color of them based on visibility
        human_circles = [plt.Circle(human.get_position(), human.radius, fill=False) for human in self.humans]

        for i in range(len(self.humans)):
            ax.add_artist(human_circles[i])
            artists.append(human_circles[i])

            # green: visible; red: invisible
            if self.detect_visible(self.robot, self.humans[i], robot1=True):
                human_circles[i].set_color(c='g')
            else:
                human_circles[i].set_color(c='r')

            # label numbers on each human (color varies based on emotion of human)
            # plt.text(self.humans[i].px - 0.1, self.humans[i].py - 0.1, str(self.humans[i].id), color='black', fontsize=12)
            plt.text(self.humans[i].px + 0.2, self.humans[i].py + 0.2, i, color=emotion_color[self.humans[i].emotion],
                     fontsize=12)

        # label of emotion's color (act as description)
        yy = 9.2
        for emotion, color in emotion_color.items():
            plt.text(-9.8, yy, emotion, color=color, fontsize=12)
            yy -= 0.6

        # plot history of detected human positions
        for i in range(len(self.humans)):
            # Caution: this is actually current human visibility (bcuz render is called after reset or step)
            if self.previous_human_visibility[i]:
                # add history of positions of each human
                for j in range(self.obs_len - 1):
                    circle = plt.Circle(self.human_states_record[j][i], self.humans[i].radius,
                                        fill=False, color='tab:olive', linewidth=0.5,
                                        alpha=0.6 * (j + 1) / self.obs_len)
                    ax.add_artist(circle)
                    artists.append(circle)

        # plot predicted human trajectory
        sgan_i = 0
        for i in range(len(self.humans)):
            # Caution: this is actually current human visibility (bcuz render is called after reset or step)
            if self.previous_human_visibility[i]:
                # add predicted positions of each human
                for j in range(self.pred_len):
                    circle = plt.Circle(self.predicted_human_states[j][sgan_i], self.humans[i].radius,
                                        fill=False, color='tab:orange', linewidth=1.0,
                                        alpha=0.8 / (j + 1))
                    ax.add_artist(circle)
                    artists.append(circle)
                sgan_i += 1

        plt.pause(0.01)
        for item in artists:
            item.remove()  # there should be a better way to do this. For example,
            # initially use add_artist and draw_artist later on
        for t in ax.texts:
            t.set_visible(False)
