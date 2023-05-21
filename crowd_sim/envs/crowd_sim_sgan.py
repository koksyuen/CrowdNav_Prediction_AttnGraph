import logging
import gym
import numpy as np
import rvo2
import random
import copy

from numpy.linalg import norm
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.info import *
from crowd_nav.policy.orca import ORCA
from crowd_sim.envs.utils.state import *
from crowd_sim.envs.utils.action import ActionRot, ActionXY



class CrowdSimSgan(gym.Env):
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


    def set_robot(self, robot):
        self.robot = robot

    def setup(self, seed, num_of_env, ax):
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
            test_case=self.test_case

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
        self.case_counter[phase] = (self.case_counter[phase] + int(1*self.nenv)) % self.case_size[phase]

        """"""


        """ Initialisation """

        self.humans = [] # remove all humans at the beginning of an episode
        self.global_time = 0
        self.step_counter=0

        # default time step: 0.25s
        for agent in [self.robot] + self.humans:
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step

        self.generate_robot_humans(phase)

        """"""

        # get first observation
        ob = self.generate_ob(reset=True)

        # initialize potential (distance to goal) reward
        self.potential = -abs(np.linalg.norm(np.array([self.robot.px, self.robot.py]) - np.array([self.robot.gx, self.robot.gy])))

        return ob


    # calculate the reward at current timestep R(s, a)
    def calc_reward(self, action):
        # collision detection
        dmin = float('inf') # for discomfort distance penalty

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
        reaching_goal = norm(np.array(self.robot.get_position()) - np.array(self.robot.get_goal_position())) < self.robot.radius

        if self.global_time >= self.time_limit - 1: # reached termination state (time limit for one episode)
            reward = 0
            done = True
            episode_info = Timeout()
        elif collision: # termination state (collision)
            reward = self.collision_penalty
            done = True
            episode_info = Collision()
        elif reaching_goal: # termination state (reached goal)
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
            r_spin = -5 * action.r**2

            # add a penalty for going backwards
            if action.v < 0:
                r_back = -2 * abs(action.v)
            else:
                r_back = 0.

            reward = reward + r_spin + r_back

        return reward, done, episode_info

    """
    Compute the observation
    :return: a 1D numpy array that consists of:
        - number of visible humans
        - parameter of robot: px, py, vx, vy, radius, gx, gy, v_pref, theta
        - database of humans' states (includes visible & non-visible humans): px, py, vx, vy, radius  
    """
    def generate_ob(self, reset):
        visible_human_states, num_visible_humans, human_visibility = self.get_num_human_in_fov()
        self.update_last_human_states(human_visibility, reset=reset)
        if self.robot.policy.name in ['lstm_ppo', 'srnn']:
            ob = [num_visible_humans]
            # append robot's state
            robotS = np.array(self.robot.get_full_state_list())
            ob.extend(list(robotS)) # 9 states

            ob.extend(list(np.ravel(self.last_human_states))) #flatten to 1D array
            ob = np.array(ob) # (num_of_humans * 5) states

        else: # for orca and sf
            # convert self.last_human_states to a list of observable state objects for old algorithms to use
            ob = self.last_human_states_obj()

        return ob


    def step(self, action, update=True):
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)
        """

        # clip the action to obey robot's constraint
        action = self.robot.policy.clip_action(action, self.robot.v_pref)

        # humans perform action first
        human_actions = self.get_human_actions()


        # compute reward and episode info
        reward, done, episode_info = self.calc_reward(action)


        # apply action and update all agents
        self.robot.step(action)
        for i, human_action in enumerate(human_actions):
            self.humans[i].step(human_action)
        self.global_time += self.time_step # max episode length=time_limit/time_step
        self.step_counter=self.step_counter+1

        ##### compute_ob goes here!!!!!
        ob = self.generate_ob(reset=False)


        if self.robot.policy.name in ['srnn']:
            info={'info':episode_info}
        else: # for orca and sf
            info=episode_info

        # Update all humans' goals randomly midway through episode
        if self.random_goal_changing:
            if self.global_time % 5 == 0: # every 5 seconds
                self.update_human_goals_randomly()


        # Update a specific human's goal once its reached its original goal
        if self.end_goal_changing:
            for human in self.humans:
                if not human.isObstacle and human.v_pref != 0 and norm((human.gx - human.px, human.gy - human.py)) < human.radius:
                    self.update_human_goal(human)

        return ob, reward, done, info

    def render(self, mode='human'):
        """ Render the current status of the environment using matplotlib """
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines
        from matplotlib import patches

        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

        robot_color = 'yellow'
        goal_color = 'red'
        arrow_color = 'red'
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



        ax=self.render_axis
        artists=[]

        # add goal
        goal=mlines.Line2D([self.robot.gx], [self.robot.gy], color=goal_color, marker='*', linestyle='None', markersize=15, label='Goal')
        ax.add_artist(goal)
        artists.append(goal)

        # add robot
        robotX,robotY=self.robot.get_position()

        robot=plt.Circle((robotX,robotY), self.robot.radius, fill=True, color=robot_color)
        ax.add_artist(robot)
        artists.append(robot)

        plt.legend([robot, goal], ['Robot', 'Goal'], fontsize=16)


        # compute orientation in each step and add arrow to show the direction
        radius = self.robot.radius
        arrowStartEnd=[]

        robot_theta = self.robot.theta if self.robot.kinematics == 'unicycle' else np.arctan2(self.robot.vy, self.robot.vx)

        arrowStartEnd.append(((robotX, robotY), (robotX + radius * np.cos(robot_theta), robotY + radius * np.sin(robot_theta))))

        for i, human in enumerate(self.humans):
            theta = np.arctan2(human.vy, human.vx)
            arrowStartEnd.append(((human.px, human.py), (human.px + radius * np.cos(theta), human.py + radius * np.sin(theta))))

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
            FOVEndPoint1 = calcFOVLineEndPoint(FOVAng, [endPointX - startPointX, endPointY - startPointY], 20. / self.robot.radius)
            FOVLine1.set_xdata(np.array([startPointX, startPointX + FOVEndPoint1[0]]))
            FOVLine1.set_ydata(np.array([startPointY, startPointY + FOVEndPoint1[1]]))
            FOVEndPoint2 = calcFOVLineEndPoint(-FOVAng, [endPointX - startPointX, endPointY - startPointY], 20. / self.robot.radius)
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

            # label numbers on each human
            # plt.text(self.humans[i].px - 0.1, self.humans[i].py - 0.1, str(self.humans[i].id), color='black', fontsize=12)
            plt.text(self.humans[i].px - 0.1, self.humans[i].py - 0.1, i, color='black', fontsize=12)



        plt.pause(5.0)
        for item in artists:
            item.remove() # there should be a better way to do this. For example,
            # initially use add_artist and draw_artist later on
        for t in ax.texts:
            t.set_visible(False)

