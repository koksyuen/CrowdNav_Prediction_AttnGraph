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
from crowd_sim.envs.utils.recorder import Recoder



class CrowdSim(gym.Env):
    """
    A base environment
    treat it as an abstract class, all other environments inherit from this one
    """
    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.
        """
        self.time_limit = None
        self.time_step = None
        self.robot = None # a Robot instance representing the robot
        self.humans = None # a list of Human instances, representing all humans in the environment
        self.global_time = None
        self.step_counter=0

        # reward function
        self.success_reward = None
        self.collision_penalty = None
        self.discomfort_dist = None
        self.discomfort_penalty_factor = None
        # simulation configuration
        self.config = None
        self.case_capacity = None
        self.case_size = None
        self.case_counter = None
        self.randomize_attributes = None

        self.circle_radius = None
        self.human_num = None


        self.action_space=None
        self.observation_space=None

        # limit FOV
        self.robot_fov = None
        self.human_fov = None

        self.dummy_human = None
        self.dummy_robot = None

        #seed
        self.thisSeed=None # the seed will be set when the env is created

        #nenv
        self.nenv=None # the number of env will be set when the env is created.
        # Because the human crossing cases are controlled by random seed, we will calculate unique random seed for each
        # parallel env.

        self.phase=None # set the phase to be train, val or test
        self.test_case=None # the test case ID, which will be used to calculate a seed to generate a human crossing case

        # for render
        self.render_axis=None

        self.humans = []

        self.potential = None

        self.desiredVelocity = [0.0, 0.0]

        self.last_left = 0.
        self.last_right = 0.


    def configure(self, config):
        """ read the config to the environment variables """

        self.config = config

        self.time_limit = config.env.time_limit
        self.time_step = config.env.time_step
        self.randomize_attributes = config.env.randomize_attributes

        self.success_reward = config.reward.success_reward
        self.collision_penalty = config.reward.collision_penalty
        self.discomfort_dist = config.reward.discomfort_dist
        self.discomfort_penalty_factor = config.reward.discomfort_penalty_factor

        """ SEED FOR RANDOMISATION """

        # seed made up of 32bit int, thus 32bit int of seed is divided into 3 parts:
        #     - val: 1000
        #     - test: 1000
        #     - train: remaining
        # case capacity: the maximum number for train(max possible int -2000), val(1000), and test(1000)
        self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}

        # case size is used to make sure that the case_counter is always between 0 and case_size[phase]
        # case size:
        #     - train: remaining size
        #     - validation: specified by user
        #     - test: specified by user
        self.case_size = {'train': np.iinfo(np.uint32).max - 2000, 'val': self.config.env.val_size,
                          'test': self.config.env.test_size}

        self.case_counter = {'train': 0, 'test': 0, 'val': 0}

        """"""

        self.circle_radius = config.sim.circle_radius
        self.human_num = config.sim.human_num

        self.arena_size = config.sim.arena_size


        logging.info('human number: {}'.format(self.human_num))
        if self.randomize_attributes:
            logging.info("Randomize human's radius and preferred speed")
        else:
            logging.info("Not randomize human's radius and preferred speed")

        logging.info('Circle width: {}'.format(self.circle_radius))


        self.robot_fov = np.pi * config.robot.FOV
        self.human_fov = np.pi * config.humans.FOV
        logging.info('robot FOV %f', self.robot_fov)
        logging.info('humans FOV %f', self.human_fov)


        # set dummy human and dummy robot
        # dummy humans, used if any human is not in view of other agents
        self.dummy_human = Human(self.config, 'humans')
        # if a human is not in view, set its state to (px = 100, py = 100, vx = 0, vy = 0, theta = 0, radius = 0)
        self.dummy_human.set(7, 7, 7, 7, 0, 0, 0) # (7, 7, 7, 7, 0, 0, 0)
        self.dummy_human.time_step = config.env.time_step

        self.dummy_robot = Robot(self.config, 'robot')
        self.dummy_robot.set(7, 7, 7, 7, 0, 0, 0)
        self.dummy_robot.time_step = config.env.time_step
        self.dummy_robot.kinematics = 'holonomic'
        self.dummy_robot.policy = ORCA(config)

        self.r = self.config.humans.radius


        # configure randomized goal changing of humans midway through episode
        self.random_goal_changing = config.humans.random_goal_changing
        if self.random_goal_changing:
            self.goal_change_chance = config.humans.goal_change_chance

        # configure randomized goal changing of humans after reaching their respective goals
        self.end_goal_changing = config.humans.end_goal_changing
        if self.end_goal_changing:
            self.end_goal_change_chance = config.humans.end_goal_change_chance

        """" Human states database """
        # include all humans (visible & non-visible)
        # observable state of a human = px, py, vx, vy, radius (5 variables)
        self.last_human_states = np.zeros((self.human_num, 5))
        """"""

        self.predict_steps = config.sim.predict_steps
        self.human_num_range = config.sim.human_num_range
        # assert self.human_num > self.human_num_range
        if self.config.action_space.kinematics == 'holonomic':
            self.max_human_num = self.human_num + self.human_num_range
            self.min_human_num = self.human_num - self.human_num_range
        else:
            self.min_human_num = 1
            self.max_human_num = 5

        # for sim2real dynamics check
        self.record=config.sim2real.record
        self.load_act=config.sim2real.load_act
        self.ROSStepInterval=config.sim2real.ROSStepInterval
        self.fixed_time_interval=config.sim2real.fixed_time_interval
        self.use_fixed_time_interval = config.sim2real.use_fixed_time_interval
        if self.record:
            self.episodeRecoder=Recoder()
            self.load_act=config.sim2real.load_act
            if self.load_act:
                self.episodeRecoder.loadActions()
        # use dummy robot and human states or use detected states from sensors
        self.use_dummy_detect = config.sim2real.use_dummy_detect



        # prediction period / control (or simulation) period
        self.pred_interval = int(config.data.pred_timestep // config.env.time_step)
        self.buffer_len = self.predict_steps * self.pred_interval

        # set robot for this envs
        rob_RL = Robot(config, 'robot')
        self.set_robot(rob_RL)


    def set_robot(self, robot):
        pass

    def generate_random_human_position(self, human_num):
        """
        Calls generate_circle_crossing_human function to generate a certain number of random humans
        :param human_num: the total number of humans to be generated
        :return: None
        """
        # initial min separation distance to avoid danger penalty at beginning
        for i in range(human_num):
            self.humans.append(self.generate_circle_crossing_human())


    def generate_circle_crossing_human(self):
        """Generate a human: generate start position on a circle, goal position is at the opposite side"""
        human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes() # random v_pref (v_max), radius of human & emotion

        # trial_num = 0

        while True:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            noise_range = 2
            px_noise = np.random.uniform(-1, 1) * noise_range
            py_noise = np.random.uniform(-1, 1) * noise_range
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise
            collide = False
            # trial_num += 1

            for i, agent in enumerate([self.robot] + self.humans):
                # keep human at least 3 meters away from robot
                if self.robot.kinematics == 'unicycle' and i == 0:
                    # Todo: if circle_radius <= 4, unicycle robot will get stuck???
                    min_dist = self.circle_radius / 2 # Todo: if circle_radius <= 4, it will get stuck here
                else:
                    min_dist = human.radius + agent.radius + self.discomfort_dist
                if norm((px - agent.px, py - agent.py)) < min_dist or norm((px - agent.gx, py - agent.gy)) < min_dist:
                    collide = True
                    break
            if not collide:
                # if trial_num > 2:
                    # print('trial_num: {}'.format(trial_num))
                break

        # px = np.random.uniform(-6, 6)
        # py = np.random.uniform(-3, 3.5)
        # human.set(px, py, px, py, 0, 0, 0)
        # def set(self, px, py, gx, gy, vx, vy, theta, radius=None, v_pref=None)
        human.set(px, py, -px, -py, 0, 0, 0)
        return human



    # update the robot belief of human states
    # if a human is visible, its state is updated to its current ground truth state
    # else we assume it keeps going in a straight line with last observed velocity
    def update_last_human_states(self, human_visibility, reset):
        """
        update the self.last_human_states array
        human_visibility: list of booleans returned by get_human_in_fov (e.x. [T, F, F, T, F])
        reset: True if this function is called by reset, False if called by step
        :return:
        """
        # keep the order of 5 humans at each timestep
        for i in range(self.human_num):
            if human_visibility[i]:
                # observable state of a human = px, py, vx, vy, radius
                humanS = np.array(self.humans[i].get_observable_state_list())
                self.last_human_states[i, :] = humanS

            else:
                if reset: # temporary fake human's state (bcuz this human haven't being detected yet)
                    humanS = np.array([15., 15., 0., 0., 0.3])
                    self.last_human_states[i, :] = humanS

                else: # if a human is not visible
                    # Plan A: linear approximation of human's next position
                    px, py, vx, vy, r = self.last_human_states[i, :]
                    px = px + vx * self.time_step
                    py = py + vy * self.time_step
                    self.last_human_states[i, :] = np.array([px, py, vx, vy, r])

                    # Plan B: assume the human doesn't move, use last observation
                    # self.last_human_states[i, :] = np.array([px, py, 0., 0., r])

                    # Plan C: put invisible humans in a dummy position
                    # humanS = np.array([15., 15., 0., 0., 0.3])
                    # self.last_human_states[i, :] = humanS

    """
    For training of trajectory predictor (not related to RL)
    return the ground truth locations of all humans
    """
    def get_true_human_states(self):
        true_human_states = np.zeros((self.human_num, 2))
        for i in range(self.human_num):
            humanS = np.array(self.humans[i].get_observable_state_list())
            true_human_states[i, :] = humanS[:2]
        return true_human_states


    # set robot initial state and generate all humans for reset function
    # for crowd nav: human_num == self.human_num
    # for leader follower: human_num = self.human_num - 1
    def generate_robot_humans(self, phase, human_num=None):
        if human_num is None:
            human_num = self.human_num

        """ randomize starting position and goal position of robot """
        if self.robot.kinematics == 'unicycle':
            angle = np.random.uniform(0, np.pi * 2)
            px = self.circle_radius * np.cos(angle)
            py = self.circle_radius * np.sin(angle)
            while True:
                gx, gy = np.random.uniform(-self.circle_radius, self.circle_radius, 2)
                if np.linalg.norm([px - gx, py - gy]) >= 6:  # at least 6 meters away
                    break
            # def set(self, px, py, gx, gy, vx, vy, theta, radius=None, v_pref=None)
            self.robot.set(px, py, gx, gy, 0, 0, np.random.uniform(0, 2*np.pi)) # randomize initial orientation (theta)
        # robot.kinematics: holonomic
        else:
            while True:
                px, py, gx, gy = np.random.uniform(-self.circle_radius, self.circle_radius, 4)
                if np.linalg.norm([px - gx, py - gy]) >= 6:
                    break
            self.robot.set(px, py, gx, gy, 0, 0, np.pi/2)


        ## generate humans
        self.generate_random_human_position(human_num=human_num)


    """ mimic the dynamics of Turtlebot2i for sim2real """
    def smooth_action(self, action):
        # if action.r is delta theta
        w = action.r / self.time_step
        # if action.r is w
        # w = action.r
        beta = 0.1
        left = (2 * action.v - 0.23 * w) / (2 * 0.035)
        right = (2 * action.v + 0.23 * w) / (2 * 0.035)

        left = np.clip(left, -17.5, 17.5)
        right = np.clip(right, -17.5, 17.5)

        # print('Before: left:', left, 'right:', right)
        if self.phase == 'test':
            left = (1. - beta) * self.last_left + beta * left
            right = (1. - beta) * self.last_right + beta * right

        self.last_left = copy.deepcopy(left)
        self.last_right = copy.deepcopy(right)

        # subtract a noisy amount of delay from wheel speeds to simulate the delay in tb2
        # do this in the last step because this happens after we send action commands to tb2
        if left > 0:
            adjust_left = left - np.random.normal(loc=1.8, scale=0.15)
            left = max(0., adjust_left)
        else:
            adjust_left = left + np.random.normal(loc=1.8, scale=0.15)
            left = min(0., adjust_left)

        if right > 0:
            adjust_right = right - np.random.normal(loc=1.8, scale=0.15)
            right = max(0., adjust_right)
        else:
            adjust_right = right + np.random.normal(loc=1.8, scale=0.15)
            right = min(0., adjust_right)

        if self.record:
            self.episodeRecoder.wheelVelList.append([left, right])
        # print('After: left:', left, 'right:', right)

        v = 0.035 / 2 * (left + right)
        r = 0.035 / 0.23 * (right - left) * self.time_step
        return ActionRot(v, r)


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


    # Update the humans' end goals in the environment
    # Produces valid end goals for each human
    def update_human_goals_randomly(self):
        # Update humans' goals randomly
        for human in self.humans:
            if human.isObstacle or human.v_pref == 0:
                continue # human (static) / wall no need to change goal
            if np.random.random() <= self.goal_change_chance: # possibility of random goal changing
                humans_copy = []
                for h in self.humans:
                    if h != human:
                        humans_copy.append(h)

                # trial_num = 0
                # Produce valid goal for human in case of circle setting
                while True: # keep on generating random goal until no collision
                    angle = np.random.random() * np.pi * 2
                    # add some noise to simulate all the possible cases robot could meet with human
                    v_pref = 1.0 if human.v_pref == 0 else human.v_pref
                    gx_noise = (np.random.random() - 0.5) * v_pref
                    gy_noise = (np.random.random() - 0.5) * v_pref
                    gx = self.circle_radius * np.cos(angle) + gx_noise
                    gy = self.circle_radius * np.sin(angle) + gy_noise
                    collide = False
                    # trial_num += 1

                    for agent in [self.robot] + humans_copy: # detect collision with robot & other pedestrians
                        min_dist = human.radius + agent.radius + self.discomfort_dist
                        if norm((gx - agent.px, gy - agent.py)) < min_dist or norm((gx - agent.gx, gy - agent.gy)) < min_dist:
                            collide = True
                            break
                    if not collide: # no collision with all other agents
                        # if trial_num > 1:
                            # print('trial num: {}'.format(trial_num))
                        break

                # Give human new (valid) goal
                human.gx = gx
                human.gy = gy
        return

    # Update the specified human's end goals in the environment randomly
    def update_human_goal(self, human):

        # Update human's goals randomly
        if np.random.random() <= self.end_goal_change_chance:
            humans_copy = []
            for h in self.humans:
                if h != human:
                    humans_copy.append(h)


            while True:
                angle = np.random.random() * np.pi * 2
                # add some noise to simulate all the possible cases robot could meet with human
                v_pref = 1.0 if human.v_pref == 0 else human.v_pref
                gx_noise = (np.random.random() - 0.5) * v_pref
                gy_noise = (np.random.random() - 0.5) * v_pref
                gx = self.circle_radius * np.cos(angle) + gx_noise
                gy = self.circle_radius * np.sin(angle) + gy_noise
                collide = False

                for agent in [self.robot] + humans_copy:
                    min_dist = human.radius + agent.radius + self.discomfort_dist
                    if norm((gx - agent.px, gy - agent.py)) < min_dist or \
                            norm((gx - agent.gx, gy - agent.gy)) < min_dist:
                        collide = True
                        break
                if not collide:
                    break

            # Give human new goal
            human.gx = gx
            human.gy = gy
        return

    # calculate the angle between the direction vector of state1 and the vector from state1 to state2
    def calc_offset_angle(self, state1, state2):
        if self.robot.kinematics == 'holonomic':
            real_theta = np.arctan2(state1.vy, state1.vx)
        else:
            real_theta = state1.theta
        # angle of center line of FOV of agent1
        v_fov = [np.cos(real_theta), np.sin(real_theta)]

        # angle between agent1 and agent2
        v_12 = [state2.px - state1.px, state2.py - state1.py]
        # angle between center of FOV and agent 2

        v_fov = v_fov / np.linalg.norm(v_fov)
        v_12 = v_12 / np.linalg.norm(v_12)

        offset = np.arccos(np.clip(np.dot(v_fov, v_12), a_min=-1, a_max=1))
        return offset

    """
    Caculate whether agent2 is in agent1's FOV
    Not the same as whether agent1 is in agent2's FOV!!!!
    arguments:
    state1, state2: can be agent instance OR state instance
    robot1: is True if state1 is robot, else is False
    return value:
    return True if state2 is visible to state1, else return False
    """
    def detect_visible(self, state1, state2, robot1 = False, custom_fov=None, custom_sensor_range=None):
        if self.robot.kinematics == 'holonomic':
            real_theta = np.arctan2(state1.vy, state1.vx)
        else:
            real_theta = state1.theta
        # angle of center line of FOV of agent1
        v_fov = [np.cos(real_theta), np.sin(real_theta)]

        # angle between agent1 and agent2
        v_12 = [state2.px - state1.px, state2.py - state1.py]
        # angle between center of FOV and agent 2

        v_fov = v_fov / np.linalg.norm(v_fov)
        v_12 = v_12 / np.linalg.norm(v_12)

        offset = np.arccos(np.clip(np.dot(v_fov, v_12), a_min=-1, a_max=1))
        if custom_fov:
            fov = custom_fov
        else:
            if robot1:
                fov = self.robot_fov
            else:
                fov = self.human_fov

        if np.abs(offset) <= fov / 2:
            inFov = True
        else:
            inFov = False

        # detect whether state2 is in state1's sensor_range
        dist = np.linalg.norm([state1.px - state2.px, state1.py - state2.py]) - state1.radius - state2.radius
        if custom_sensor_range:
            inSensorRange = dist <= custom_sensor_range
        else:
            if robot1:
                inSensorRange = dist <= self.robot.sensor_range
            else: # Humans have infinite sensor range
                inSensorRange = True

        return (inFov and inSensorRange)


    """
    For robot only
    :return:
        - list of visible humans (obj) to robot 
        - number of visible humans
        - human_ids: a list of True/False to indicate whether each human is visible
    """
    def get_num_human_in_fov(self):
        human_ids = []
        humans_in_view = []
        num_humans_in_view = 0

        for i in range(self.human_num):
            visible = self.detect_visible(self.robot, self.humans[i], robot1=True)
            if visible:
                humans_in_view.append(self.humans[i])
                num_humans_in_view = num_humans_in_view + 1
                human_ids.append(True)
            else:
                human_ids.append(False)

        return humans_in_view, num_humans_in_view, human_ids

    '''
    convert self.last_human_states to a list of observable state objects for old algorithms to use
    '''
    def last_human_states_obj(self):
        humans = []
        for i in range(self.human_num):
            h = ObservableState(*self.last_human_states[i])
            humans.append(h)
        return humans


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

    def get_human_actions(self):
        # step all humans
        human_actions = []  # a list of all humans' actions

        for i, human in enumerate(self.humans):
            # observation for humans is always coordinates
            ob = []
            for other_human in self.humans:
                if other_human != human:
                    # Else detectable humans are always observable to each other
                    if self.detect_visible(human, other_human):
                        ob.append(other_human.get_observable_state())
                    else: # out of human's fov (note: human has infinite sensor range)
                        ob.append(self.dummy_human.get_observable_state())

            if self.robot.visible:
                if self.detect_visible(self.humans[i], self.robot):
                    ob += [self.robot.get_observable_state()]
                else:
                    ob += [self.dummy_robot.get_observable_state()]

            human_actions.append(human.act(ob))

        return human_actions

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

