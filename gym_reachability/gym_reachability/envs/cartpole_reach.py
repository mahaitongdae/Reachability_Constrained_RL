# Copyright (c) 2019–2020, The Regents of the University of California.
# All rights reserved.
#
# This file is subject to the terms and conditions defined in the LICENSE file
# included in this repository.
#
# Please contact the author(s) of this library if you have any questions.
# Authors: Neil Lugovoy   ( nflugovoy@berkeley.edu )

import gym.spaces
import numpy as np
from gym.envs.classic_control.cartpole import CartPoleEnv
from ray.rllib.utils.annotations import override
import math
import scipy.io as sio
from gym_reachability.gym_reachability.utils import q_values_from_q_func, offsets


class CartPoleReachabilityEnv(CartPoleEnv):

    seed_val = 0
    cartpole_mat_file_path = '../../../cartpole_data.mat'

    @override(CartPoleEnv)
    def __init__(self):

        # super init
        super().__init__()

        # state bounds
        # 0 axis = state, 1 axis = low vs high
        self.bounds = np.array([[-self.x_threshold, self.x_threshold],
                                [-2.0, 2.0],
                                [-math.radians(50), math.radians(50)],
                                [-self.theta_threshold_radians, self.theta_threshold_radians]])
        self.low = self.low = np.array(self.bounds)[:, 0]
        self.high = np.array(self.bounds)[:, 1]
        # multiply by 2 to give buffer since ray will crash if outside box
        self.observation_space = gym.spaces.Box(2 * self.low, 2 * self.high, dtype=np.float32)

        # prevents env from warning when stepping in failure set
        self.steps_beyond_done = 1

        # seeding
        self.seed(CartPoleReachabilityEnv.seed_val)

    @override(CartPoleEnv)
    def reset(self):
        """
        if self.random_start is false this is the same as normal cartpole else the environment will
        start from a random state anywhere in the observation space
        :return: the current state
        """
        super(CartPoleReachabilityEnv, self).reset()
        self.state = np.random.uniform(low=self.low, high=self.high)
        self.steps_beyond_done = 1  # prevents env from warning when stepping in failure set
        return np.array(self.state)

    @override(CartPoleEnv)
    def step(self, action):
        """

        :param action: input action
        :return: tuple of resulting state, l of new state, whether the episode is done, info
        dictionary
        """
        # reward must be computed before the environment steps see readme for a proof
        r = self.signed_distance(self.state)
        s, _, done, info = super(CartPoleReachabilityEnv, self).step(action)

        # allow the environment to run into failure set to let negative values propagate
        info['done'] = done  # done info is provided in case algorithm wants to use it
        return s, r, False, info

    def signed_distance(self, s):
        """
        :param s: state
        :return: the signed distance of the environment at state s to the failure set. For
        this problem the set is only bounded in the x and theta dimensions so any x_dot and
        theta_dot are permitted.

        """
        # calculate in x dimension
        x_low, x_high = self.bounds[0]
        x = self.state[0]
        x_dist = min(x_high - x, x - x_low)  # signed distance to boundary
        x_in = x_dist < 0

        # calculate in theta dimension
        theta_low, theta_high = self.bounds[2]
        theta = self.state[2]
        theta_dist = min(theta_high - theta, theta - theta_low)
        theta_in = theta_dist < 0

        if x_in or theta_in:
            return -((x_in * (x_dist ** 2) + theta_in * (theta_dist ** 2)) ** 0.5)

        return min(x_dist, theta_dist)

    def ground_truth_v(self):
        cartpole_data = sio.loadmat(self.cartpole_mat_file_path)
        v = cartpole_data['data'][:, :, :, :, -1]  # take value computed at last timestep
        v = v[3:-3][:][5:-5][:]  # slice to take values computed for correct bounds
        return v

    def ground_truth_comparison(self, q_func):
        """
        compares value function induced from q_func against ground truth for double integrator
        :param q_func: a function that takes in the state and outputs the q values for each action
        at that state
        :return: dictionary of statistics about comparison
        """

        ground_truth_v = self.ground_truth_v()
        computed_v = q_values_from_q_func(q_func, self.grid_cells, self.bounds, 2)
        misclassified_safe = 0
        misclassified_unsafe = 0
        misclassified_safe_adjusted = 0
        it = np.nditer(ground_truth_v, flags=['multi_index'])
        while not it.finished:
            if ground_truth_v[it.multi_index] < 0 < computed_v[it.multi_index]:
                misclassified_safe += 1
                misclassified_safe_adjusted += 1
                for index_offset in offsets(4):  # check grid cells one away
                    if ground_truth_v[it.multi_index + index_offset] > 0:
                        misclassified_safe_adjusted -= 1
                        break
            elif computed_v[it.multi_index] < 0 < ground_truth_v[it.multi_index]:
                misclassified_unsafe += 1
            it.iternext()
        return misclassified_safe, misclassified_unsafe, misclassified_safe_adjusted

