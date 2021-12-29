# Copyright (c) 2019–2020, The Regents of the University of California.
# All rights reserved.
#
# This file is subject to the terms and conditions defined in the LICENSE file
# included in this repository.
#
# Please contact the author(s) of this library if you have any questions.
# Authors: Neil Lugovoy   ( nflugovoy@berkeley.edu )

import numpy as np
import gym
from gym import spaces
from gym.envs.box2d.lunar_lander import LunarLander
from gym.envs.box2d.lunar_lander import SCALE, VIEWPORT_W, VIEWPORT_H, LEG_DOWN, FPS, LEG_AWAY, \
    LANDER_POLY, LEG_H, LEG_W
# NOTE the overrides cause crashes with ray in this file but I would like to include them for
# clarity in the future
from ray.rllib.utils.annotations import override

# these variables are needed to do calculations involving the terrain but are local variables
# in LunarLander reset() unfortunately

W = VIEWPORT_W / SCALE
CHUNKS = 11  # number of polygons used to make the lunar surface
HELIPAD_Y = (VIEWPORT_H / SCALE) / 4  # height of helipad in simulator scale

# height of lander body in simulator scale. LANDER_POLY has the (x,y) points that define the
# shape of the lander in pixel scale
LANDER_POLY_X = np.array(LANDER_POLY)[:, 0]
LANDER_POLY_Y = np.array(LANDER_POLY)[:, 1]

LANDER_W = (np.max(LANDER_POLY_X) - np.min(LANDER_POLY_X)) / SCALE
LANDER_H = (np.max(LANDER_POLY_Y) - np.min(LANDER_POLY_Y)) / SCALE

# distance of edge of legs from center of lander body in simulator scale
LEG_X_DIST = LEG_AWAY / SCALE
LEG_Y_DIST = LEG_DOWN / SCALE

# radius around lander to check for collisions
LANDER_RADIUS = ((LANDER_H / 2 + LEG_Y_DIST + LEG_H / SCALE) ** 2 +
                 (LANDER_W / 2 + LEG_X_DIST + LEG_W / SCALE) ** 2) ** 0.5


class LunarLanderReachability(LunarLander):

    # in the LunarLander environment the variables LANDER_POLY, LEG_AWAY, LEG_DOWN, LEG_W, LEG_H
    # SIDE_ENGINE_HEIGHT, SIDE_ENGINE_AWAY, VIEWPORT_W and VIEWPORT_H are measured in pixels
    #
    # the x and y coordinates (and their time derivatives) used for physics calculations in the
    # simulator use those values scaled by 1 / SCALE
    #
    # the observations sent to the learning algorithm when reset() or step() is called use those
    # values scaled by SCALE / (2 * VIEWPORT_H) and SCALE / (2 * VIEWPORT_Y) and centered at
    # (2 * VIEWPORT_W) / SCALE and HELIPAD_Y + LEG_DOWN / SCALE for x and y respectively
    # theta_dot is scaled by 20.0 / FPS
    #
    # this makes reading the lunar_lander.py file difficult so I have tried to make clear what scale
    # is being used here by calling them: pixel scale, simulator scale, and observation scale

    def __init__(self):

        # in LunarLander init() calls reset() which calls step() so some variables need
        # to be set up before calling init() to prevent problems from variables not being defined

        self.before_parent_init = True

        # safety problem limits in simulator scale

        self.land_min_v = -1.6  # fastest that lander can be falling when it hits the ground

        self.land_min_x = W / (CHUNKS - 1) * (CHUNKS // 2 - 1)  # calc of edges of landing pad based
        self.land_max_x = W / (CHUNKS - 1) * (CHUNKS // 2 + 1)  # on calc in parent reset()

        self.theta_land_max = np.radians(15.0)  # most the lander can be tilted when landing
        self.theta_land_min = np.radians(-15.0)

        self.fly_min_x = 0  # first chunk
        self.fly_max_x = W / (CHUNKS - 1) * (CHUNKS - 1)  # last chunk

        self.fly_max_y = VIEWPORT_H / SCALE
        self.fly_min_y = HELIPAD_Y

        # set up state space bounds used in evaluating the q value function
        self.vx_bound = 10  # bounds centered at 0 so take negative for lower bound
        self.vy_bound = 10  # this is in simulator scale
        self.theta_bound = np.radians(90)
        self.theta_dot_bound = np.radians(50)

        super(LunarLanderReachability, self).__init__()

        self.before_parent_init = False

        # we don't use the states about whether the legs are touching so 6 dimensions total
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32)

        # this is the hack from above to make the ground flat
        self.np_random = RandomAlias

        self.bounds = np.array([[self.fly_min_x, self.fly_max_x],
                                [self.fly_min_y, self.fly_max_y],
                                [-self.vx_bound, self.vx_bound],
                                [-self.vy_bound, self.vy_bound],
                                [-self.theta_bound, self.theta_bound],
                                [-self.theta_dot_bound, self.theta_dot_bound]])

        # convert to observation scale so network can be evaluated
        self.bounds[:, 0] = self.simulator_scale_to_obs_scale(self.bounds[:, 0].T)
        self.bounds[:, 1] = self.simulator_scale_to_obs_scale(self.bounds[:, 1].T)

    def reset(self):
        """
        resets the environment accoring to a uniform distribution
        :return: current state as 6d NumPy array of floats
        """

        s = super(LunarLanderReachability, self).reset()

        # have to sample uniformly to get good coverage of the state space
        self.lander.position = np.random.uniform(low=[self.fly_min_x, self.fly_min_y],
                                                 high=[self.fly_max_x, self.fly_max_y])
        self.lander.angle = np.random.uniform(low=-self.theta_bound, high=self.theta_bound)

        # after lander position is set have to set leg positions to be where new lander position is
        self.legs[0].position = np.array([self.lander.position[0] + LEG_AWAY/SCALE,
                                          self.lander.position[1]])
        self.legs[1].position = np.array([self.lander.position[0] - LEG_AWAY/SCALE,
                                          self.lander.position[1]])

        # convert from simulator scale to observation scale
        s[0:3] = self.simulator_scale_to_obs_scale(np.array([self.lander.position[0],
                                                             self.lander.position[1],
                                                             self.lander.angle, 0, 0, 0]))[0:3]

        return s

    def step(self, action):
        if self.before_parent_init:
            r = None  # can't be computed
        else:
            # note that l function must be computed before environment steps see reamdme for proof
            r = self.signed_distance()

        s, _, done, info = super(LunarLanderReachability, self).step(action)

        s = s[:-2]  # chop off last two states since they aren't used

        # LunarLander will end when landed so very few states will be sampled near landing
        x, y = self.lander.position.x, self.lander.position.y
        done = x < self.fly_min_x or x > self.fly_max_x or y > self.fly_max_y
        return s, r, done, info

    def signed_distance(self):
        """

        :return: the signed distance of the environment at state s to the failure set
        """
        # all in simulation scale
        x = self.lander.position.x
        y = self.lander.position.y
        y_dot = self.lander.linearVelocity.y
        theta = self.lander.angle

        # compute l_fly from [ICRA 2019], use vectorized min for performance
        flying_distance = np.min([x - self.fly_min_x - LANDER_RADIUS,  # distance to left wall
                                  self.fly_max_x - x - LANDER_RADIUS,  # distance to right wall
                                  self.fly_max_y - y - LANDER_RADIUS,  # distance to ceiling
                                  y - self.fly_min_y - LANDER_RADIUS])  # distance to ground

        # compute l_land from [ICRA 2019]
        landing_distance = np.min([10 * (theta - self.theta_land_min),  # heading error multiply 10
                                   10 * (self.theta_land_max - theta),  # for similar scale of units
                                   x - self.land_min_x - LANDER_RADIUS,  # dist to left edge of landing pad
                                   self.land_max_x - x - LANDER_RADIUS,  # dist to right edge of landing pad
                                   y_dot - self.land_min_v])  # speed check

        # landing or flying is acceptable. a max is equivalent to or
        return max(flying_distance, landing_distance)

    @staticmethod
    def simulator_scale_to_obs_scale(state):
        """
        converts from simulator scale to observation scale see comment at top of class
        :param state: length 6 array of x, y, x_dot, y_dot, theta, theta_dot in simulator scale
        :return: length 6 array of x, y, x_dot, y_dot, theta, theta_dot in obs scale
        """
        x, y, x_dot, y_dot, theta, theta_dot = state
        return np.array([(x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
                         (y - (HELIPAD_Y + LEG_DOWN/SCALE)) / (VIEWPORT_H / SCALE / 2),
                         x_dot * (VIEWPORT_W / SCALE / 2) / FPS,
                         y_dot * (VIEWPORT_H / SCALE / 2) / FPS,
                         theta,
                         theta_dot])

    def get_state(self):
        """
        gets the current state of the environment
        :return: length 6 array of x, y, x_dot, y_dot, theta, theta_dot in simulator scale
        """
        return np.array([self.lander.position.x,
                        self.lander.position.y,
                        self.lander.linearVelocity.x,
                        self.lander.linearVelocity.y,
                        self.lander.angle,
                        self.lander.angularVelocity])


class RandomAlias:
    # Note: This is a little hacky. The LunarLander uses the instance attribute self.np_random to
    # pick the moon chunks placements and also determine the randomness in the dynamics and
    # starting conditions. The size argument is only used for determining the height of the
    # chunks so this can be used to set the height of the chunks. When low=-1.0 and high=1.0 the
    # dispersion on the particles is determined on line 247 in step LunarLander which makes the
    # dynamics probabilistic. Safety Bellman Equation assumes deterministic dynamics so we set that
    # to be constant

    @staticmethod
    def uniform(low, high, size=None):
        if size is None:
            if low == -1.0 and high == 1.0:
                return 0
            else:
                return np.random.uniform(low=low, high=high)
        else:
            return np.ones(12) * HELIPAD_Y  # this makes the ground flat
