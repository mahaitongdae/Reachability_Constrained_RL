#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/12/4
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: dummy_vec_env.py
# =====================================
import numpy as np
from gym.core import Wrapper


def _get_state4inverted_double_pendulumv2(obs):
    p, sintheta1, sintheta2, costheta1, costheta2, pdot, theta1dot, theta2dot \
        = obs[0], obs[1], obs[2], obs[3], obs[4], obs[5], obs[6], obs[7]
    theta1 = np.arctan2(sintheta1, costheta1)
    theta2 = np.arctan2(sintheta2, costheta2)
    return np.array([p, theta1, theta2, pdot, theta1dot, theta2dot])


class DummyVecEnv(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.num_agent = 1
        self.done = False
        self.obs = self.env.reset()

    def step(self, actions):
        self.obs, rew, self.done, info = self.env.step(actions[0])
        return self.obs[np.newaxis, :], np.array([rew]), np.array([self.done], np.bool), [info]

    def reset(self, **kwargs):
        if 'init_obs' in kwargs.keys():
            init_obs = kwargs.get('init_obs')
            state = _get_state4inverted_double_pendulumv2(init_obs[0])
            self.env.reset()
            self.env.set_state(state[:3], state[3:])
            return init_obs
        else:
            if self.done:
                self.obs = self.env.reset()
                return self.obs[np.newaxis, :]
            else:
                return self.obs[np.newaxis, :]


