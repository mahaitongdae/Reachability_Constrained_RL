#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/12/4
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: dummy_vec_env.py
# =====================================
import numpy as np
from gym.core import Wrapper


class DummyVecEnv(Wrapper):  # todo: only for conti task
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
            state = init_obs[0]  # todo: only for inverted pendulum
            self.env.reset()
            self.env.set_state(state[:2], state[2:])  # todo: only for inverted pendulum
            return init_obs
        else:
            if self.done:
                self.obs = self.env.reset()
                return self.obs[np.newaxis, :]
            else:
                return self.obs[np.newaxis, :]


