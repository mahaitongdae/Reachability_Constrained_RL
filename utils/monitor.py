#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/6/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: monitor.py
# =====================================

import time

from gym.core import Wrapper


class MonitorMultiAgent(Wrapper):
    def __init__(self, env):
        Wrapper.__init__(self, env=env)
        self.tstart = time.time()
        self.total_steps = 0
        self.num_agent = self.env.num_agent
        self.rewards = [[] for _ in range(self.num_agent)]
        self.needs_reset = [False for _ in range(self.num_agent)]

    def reset(self, **kwargs):
        self.reset_state()
        return self.env.reset(**kwargs)

    def reset_state(self):
        for i, needs_reset in enumerate(self.needs_reset):
            if needs_reset:
                self.rewards[i] = []
                self.needs_reset[i] = False

    def step(self, action):
        ob, rew, done, info = self.env.step(action)
        self.update(ob, rew, done, info)
        return ob, rew, done, info

    def update(self, ob, rew, done, info):
        epinfos = []
        for i in range(self.num_agent):
            self.rewards[i].append(rew[i])
            if done[i]:
                self.needs_reset[i] = True
                eprew = sum(self.rewards[i])
                eplen = len(self.rewards[i])
                epinfos.append({"r": round(eprew, 6), "l": eplen})
                assert isinstance(info, dict)
        if isinstance(info, dict):
            info['episode'] = epinfos

        self.total_steps += 1


class Monitor(Wrapper):
    def __init__(self, env):
        Wrapper.__init__(self, env=env)
        self.tstart = time.time()
        self.rewards = None
        self.needs_reset = True
        self.total_steps = 0

    def reset(self, **kwargs):
        self.reset_state()
        return self.env.reset(**kwargs)

    def reset_state(self):
        self.rewards = []
        self.needs_reset = False

    def step(self, action):
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        ob, rew, done, info = self.env.step(action)
        self.update(ob, rew, done, info)
        return ob, rew, done, info

    def update(self, ob, rew, done, info):
        self.rewards.append(rew)
        if done:
            self.needs_reset = True
            eprew = sum(self.rewards)
            eplen = len(self.rewards)
            epinfo = {"r": round(eprew, 6), "l": eplen, "t": round(time.time() - self.tstart, 6)}
            assert isinstance(info, dict)
            if isinstance(info, dict):
                info['episode'] = epinfo

        self.total_steps += 1

