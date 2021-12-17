#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2021/12/17
# @Author  : Haitong Ma (Tsinghua Univ.)
# =====================================

from gym.envs.registration import (
    registry,
    register,
    make,
    spec,
    load_env_plugins as _load_env_plugins,
)

# Hook to load plugins from entry points
_load_env_plugins()

# User defined
# --------

register(
    id='Air3d-v0',
    entry_point='dynamics.envs:Air3d',
    max_episode_steps=200,
)

register(
    id='PendulumEnv-v1',
    entry_point='dynamics.envs:PendulumEnv',
    max_episode_steps=200,
)

register(
    id='EmergencyBraking-v0',
    entry_point='dynamics.envs:EmergencyBraking',
    max_episode_steps=200,
)

register(
    id='UpperTriangle-v0',
    entry_point='dynamics.envs:UpperTriangle',
    max_episode_steps=200,
)
