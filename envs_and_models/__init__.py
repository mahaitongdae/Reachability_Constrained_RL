#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/12/2
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: __init__.py.py
# =====================================
from envs_and_models.inverted_double_pendulum_model import InvertedDoublePendulumModel
from envs_and_models.path_tracking_env import PathTrackingModel

NAME2MODELCLS = dict([('PathTracking-v0', PathTrackingModel),
                      ('InvertedDoublePendulum-v2', InvertedDoublePendulumModel)])

