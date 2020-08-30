#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/8/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: misc.py
# =====================================

import numpy as np
import random


def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)


def random_choice_with_index(obj_list):
    obj_len = len(obj_list)
    random_index = random.choice(list(range(obj_len)))
    random_value = obj_list[random_index]
    return random_value, random_index


def judge_is_nan(list_of_np_or_tensor):
    for m in list_of_np_or_tensor:
        if hasattr(m, 'numpy'):
            if np.any(np.isnan(m.numpy())):
                print(list_of_np_or_tensor)
                raise ValueError
        else:
            if np.any(np.isnan(m)):
                print(list_of_np_or_tensor)
                raise ValueError
