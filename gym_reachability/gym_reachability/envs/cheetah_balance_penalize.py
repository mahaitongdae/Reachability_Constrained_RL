# Copyright (c) 2019–2020, The Regents of the University of California.
# All rights reserved.
#
# This file is subject to the terms and conditions defined in the LICENSE file
# included in this repository.
#
# Please contact the author(s) of this library if you have any questions.
# Authors: Neil Lugovoy   ( nflugovoy@berkeley.edu )

import gym.spaces  # needed to avoid warning from gym
from gym_reachability.gym_reachability.envs.cheetah_balance import CheetahBalanceEnv
from ray.rllib.utils.annotations import override

"""
in this version of the cheetah problem the reward will be -1 for all states in the failure set
(head or front leg touching the ground) and 0 for all other states. thus only providing reward info
through penalization
"""


class CheetahBalancePenalizeEnv(CheetahBalanceEnv):

    @override(CheetahBalanceEnv)
    def signed_distance(self):
        return -1.0 if self.detect_contact() else 0.0

