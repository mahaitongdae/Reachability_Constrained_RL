# Copyright (c) 2019–2020, The Regents of the University of California.
# All rights reserved.
#
# This file is subject to the terms and conditions defined in the LICENSE file
# included in this repository.
#
# Please contact the author(s) of this library if you have any questions.
# Authors: Neil Lugovoy   ( nflugovoy@berkeley.edu )

from setuptools import setup

setup(name="gym_reachability",
      version="0.0.1",
      install_requires=["gym", "numpy", "pyglet", "Box2D", "mujoco_py"])
