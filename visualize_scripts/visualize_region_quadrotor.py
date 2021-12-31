#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2021/12/27
# @Author  : Dongjie Yu (Tsinghua Univ.)
# @FileName: visualize_region_quadrotor.py
# =====================================

import os
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib as mpl

import json
import argparse
import datetime
import munch

import gym
import safe_control_gym

from policy import PolicyWithMu
from evaluator import EvaluatorWithCost


class Visualizer_quadrotor(object):
    def __init__(self, policy_dir, iteration,
                bound=(-1.5, 1.5, 0.5, 1.5),
                z_dot_list=[-1., 0., 1.],
                baseline=False):
        # 1 Load params and models
        params = json.loads(open(policy_dir + '/config.json').read())
        time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        test_log_dir = params['log_dir'] + '/tester/test-region-{}'.format(time_now)
        params.update(dict(mode='testing',
                           test_dir=policy_dir,
                           test_log_dir=test_log_dir, ))
        parser = argparse.ArgumentParser()
        for key, val in params.items():
            parser.add_argument("-" + key, default=val)
        args = parser.parse_args()
        args.config_eval = munch.munchify(args.config_eval)
        evaluator = EvaluatorWithCost(PolicyWithMu, args.env_id, args)
        evaluator.load_weights(os.path.join(policy_dir, 'models'), iteration)

        self.args = args
        self.evaluator = evaluator
        self.quadrotor_config = self.args.config_eval.quadrotor_config
        self.z_dot_list = z_dot_list

        # 2 Generate batch observations
        # 2.0 Generate ref trj
        TASK_INFO = self.quadrotor_config.task_info
        self.X_GOAL = self._generate_ref_trj(traj_type=TASK_INFO["trajectory_type"],
                                             traj_length=self.quadrotor_config.episode_len_sec,
                                             num_cycles=TASK_INFO["num_cycles"],
                                             traj_plane=TASK_INFO["trajectory_plane"],
                                             position_offset=TASK_INFO["trajectory_position_offset"],
                                             scaling=TASK_INFO["trajectory_scale"],
                                             sample_time=1./self.quadrotor_config.ctrl_freq
                                             )  # shape: (epi_len_sec * ctrl_freq, 6) = (360, 6)

        # 2.1 Generate location obses
        x = np.linspace(bound[0], bound[1], 100)
        z = np.linspace(bound[2], bound[3], 100)
        X, Z = np.meshgrid(x, z)
        flatten_x = X.ravel()
        flatten_z = Z.ravel()
        batch_obses = np.zeros((len(flatten_x), self.X_GOAL.shape[1] * 2), dtype=np.float32)  # (100*100, 12)
        assert batch_obses.shape == (100*100, 12)
        batch_obses[:, 0] = flatten_x
        batch_obses[:, 2] = flatten_z
        self.X = X
        self.Z = Z

        # 2.2 Allocate ref point for each obs
        batch_location = batch_obses[:, [0, 2]]
        target_location = self.X_GOAL[:, [0, 2]]
        indexes = self._find_closet_target(target_location, batch_location)
        batch_targets = self.X_GOAL[indexes, :]
        batch_obses[:, 6:] = batch_targets

        # 2.3 Copy batch obses to the num of z_dot_list
        self.batch_obses_list = []
        for z_dot in z_dot_list:
            obses = batch_obses.copy()
            obses[:, 3] = z_dot  # assign z_dot
            obses[:, 1] = batch_targets[:, 1]  # assign x_dot (same with target point)
            self.batch_obses_list.append(obses)

    def _generate_ref_trj(self,
                          traj_type="circle",
                          traj_length=6.0,
                          num_cycles=1,
                          traj_plane="xz",
                          position_offset=np.array([0, 0]),
                          scaling=1.0,
                          sample_time=0.01):
        def _circle(t,
                    traj_period,
                    scaling
                    ):
            """Computes the coordinates of a circle trajectory at time t.

            Args:
                t (float): The time at which we want to sample one trajectory point.
                traj_period (float): The period of the trajectory in seconds.
                scaling (float, optional): Scaling factor for the trajectory.

            Returns:
                float: The position in the first coordinate.
                float: The position in the second coordinate.
                float: The velocity in the first coordinate.
                float: The velocity in the second coordinate.

            """
            traj_freq = 2.0 * np.pi / traj_period
            coords_a = scaling * np.cos(traj_freq * t)
            coords_b = scaling * np.sin(traj_freq * t)
            coords_a_dot = -scaling * traj_freq * np.sin(traj_freq * t)
            coords_b_dot = scaling * traj_freq * np.cos(traj_freq * t)
            return coords_a, coords_b, coords_a_dot, coords_b_dot

        def _get_coordinates(t,
                             traj_type,
                             traj_period,
                             coord_index_a,
                             coord_index_b,
                             position_offset_a,
                             position_offset_b,
                             scaling
                             ):
            """Computes the coordinates of a specified trajectory at time t.

            Args:
                t (float): The time at which we want to sample one trajectory point.
                traj_type (str, optional): The type of trajectory (circle, square, figure8).
                traj_period (float): The period of the trajectory in seconds.
                coord_index_a (int): The index of the first coordinate of the trajectory plane.
                coord_index_b (int): The index of the second coordinate of the trajectory plane.
                position_offset_a (float): The offset in the first coordinate of the trajectory plane.
                position_offset_b (float): The offset in the second coordinate of the trajectory plane.
                scaling (float, optional): Scaling factor for the trajectory.

            Returns:
                ndarray: The position in x, y, z, at time t.
                ndarray: The velocity in x, y, z, at time t.

            """
            # Get coordinates for the trajectory chosen.
            if traj_type == "circle":
                coords_a, coords_b, coords_a_dot, coords_b_dot = _circle(
                    t, traj_period, scaling)
            else:
                raise NotImplementedError("Unknown shape of trajectory")
            # Initialize position and velocity references.
            pos_ref = np.zeros((3,))
            vel_ref = np.zeros((3,))
            # Set position and velocity references based on the plane of the trajectory chosen.
            pos_ref[coord_index_a] = coords_a + position_offset_a
            vel_ref[coord_index_a] = coords_a_dot
            pos_ref[coord_index_b] = coords_b + position_offset_b
            vel_ref[coord_index_b] = coords_b_dot
            return pos_ref, vel_ref

        # Get trajectory type.
        valid_traj_type = ["circle"]  # "square", "figure8"
        if traj_type not in valid_traj_type:
            raise ValueError("Trajectory type should be one of [circle, square, figure8].")
        traj_period = traj_length / num_cycles
        direction_list = ["x", "y", "z"]
        # Get coordinates indexes.
        if traj_plane[0] in direction_list and traj_plane[
            1] in direction_list and traj_plane[0] != traj_plane[1]:
            coord_index_a = direction_list.index(traj_plane[0])
            coord_index_b = direction_list.index(traj_plane[1])
        else:
            raise ValueError("Trajectory plane should be in form of ab, where a and b can be {x, y, z}.")
        # Generate time stamps.
        times = np.arange(0, traj_length, sample_time)
        pos_ref_traj = np.zeros((len(times), 3))
        vel_ref_traj = np.zeros((len(times), 3))
        # Compute trajectory points.
        for t in enumerate(times):
            pos_ref_traj[t[0]], vel_ref_traj[t[0]] = _get_coordinates(t[1],
                                                                      traj_type,
                                                                      traj_period,
                                                                      coord_index_a,
                                                                      coord_index_b,
                                                                      position_offset[0],
                                                                      position_offset[1],
                                                                      scaling)

        return np.vstack([
            pos_ref_traj[:, 0],
            vel_ref_traj[:, 0],
            pos_ref_traj[:, 2],
            vel_ref_traj[:, 2],
            np.zeros(pos_ref_traj.shape[0]),
            np.zeros(vel_ref_traj.shape[0])
        ]).transpose()

    def _find_closet_target(self, targets, points_list):
        import scipy
        mytree = scipy.spatial.cKDTree(targets)
        dist, indexes = mytree.query(points_list, k=1)
        return indexes

    def plot_region(self, metrics):
        def add_right_cax(ax, pad, width):
            '''
            在一个ax右边追加与之等高的cax.
            pad是cax与ax的间距.
            width是cax的宽度.
            '''
            axpos = ax.get_position()
            caxpos = mpl.transforms.Bbox.from_extents(
                axpos.x1 + pad,
                axpos.y0,
                axpos.x1 + pad + width,
                axpos.y1
            )
            cax = ax.figure.add_axes(caxpos)

            return cax

        for metric in metrics:
            assert metric in ['fea', 'cs', 'mu']

        fig, axes = plt.subplots(nrows=len(metrics), ncols=len(self.batch_obses_list), figsize=(12, 3))
        colorbar_types = []
        for i, obses in enumerate(self.batch_obses_list):
            preprocess_obs = self.evaluator.preprocessor.np_process_obses(obses)
            flatten_mu = self.evaluator.policy_with_value.compute_lam(preprocess_obs).numpy()

            processed_obses = self.evaluator.preprocessor.tf_process_obses(obses)
            actions, _ = self.evaluator.policy_with_value.compute_action(processed_obses)
            flatten_cost_q = self.evaluator.policy_with_value.compute_QC1(processed_obses, actions).numpy()
            flatten_fea_v = flatten_cost_q

            flatten_cs = np.multiply(flatten_fea_v, flatten_mu)
            NAME2VALUE = dict(zip(['fea', 'cs', 'mu'], [flatten_fea_v, flatten_cs, flatten_mu]))
            for j, metric in enumerate(metrics):
                if len(metrics) == 1 and len(self.z_dot_list) == 1:
                    sub_ax = axes
                elif len(axes.shape) > 1:
                    sub_ax = axes[j][i]
                elif len(self.z_dot_list) > 1:
                    sub_ax = axes[i]
                elif len(metrics) > 1:
                    sub_ax = axes[j]

                ct = sub_ax.contourf(self.X, self.Z, NAME2VALUE[metric].reshape(self.X.shape), cmap='Accent',
                                     levels=2)
                sub_ax.set_title(metric + ', ' + r'$\dot{z}=$' + str(self.z_dot_list[i]))

                if i == len(self.z_dot_list) - 1:
                    cax = add_right_cax(sub_ax, pad=0.01, width=0.02)
                    colorbar_types.append((ct, cax))
        fig.supxlabel(r'$x$')
        fig.supylabel(r'$z$')

        for i, pairs in enumerate(colorbar_types):
            plt.colorbar(pairs[0], cax=pairs[1])
        plt.show()


if __name__ == '__main__':
    vizer = Visualizer_quadrotor('../results/quadrotor/RAC-feasibility/2021-12-30-13-00-03-Zero_violation',
                                 2000000,
                                 z_dot_list=[-1., 0., 1.])
    vizer.plot_region(['fea'])