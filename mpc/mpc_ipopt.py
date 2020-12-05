#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/8/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: mpc_ipopt.py
# =====================================

import argparse
import json
import math
import time

import gym
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from casadi import *

from envs_and_models.path_tracking_env import PathTrackingEnv
from policy import PolicyWithQs
from preprocessor import Preprocessor


def deal_with_phi_diff(phi_diff):
    phi_diff = np.where(phi_diff > 180., phi_diff - 360., phi_diff)
    phi_diff = np.where(phi_diff < -180., phi_diff + 360., phi_diff)
    return phi_diff


class LoadPolicy(object):
    def __init__(self, model_dir, iter):
        parser = argparse.ArgumentParser()
        params = json.loads(open(model_dir + '/config.json').read())
        for key, val in params.items():
            parser.add_argument("-" + key, default=val)
        self.args = parser.parse_args()
        env = PathTrackingEnv(num_agent=1)
        self.policy = PolicyWithQs(env.observation_space, env.action_space, self.args)
        self.policy.load_weights(model_dir, iter)
        self.preprocessor = Preprocessor(env.observation_space, self.args.obs_preprocess_type, self.args.reward_preprocess_type,
                                         self.args.obs_scale_factor, self.args.reward_scale_factor,
                                         gamma=self.args.gamma)
        # self.preprocessor.load_params(load_dir)
        self.run(env.reset())
        env.close()

    @tf.function
    def run(self, obs):
        processed_obs = self.preprocessor.tf_process_obses(obs)
        action, logp = self.policy.compute_action(processed_obs)
        return action


class TimerStat:
    def __init__(self, window_size=10):
        self._window_size = window_size
        self._samples = []
        self._units_processed = []
        self._start_time = None
        self._total_time = 0.0
        self.count = 0

    def __enter__(self):
        assert self._start_time is None, "concurrent updates not supported"
        self._start_time = time.time()

    def __exit__(self, type, value, tb):
        assert self._start_time is not None
        time_delta = time.time() - self._start_time
        self.push(time_delta)
        self._start_time = None

    def push(self, time_delta):
        self._samples.append(time_delta)
        if len(self._samples) > self._window_size:
            self._samples.pop(0)
        self.count += 1
        self._total_time += time_delta

    @property
    def mean(self):
        if not self._samples:
            return 0.0
        return float(np.mean(self._samples))


class VehicleDynamics(object):
    def __init__(self, ):
        self.vehicle_params = dict(C_f=-128915.5,  # front wheel cornering stiffness [N/rad]
                                   C_r=-85943.6,  # rear wheel cornering stiffness [N/rad]
                                   a=1.06,  # distance from CG to front axle [m]
                                   b=1.85,  # distance from CG to rear axle [m]
                                   mass=1412.,  # mass [kg]
                                   I_z=1536.7,  # Polar moment of inertia at CG [kg*m^2]
                                   miu=1.0,  # tire-road friction coefficient
                                   g=9.81,  # acceleration of gravity [m/s^2]
                                   )
        a, b, mass, g = self.vehicle_params['a'], self.vehicle_params['b'], \
                        self.vehicle_params['mass'], self.vehicle_params['g']
        F_zf, F_zr = b * mass * g / (a + b), a * mass * g / (a + b)
        self.vehicle_params.update(dict(F_zf=F_zf,
                                        F_zr=F_zr))

    def f_xu(self, x, u, tau):
        v_x, v_y, r, y, phi, x = x[0], x[1], x[2], x[3], x[4], x[5]
        steer, a_x = u[0], u[1]
        C_f = self.vehicle_params['C_f']
        C_r = self.vehicle_params['C_r']
        a = self.vehicle_params['a']
        b = self.vehicle_params['b']
        mass = self.vehicle_params['mass']
        I_z = self.vehicle_params['I_z']
        miu = self.vehicle_params['miu']
        g = self.vehicle_params['g']

        next_state = [v_x + tau * (a_x + v_y * r),
                      (mass * v_y * v_x + tau * (
                                  a * C_f - b * C_r) * r - tau * C_f * steer * v_x - tau * mass * pow(
                          v_x, 2) * r) / (mass * v_x - tau * (C_f + C_r)),
                      (-I_z * r * v_x - tau * (a * C_f - b * C_r) * v_y + tau * a * C_f * steer * v_x) / (
                                  tau * (pow(a, 2) * C_f + pow(b, 2) * C_r) - I_z * v_x),
                      y + tau * (v_x * sin(phi) + v_y * cos(phi)),
                      phi + tau * r,
                      x + tau * (v_x * cos(phi) - v_y * sin(phi)),
                      ]

        return next_state


class ModelPredictiveControl:
    def __init__(self, horizon):
        self.horizon = horizon
        self.vehicle_dynamics = VehicleDynamics()
        self.base_frequency = 10.
        self.num_future_data = 0
        self.expected_v = 20.
        self.DYNAMICS_DIM = 6
        self.ACTION_DIM = 2
        self._sol_dic = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0}

    def mpc_solver(self, x_init):
        """
        Solver of nonlinear MPC

        Parameters
        ----------
        x_init: list
            input state for MPC.

        Returns
        ----------
        state: np.array     shape: [predict_steps+1, state_dimension]
            state trajectory of MPC in the whole predict horizon.
        control: np.array   shape: [predict_steps, control_dimension]
            control signal of MPC in the whole predict horizon.
        """
        x = SX.sym('x', self.DYNAMICS_DIM)
        u = SX.sym('u', self.ACTION_DIM)

        # discrete dynamic model
        f = vertcat(*self.vehicle_dynamics.f_xu(x, u, 1./self.base_frequency))

        # Create solver instance
        F = Function("F", [x, u], [f])
        # G_f = Function('Gf', [x,u], [g])

        # Create empty NLP
        w = []
        lbw = []
        ubw = []
        lbg = []
        ubg = []
        G = []
        J = 0

        # Initial conditions
        Xk = MX.sym('X0', self.DYNAMICS_DIM)
        w += [Xk]
        lbw += x_init
        ubw += x_init

        for k in range(1, self.horizon + 1):
            # Local control
            Uname = 'U' + str(k - 1)
            Uk = MX.sym(Uname, self.ACTION_DIM)
            w += [Uk]
            lbw += [-1.2*math.pi, -3.]
            ubw += [1.2*math.pi, 3]
            # Gk = self.G_f(Xk,Uk)

            Fk = F(Xk, Uk)
            Xname = 'X' + str(k)
            Xk = MX.sym(Xname, self.DYNAMICS_DIM)

            # Dynamic Constriants
            G += [Fk - Xk]
            lbg += [0., 0., 0., 0., 0., 0.]
            ubg += [0., 0., 0., 0., 0., 0.]
            # G += [Gk]
            # lbg += [0.0, 0.0]
            # ubg += [0.0, 0.0]
            w += [Xk]
            lbw += [-inf, -inf, -inf, -inf, -inf, -inf]
            ubw += [inf, inf, inf, inf, inf, inf]
            # w += [Gk]
            # lbw += [-inf,-inf]
            # ubw += [inf, inf]
            # lbw += [-inf, -inf, -inf, -inf, -inf]
            # ubw += [inf, inf, inf, inf, inf]

            # Cost function
            F_cost = Function('F_cost', [x, u], [0.01 * (x[0]-self.expected_v) ** 2
                                                 + 0.04 * x[3] ** 2
                                                 + 0.1 * x[4] ** 2
                                                 + 0.02 * x[2] ** 2
                                                 + 5. * u[0] ** 2
                                                 + 0.05 * u[1] ** 2])
            J += F_cost(w[k * 2], w[k * 2 - 1])

        # Create NLP solver
        nlp = dict(f=J, g=vertcat(*G), x=vertcat(*w))
        S = nlpsol('S', 'ipopt', nlp, self._sol_dic)

        # Solve NLP
        r = S(lbx=lbw, ubx=ubw, x0=0, lbg=lbg, ubg=ubg)
        # print(r['x'])
        state_all = np.array(r['x'])
        state = np.zeros([self.horizon, self.DYNAMICS_DIM])
        control = np.zeros([self.horizon, self.ACTION_DIM])
        nt = self.DYNAMICS_DIM + self.ACTION_DIM  # total variable per step

        # save trajectories
        for i in range(self.horizon):
            state[i] = state_all[nt * i: nt * (i+1) - self.ACTION_DIM].reshape(-1)
            control[i] = state_all[nt * (i+1) - self.ACTION_DIM: nt * (i+1)].reshape(-1)
        return state, control


def run_mpc(rl_load_dir, rl_ite):
    horizon_list = [25]
    done = 0
    mpc_timer, rl_timer = TimerStat(), TimerStat()
    env4mpc = gym.make('PathTracking-v0', num_future_data=0, num_agent=1)
    env4rl = gym.make('PathTracking-v0', num_future_data=0, num_agent=1)

    rl_policy = LoadPolicy(rl_load_dir, rl_ite)

    for horizon in horizon_list:
        for i in range(1):
            data2plot = []
            obs = env4mpc.reset()
            obs4rl = env4rl.reset(init_obs=obs)
            mpc = ModelPredictiveControl(horizon)
            rew, rew4rl = [0.], [0.]
            for _ in range(100):
                with mpc_timer:
                    state, control = mpc.mpc_solver(list(obs[0]))
                mpc_action = control[0]
                with rl_timer:
                    rl_action_mpc = rl_policy.run(obs).numpy()[0]
                    rl_action = rl_policy.run(obs4rl).numpy()[0]

                data2plot.append(dict(mpc_obs=obs,
                                      rl_obs=obs4rl,
                                      mpc_action=mpc_action,
                                      rl_action=rl_action,
                                      rl_action_mpc=rl_action_mpc,
                                      mpc_time=mpc_timer.mean,
                                      rl_time=rl_timer.mean,
                                      mpc_rew=rew[0],
                                      rl_rew=rew4rl[0]))

                mpc_action = mpc_action[np.newaxis, :].astype(np.float32)
                obs, rew, _, _ = env4mpc.step(mpc_action)
                obs4rl, rew4rl, _, _ = env4rl.step(np.array([rl_action]))
                env4mpc.render()
            np.save('mpc_rl.npy', np.array(data2plot))


def plot_mpc_rl(file_dir, mpc_name):
    data = np.load(file_dir, allow_pickle=True)
    iteration = np.array([i for i in range(len(data))])
    mpc_delta_v = np.array([trunk['mpc_obs'][0][0]-20. for trunk in data])
    rl_delta_v = np.array([trunk['rl_obs'][0][0]-20. for trunk in data])
    mpc_delta_y = np.array([trunk['mpc_obs'][0][3] for trunk in data])
    rl_delta_y = np.array([trunk['rl_obs'][0][3] for trunk in data])
    mpc_delta_phi = np.array([trunk['mpc_obs'][0][4] for trunk in data])
    rl_delta_phi = np.array([trunk['rl_obs'][0][4] for trunk in data])
    mpc_steer = np.array([0.4*trunk['mpc_action'][0] for trunk in data])
    mpc_acc = np.array([3*trunk['mpc_action'][1] for trunk in data])
    mpc_time = np.array([trunk['mpc_time'] for trunk in data])
    mpc_rew = np.array([trunk['mpc_rew'] for trunk in data])
    rl_steer = np.array([0.4 * trunk['rl_action'][0] for trunk in data])
    rl_acc = np.array([3 * trunk['rl_action'][1] for trunk in data])
    rl_steer_mpc = np.array([0.4 * trunk['rl_action_mpc'][0] for trunk in data])
    rl_acc_mpc = np.array([3 * trunk['rl_action_mpc'][1] for trunk in data])
    rl_time = np.array([trunk['rl_time'] for trunk in data])
    rl_rew = np.array([trunk['rl_rew'] for trunk in data])

    print("mean_mpc_time: {}, mean_rl_time: {}".format(np.mean(mpc_time), np.mean(rl_time)))
    print("var_mpc_time: {}, var_rl_time: {}".format(np.var(mpc_time), np.var(rl_time)))
    print("mpc_delta_y_mse: {}, rl_delta_y_mse: {}".format(np.sqrt(np.mean(np.square(mpc_delta_y))),
                                                           np.sqrt(np.mean(np.square(rl_delta_y)))))
    print("mpc_delta_v_mse: {}, rl_delta_v_mse: {}".format(np.sqrt(np.mean(np.square(mpc_delta_v))),
                                                           np.sqrt(np.mean(np.square(rl_delta_v)))))
    print("mpc_delta_phi_mse: {}, rl_delta_phi_mse: {}".format(np.sqrt(np.mean(np.square(mpc_delta_phi))),
                                                               np.sqrt(np.mean(np.square(rl_delta_phi)))))
    print("mpc_rew_sum: {}, rl_rew_sum: {}".format(np.sum(mpc_rew), np.sum(rl_rew)))

    df_mpc = pd.DataFrame({'algorithms': mpc_name,
                           'iteration': iteration,
                           'steer': mpc_steer,
                           'acc': mpc_acc,
                           'time': mpc_time,
                           'delta_v': mpc_delta_v,
                           'delta_y': mpc_delta_y,
                           'delta_phi': mpc_delta_phi,
                           'rew': mpc_rew})
    df_rl = pd.DataFrame({'algorithms': 'AMPC',
                          'iteration': iteration,
                          'steer': rl_steer,
                          'acc': rl_acc,
                          'time': rl_time,
                          'delta_v': rl_delta_v,
                          'delta_y': rl_delta_y,
                          'delta_phi': rl_delta_phi,
                          'rew': rl_rew})
    df_rl_same_obs_as_mpc = pd.DataFrame({'algorithms': 'AMPC_sameobs_mpc',
                                          'iteration': iteration,
                                          'steer': rl_steer_mpc,
                                          'acc': rl_acc_mpc,
                                          'time': rl_time,
                                          'delta_v': mpc_delta_v,
                                          'delta_y': mpc_delta_y,
                                          'delta_phi': mpc_delta_phi,
                                          'rew': mpc_rew})
    total_df = df_mpc.append([df_rl, df_rl_same_obs_as_mpc], ignore_index=True)
    f1 = plt.figure(1)
    ax1 = f1.add_axes([0.155, 0.12, 0.82, 0.86])
    sns.lineplot(x="iteration", y="steer", hue="algorithms", data=total_df, linewidth=2, palette="bright",)
    # ax1.set_ylabel('Average Q-value Estimation Bias', fontsize=15)
    # ax1.set_xlabel("Million iterations", fontsize=15)
    # plt.xlim(0, 3)
    # plt.ylim(-40, 80)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)

    f2 = plt.figure(2)
    ax2 = f2.add_axes([0.155, 0.12, 0.82, 0.86])
    sns.lineplot(x="iteration", y="acc", hue="algorithms", data=total_df, linewidth=2, palette="bright", )
    # ax2.set_ylabel('Average Q-value Estimation Bias', fontsize=15)
    # ax2.set_xlabel("Million iterations", fontsize=15)
    # plt.xlim(0, 3)
    # plt.ylim(-40, 80)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)

    f3 = plt.figure(3)
    ax3 = f3.add_axes([0.155, 0.12, 0.82, 0.86])
    sns.lineplot(x="iteration", y="time", hue="algorithms", data=total_df, linewidth=2, palette="bright", )
    # ax3.set_ylabel('Average Q-value Estimation Bias', fontsize=15)
    # ax3.set_xlabel("Million iterations", fontsize=15)
    # plt.xlim(0, 3)
    # plt.ylim(-40, 80)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)

    f4 = plt.figure(4)
    ax4 = f4.add_axes([0.155, 0.12, 0.82, 0.86])
    sns.lineplot(x="iteration", y="delta_v", hue="algorithms", data=total_df, linewidth=2, palette="bright")
    # ax3.set_ylabel('Average Q-value Estimation Bias', fontsize=15)
    # ax3.set_xlabel("Million iterations", fontsize=15)
    # plt.xlim(0, 3)
    # plt.ylim(-40, 80)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)

    f5 = plt.figure(5)
    ax5 = f5.add_axes([0.155, 0.12, 0.82, 0.86])
    sns.lineplot(x="iteration", y="delta_y", hue="algorithms", data=total_df, linewidth=2, palette="bright")
    # ax3.set_ylabel('Average Q-value Estimation Bias', fontsize=15)
    # ax3.set_xlabel("Million iterations", fontsize=15)
    # plt.xlim(0, 3)
    # plt.ylim(-40, 80)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)

    f6 = plt.figure(6)
    ax6 = f6.add_axes([0.155, 0.12, 0.82, 0.86])
    sns.lineplot(x="iteration", y="delta_phi", hue="algorithms", data=total_df, linewidth=2, palette="bright")
    # ax3.set_ylabel('Average Q-value Estimation Bias', fontsize=15)
    # ax3.set_xlabel("Million iterations", fontsize=15)
    # plt.xlim(0, 3)
    # plt.ylim(-40, 80)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)

    f7 = plt.figure(7)
    ax7 = f7.add_axes([0.155, 0.12, 0.82, 0.86])
    sns.lineplot(x="iteration", y="rew", hue="algorithms", data=total_df, linewidth=2, palette="bright")
    # ax3.set_ylabel('Average Q-value Estimation Bias', fontsize=15)
    # ax3.set_xlabel("Million iterations", fontsize=15)
    # plt.xlim(0, 3)
    # plt.ylim(-40, 80)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.show()


if __name__ == '__main__':
    # run_mpc('../results/ampc/experiment-2020-09-16-14-54-54/models', 20000)
    plot_mpc_rl('./mpc_rl.npy', 'IPOPT')

