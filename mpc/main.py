import time

import gym
import numpy as np
from scipy.optimize import minimize


def deal_with_phi_diff(phi_diff):
    phi_diff = np.where(phi_diff > 180., phi_diff - 360., phi_diff)
    phi_diff = np.where(phi_diff < -180., phi_diff + 360., phi_diff)
    return phi_diff


# class VehicleDynamics(object):
#     def __init__(self, ):
#         # self.vehicle_params = dict(C_f=88000.,  # front wheel cornering stiffness [N/rad]
#         #                            C_r=94000.,  # rear wheel cornering stiffness [N/rad]
#         #                            a=1.14,  # distance from CG to front axle [m]
#         #                            b=1.40,  # distance from CG to rear axle [m]
#         #                            mass=1500.,  # mass [kg]
#         #                            I_z=2420.,  # Polar moment of inertia at CG [kg*m^2]
#         #                            miu=1.0,  # tire-road friction coefficient
#         #                            g=9.81,  # acceleration of gravity [m/s^2]
#         #                            )
#         self.vehicle_params = dict(C_f=88000.,  # front wheel cornering stiffness [N/rad]
#                                    C_r=94000.,  # rear wheel cornering stiffness [N/rad]
#                                    a=1.14,  # distance from CG to front axle [m]
#                                    b=1.40,  # distance from CG to rear axle [m]
#                                    mass=1500.,  # mass [kg]
#                                    I_z=2420.,  # Polar moment of inertia at CG [kg*m^2]
#                                    miu=1.0,  # tire-road friction coefficient
#                                    g=9.81,  # acceleration of gravity [m/s^2]
#                                    )
#         a, b, mass, g = self.vehicle_params['a'], self.vehicle_params['b'], \
#                         self.vehicle_params['mass'], self.vehicle_params['g']
#         F_zf, F_zr = b * mass * g / (a + b), a * mass * g / (a + b)
#         self.vehicle_params.update(dict(F_zf=F_zf,
#                                         F_zr=F_zr))
#
#     def f_xu(self, states, actions):  # states and actions are tensors, [[], [], ...]
#         v_x, v_y, r, x, y, phi = states[:,0],states[:,1],states[:,2],states[:,3], states[:,4],states[:,5]
#         phi = phi * np.pi / 180.
#         steer, a_x = actions[:,0],actions[:,1]
#         C_f = self.vehicle_params['C_f']
#         C_r = self.vehicle_params['C_r']
#         a = self.vehicle_params['a']
#         b = self.vehicle_params['b']
#         mass = self.vehicle_params['mass']
#         I_z = self.vehicle_params['I_z']
#         miu = self.vehicle_params['miu']
#         g = self.vehicle_params['g']
#
#         F_zf, F_zr = b * mass * g / (a + b), a * mass * g / (a + b)
#         F_xf = np.where(a_x < 0, mass * a_x / 2, np.zeros_like(a_x))
#         F_xr = np.where(a_x < 0, mass * a_x / 2, mass * a_x)
#         miu_f = np.sqrt(np.square(miu * F_zf) - np.square(F_xf)) / F_zf
#         miu_r = np.sqrt(np.square(miu * F_zr) - np.square(F_xr)) / F_zr
#         alpha_f = np.arctan((v_y + a * r) / v_x) - steer
#         alpha_r = np.arctan((v_y - b * r) / v_x)
#
#         Ff_w1 = np.square(C_f) / (3 * F_zf * miu_f)
#         Ff_w2 = np.power(C_f, 3) / (27 * np.power(F_zf * miu_f, 2))
#         F_yf_max = F_zf * miu_f
#
#         Fr_w1 = np.square(C_r) / (3 * F_zr * miu_r)
#         Fr_w2 = np.power(C_r, 3) / (27 * np.power(F_zr * miu_r, 2))
#         F_yr_max = F_zr * miu_r
#
#         F_yf = - C_f * np.tan(alpha_f) + Ff_w1 * np.tan(alpha_f) * np.abs(
#             np.tan(alpha_f)) - Ff_w2 * np.power(np.tan(alpha_f), 3)
#         F_yr = - C_r * np.tan(alpha_r) + Fr_w1 * np.tan(alpha_r) * np.abs(
#             np.tan(alpha_r)) - Fr_w2 * np.power(np.tan(alpha_r), 3)
#
#         F_yf = np.minimum(F_yf, F_yf_max)
#         F_yf = np.minimum(F_yf, -F_yf_max)
#
#         F_yr = np.minimum(F_yr, F_yr_max)
#         F_yr = np.minimum(F_yr, -F_yr_max)
#
#         state_deriv = [a_x + v_y * r,
#                        (F_yf * np.cos(steer) + F_yr) / mass - v_x * r,
#                        (a * F_yf * np.cos(steer) - b * F_yr) / I_z,
#                        v_x * np.cos(phi) - v_y * np.sin(phi),
#                        v_x * np.sin(phi) + v_y * np.cos(phi),
#                        r * 180 / np.pi,
#                        ]
#
#         state_deriv_stack = np.stack(state_deriv, 1)
#         params = np.stack([miu_f, miu_r], 1)
#
#         return state_deriv_stack, params
#
#     def prediction(self, x_1, u_1, frequency, RK):
#         f_xu_1, params = self.f_xu(x_1, u_1)
#         x_next = f_xu_1 / frequency + x_1
#         return x_next, params

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

    def f_xu(self, states, actions, tau):  # states and actions are tensors, [[], [], ...]
        v_x, v_y, r, y, phi, x = states[:, 0], states[:, 1], states[:, 2], states[:, 3], states[:, 4], states[:, 5]
        steer, a_x = actions[:, 0], actions[:, 1]
        C_f = self.vehicle_params['C_f']
        C_r = self.vehicle_params['C_r']
        a = self.vehicle_params['a']
        b = self.vehicle_params['b']
        mass = self.vehicle_params['mass']
        I_z = self.vehicle_params['I_z']
        miu = self.vehicle_params['miu']
        g = self.vehicle_params['g']

        F_zf, F_zr = b * mass * g / (a + b), a * mass * g / (a + b)
        F_xf = np.where(a_x < 0, mass * a_x / 2, np.zeros_like(a_x))
        F_xr = np.where(a_x < 0, mass * a_x / 2, mass * a_x)
        miu_f = np.sqrt(np.square(miu * F_zf) - np.square(F_xf)) / F_zf
        miu_r = np.sqrt(np.square(miu * F_zr) - np.square(F_xr)) / F_zr

        next_state = [v_x + tau * (a_x + v_y * r),
                      (mass * v_y * v_x + tau * (
                                  a * C_f - b * C_r) * r - tau * C_f * steer * v_x - tau * mass * np.square(
                          v_x) * r) / (mass * v_x - tau * (C_f + C_r)),
                      (-I_z * r * v_x - tau * (a * C_f - b * C_r) * v_y + tau * a * C_f * steer * v_x) / (
                                  tau * (np.square(a) * C_f + np.square(b) * C_r) - I_z * v_x),
                      y + tau * (v_x * np.sin(phi) + v_y * np.cos(phi)),
                      phi + tau * r,
                      x + tau * (v_x * np.cos(phi) - v_y * np.sin(phi)),
                      ]

        return np.stack(next_state, 1), np.stack([miu_f, miu_r], 1)

    def prediction(self, x_1, u_1, frequency):
        x_next, next_params = self.f_xu(x_1, u_1, 1 / frequency)
        return x_next, next_params


class ModelPredictiveControl:
    def __init__(self, init_x, horizon):
        self.init_x = init_x
        self.horizon = horizon
        self.vehicle_dynamics = VehicleDynamics()
        self.base_frequency = 10.
        self.num_future_data = 0
        self.expected_vs = 20.

    def reset_init_x(self, init_x):
        self.init_x = init_x

    def plant_model(self, u, x):
        x_copy = x.copy()
        x_copy = self.compute_next_obses(x_copy[np.newaxis, :], u[np.newaxis, :])[0]
        return x_copy

    def compute_rew(self, obses, actions):
        v_xs, v_ys, rs, delta_ys, delta_phis, xs = obses[:, 0], obses[:, 1], obses[:, 2], \
                                               obses[:, 3], obses[:, 4], obses[:, 5]
        steers, a_xs = actions[:, 0], actions[:, 1]

        devi_v = -np.square(v_xs - self.expected_vs)
        devi_y = -np.square(delta_ys)
        devi_phi = -np.square(delta_phis)
        punish_yaw_rate = -np.square(rs)
        punish_steer = -np.square(steers)
        punish_a_x = -np.square(a_xs)

        rewards = 0.01 * devi_v + 0.04 * devi_y + 0.1 * devi_phi + 0.02 * punish_yaw_rate + \
                  0.05 * punish_steer + 0.0005 * punish_a_x

        return rewards

    def compute_next_obses(self, obses, actions):
        veh_states, _ = self.vehicle_dynamics.prediction(obses[:, :6], actions, self.base_frequency)
        v_xs, v_ys, rs, delta_ys, delta_phis, xs = veh_states[:, 0], veh_states[:, 1], veh_states[:, 2], \
                                                   veh_states[:, 3], veh_states[:, 4], veh_states[:, 5]
        v_xs = np.clip(v_xs, 1, 35)
        delta_phis = np.where(delta_phis > np.pi, delta_phis - 2 * np.pi, delta_phis)
        delta_phis = np.where(delta_phis <= -np.pi, delta_phis + 2 * np.pi, delta_phis)
        lists_to_stack = [v_xs, v_ys, rs, delta_ys, delta_phis, xs] + \
                         [delta_ys for _ in range(self.num_future_data)]
        return np.stack(lists_to_stack, axis=1)

    def cost_function(self, u):
        u = u.reshape(self.horizon, 2)
        loss = 0.
        x = self.init_x.copy()
        for i in range(0, self.horizon):
            u_i = u[i] * np.array([0.4, 3.])
            loss -= self.compute_rew(x[np.newaxis, :], u_i[np.newaxis, :])[0]
            x = self.plant_model(u_i, x)

        return loss


def run_mpc():
    horizon_list = [25]
    done = 0
    env = gym.make('PathTracking-v0', num_future_data=0, num_agent=1)

    for horizon in horizon_list:
        for i in range(1):
            data2plot = []
            obs = env.reset()
            mpc = ModelPredictiveControl(obs, horizon)
            bounds = [(-1., 1.), (-1., 1.)] * horizon
            u_init = np.zeros((horizon, 2))
            mpc.reset_init_x(obs[0])
            for _ in range(90):
                results = minimize(mpc.cost_function,
                                   x0=u_init.flatten(),
                                   method='SLSQP',
                                   bounds=bounds,
                                   tol=1e-1,
                                   options={'disp': True}
                                   )
                mpc_action = results.x

                # print(mpc_action)
                # print(results.success, results.message)
                # u_init = np.concatenate([mpc_action[2:], mpc_action[-2:]])
                if not results.success:
                    print('fail')
                    mpc_action = [0., 0.]
                mpc_action = mpc_action[:2][np.newaxis, :].astype(np.float32)
                obs, reward, done, info = env.step(mpc_action)
                mpc.reset_init_x(obs[0])
                env.render()


if __name__ == '__main__':
    run_mpc()

