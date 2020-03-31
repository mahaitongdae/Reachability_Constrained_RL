from abc import ABC
import gym
from collections import OrderedDict
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class VehicleDynamics(object):
    def __init__(self, ):
        self.vehicle_params = OrderedDict(C_f=88000,  # front wheel cornering stiffness [N/rad]
                                          C_r=94000,  # rear wheel cornering stiffness [N/rad]
                                          a=1.14,  # distance from CG to front axle [m]
                                          b=1.40,  # distance from CG to rear axle [m]
                                          mass=1500,  # mass [kg]
                                          I_z=2420,  # Polar moment of inertia at CG [kg*m^2]
                                          miu=1.0,  # tire-road friction coefficient
                                          g=9.8,  # acceleration of gravity [m/s^2]
                                          )

    def f_xu(self, states, actions):  # states and actions are tensors, [[], [], ...]
        v_y, r, v_x, phi, x, y = states[:, 0], states[:, 1], states[:, 2], states[:, 3], states[:, 4], states[:, 5]
        delta, a_x = actions[:, 0], actions[:, 1]
        C_f, C_r, a, b, mass, I_z, miu, g = list(
            map(lambda x: tf.convert_to_tensor(x, dtype=tf.float32), list(self.vehicle_params.values())))
        F_zf, F_zr = b * mass * g / (a + b), a * mass * g / (a + b)
        F_xf = tf.where(a_x < 0, mass * a_x / 2, tf.zeros_like(a_x, dtype=tf.float32))
        F_xr = tf.where(a_x < 0, mass * a_x / 2, tf.zeros_like(mass * a_x, dtype=tf.float32))
        miu_f = tf.sqrt(tf.square(miu * F_zf) - tf.square(F_xf)) / F_zf
        miu_r = tf.sqrt(tf.square(miu * F_zr) - tf.square(F_xr)) / F_zr
        alpha_f = tf.atan((v_y + a * r) / v_x) - delta
        alpha_r = tf.atan((v_y - b * r) / v_x)
        tmp_f = tf.square(C_f * tf.tan(alpha_f)) / (27 * tf.square(miu_f * F_zf)) - C_f * tf.abs(tf.tan(alpha_f)) / (
                3 * miu_f * F_zf) + 1
        tmp_r = tf.square(C_r * tf.tan(alpha_r)) / (27 * tf.square(miu_r * F_zr)) - C_r * tf.abs(tf.tan(alpha_r)) / (
                3 * miu_r * F_zr) + 1

        F_yf = -tf.sign(alpha_f) * tf.minimum(tf.abs(C_f * tf.tan(alpha_f) * tmp_f), tf.abs(miu_f * F_zf))
        F_yr = -tf.sign(alpha_r) * tf.minimum(tf.abs(C_r * tf.tan(alpha_r) * tmp_r), tf.abs(miu_r * F_zr))

        state_deriv = [(F_yf * tf.cos(delta) + F_yr) / mass - v_x * r,
                       (a * F_yf * tf.cos(delta) - b * F_yr) / I_z,
                       a_x + v_y * r - F_yf * tf.sin(delta) / mass,
                       r,
                       v_x * tf.cos(phi) + v_y * tf.sin(phi),
                       v_x * tf.sin(phi) + v_y * tf.cos(phi),
                       ]
        return tf.stack(state_deriv, axis=1)

    def next_states(self, xs, us, base_frequency=200, out_frequency=40):
        time = 0
        states = xs
        actions = us
        while time < 1 / out_frequency:
            delta_time = 1 / base_frequency
            time += delta_time
            state_deriv = self.f_xu(states, actions)
            delta_state = state_deriv * delta_time
            states = states + delta_state

        return states

    def model_next_states(self, xs, us, base_frequency=40, out_frequency=40):
        time = 0
        states = xs
        actions = us
        while time < 1 / out_frequency:
            delta_time = 1 / base_frequency
            time += delta_time
            state_deriv = self.f_xu(states, actions)
            delta_state = state_deriv * delta_time
            states = states + delta_state

        return states


class ReferencePath(object):
    def __init__(self):
        self.curve_list = [(10, 100, 0), (15, 150, 0), (20, 200, 0)]

    def compute_path_y(self, x):
        y = tf.zeros_like(x, dtype=tf.float32)
        for curve in self.curve_list:
            magnitude, T, shift = curve
            y += magnitude * tf.sin((x - shift) * 2 * tf.convert_to_tensor(np.pi) / T)
        return y

    def compute_path_phi(self, x):
        deriv = tf.zeros_like(x, dtype=tf.float32)
        for curve in self.curve_list:
            magnitude, T, shift = curve
            deriv += magnitude * 2 * tf.convert_to_tensor(np.pi) / T * tf.cos(
                (x - shift) * 2 * tf.convert_to_tensor(np.pi) / T)
        return tf.atan(deriv)

    def compute_y_deviation(self, x, y):
        y_ref = self.compute_path_y(x)
        return y - y_ref

    def compute_y(self, x, delta_y):
        y_ref = self.compute_path_y(x)
        return delta_y + y_ref

    def compute_phi(self, x, delta_phi):
        phi_ref = self.compute_path_phi(x)
        return delta_phi + phi_ref

    def compute_phi_deviation(self, x, phi):
        phi_ref = self.compute_path_phi(x)
        return phi - phi_ref


class EnvironmentModel(object):  # all tensors
    def __init__(self):
        self.vehicle_dynamics = VehicleDynamics()
        self.path = ReferencePath()
        self.expected_vs = None
        self.base_frequency = 40
        self.out_frequency = 40
        self.obses = None
        self.veh_states = None

    def reset(self, obses):
        self.obses = obses
        self.veh_states = self._get_states(self.obses)

    def compute_rewards(self, obses, actions):  # obses and actions are tensors
        v_xs, v_ys, rs, delta_ys, delta_phis, exp_vs, xs = obses[:, 0], obses[:, 1], obses[:, 2], obses[:, 3], \
                                                           obses[:, 4], obses[:, 5], obses[:, 6]
        v = tf.sqrt(tf.square(v_xs) + tf.square(v_ys))
        deltas, a_xs = actions[:, 0], actions[:, 1]

        devi_v = -tf.pow(v - exp_vs, 2)
        devi_y = -tf.pow(delta_ys, 2)
        devi_phi = -tf.pow(delta_phis, 2)
        punish_yaw_rate = -tf.abs(rs)
        punish_delta = -tf.abs(deltas)
        punish_a_x = -tf.abs(a_xs)

        rewards = 1 * devi_v + 1 * devi_y + 1 * devi_phi + 1 * punish_yaw_rate + 1 * punish_delta + 1 * punish_a_x

        return rewards

    def _get_states(self, obses):  # obses are tensors
        v_xs, v_ys, rs, delta_ys, delta_phis, exp_vs, xs = obses[:, 0], obses[:, 1], obses[:, 2], obses[:, 3], \
                                                           obses[:, 4], obses[:, 5], obses[:, 6]
        self.expected_vs = exp_vs
        path_ys, path_phis = self.path.compute_path_y(xs), self.path.compute_path_phi(xs)
        ys, phis = path_ys + delta_ys, path_phis + delta_phis
        veh_states = tf.stack([v_ys, rs, v_xs, phis, xs, ys], axis=1)
        return veh_states

    def _get_obses(self, veh_states):
        v_ys, rs, v_xs, phis, xs, ys = veh_states[:, 0], veh_states[:, 1], veh_states[:, 2], veh_states[:, 3], \
                                       veh_states[:, 4], veh_states[:, 5]
        delta_ys, delta_phis = self.path.compute_y_deviation(xs, ys), self.path.compute_phi_deviation(xs, phis)
        obses = tf.stack([v_xs, v_ys, rs, delta_ys, delta_phis, self.expected_vs, xs], axis=1)
        return obses

    def rollout_out(self, actions):  # obses and actions are tensors
        rewards = self.compute_rewards(self.obses, actions)
        self.veh_states = self.vehicle_dynamics.model_next_states(self.veh_states, actions,
                                                                  self.base_frequency, self.out_frequency)
        self.obses = self._get_obses(self.veh_states)
        return self.obses, rewards


class PathTrackingEnv(gym.Env, ABC):
    def __init__(self):
        self.vehicle_dynamics = VehicleDynamics()
        self.veh_state = None
        self.obs = None
        self.path = ReferencePath()
        self.expected_v = 10
        self.action = None
        self.base_frequency = 200
        self.simulation_frequency = 40
        self.simulation_time = 0
        self.observation_space = gym.spaces.Box(low=np.array([0, 0, -np.pi / 6, -np.inf, -np.pi, 0, -np.inf]),
                                                high=np.array([35, 1.5, np.pi / 6, np.inf, np.pi, 35, np.inf]),
                                                dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([-np.pi / 6, -5]),
                                           high=np.array([np.pi / 6, 3]),
                                           dtype=np.float32)
        plt.ion()

    def reset(self):
        self.simulation_time = 0
        self.expected_v = np.random.rand() * 20
        init_x = np.random.rand() * 1000
        init_path_y = self.path.compute_path_y(init_x).numpy()
        init_path_phi = self.path.compute_path_phi(init_x).numpy()
        init_y = init_path_y + 5 * 2 * (np.random.rand() - 0.5)
        init_phi = init_path_phi + 0.8 * 2 * (np.random.rand() - 0.5)
        init_v_x = np.random.rand() * 20
        init_v_y = 0
        init_r = 0
        self.veh_state = np.array([init_v_y, init_r, init_v_x, init_phi, init_x, init_y])
        self.obs = self._get_obs()
        return self.obs

    def _get_obs(self):
        v_y, r, v_x, phi, x, y = self.veh_state
        delta_y, delta_phi = self.path.compute_y_deviation(x, y).numpy(), self.path.compute_phi_deviation(x, phi).numpy()
        return np.array([v_x, v_y, r, delta_y, delta_phi, self.expected_v, x])

    def step(self, action):
        self.simulation_time += 1 / self.simulation_frequency
        self.action = action
        reward = self.compute_reward(self.obs, action)

        veh_state_tensor = tf.convert_to_tensor(self.veh_state[np.newaxis, :], dtype=tf.float32)
        action_tensor = tf.convert_to_tensor(action[np.newaxis, :], dtype=tf.float32)
        self.veh_state = self.vehicle_dynamics.next_states(veh_state_tensor, action_tensor,
                                                           base_frequency=self.base_frequency,
                                                           out_frequency=self.simulation_frequency).numpy()[0]
        self.obs = self._get_obs()
        done = self.judge_done(self.obs)
        info = {}
        return self.obs, reward, done, info

    def compute_reward(self, obs, action):
        v_x, v_y, r, delta_y, delta_phi, exp_v, x = obs
        v = np.sqrt(v_x ** 2 + v_y ** 2)
        delta, a_x = action

        devi_v = -pow(v - exp_v, 2)
        devi_y = -pow(delta_y, 2)
        devi_phi = -pow(delta_phi, 2)
        punish_yaw_rate = -abs(r)
        punish_delta = -abs(delta)
        punish_a_x = -abs(a_x)

        rewards = np.array([devi_v, devi_y, devi_phi, punish_yaw_rate, punish_delta, punish_a_x])
        coeffis = np.array([1, 1, 1, 1, 1, 1])

        reward = np.sum(rewards * coeffis)

        return reward

    def judge_done(self, obs):
        v_x, v_y, r, delta_y, delta_phi, expected_v, x = obs
        if abs(delta_y) > 10 or abs(delta_phi) > 1.6:
            return 1
        else:
            return 0

    def render(self, mode='human'):
        plt.cla()
        v_x, v_y, r, delta_y, delta_phi, exp_v, x = self.obs
        y, phi = self.path.compute_y(x, delta_y).numpy(), self.path.compute_phi(x, delta_phi).numpy()
        path_y, path_phi = self.path.compute_path_y(x).numpy(), self.path.compute_path_phi(x).numpy()

        plt.title("Demo")
        range_x, range_y = 100, 100
        plt.axis('equal')
        path_xs = np.linspace(x - range_x / 2, x + range_x / 2, 1000)
        path_ys = self.path.compute_path_y(path_xs).numpy()
        plt.plot(path_xs, path_ys)

        def rotate_coordination(orig_x, orig_y, orig_d, coordi_rotate_d):
            coordi_rotate_d_in_rad = coordi_rotate_d * np.pi / 180
            transformed_x = orig_x * np.cos(coordi_rotate_d_in_rad) + orig_y * np.sin(coordi_rotate_d_in_rad)
            transformed_y = -orig_x * np.sin(coordi_rotate_d_in_rad) + orig_y * np.cos(coordi_rotate_d_in_rad)
            transformed_d = orig_d - coordi_rotate_d
            if transformed_d > 180:
                while transformed_d > 180:
                    transformed_d = transformed_d - 360
            elif transformed_d <= -180:
                while transformed_d <= -180:
                    transformed_d = transformed_d + 360
            else:
                transformed_d = transformed_d
            return transformed_x, transformed_y, transformed_d

        def draw_rotate_rec(x, y, a, l, w, color='black'):
            RU_x, RU_y, _ = rotate_coordination(l / 2, w / 2, 0, -a)
            RD_x, RD_y, _ = rotate_coordination(l / 2, -w / 2, 0, -a)
            LU_x, LU_y, _ = rotate_coordination(-l / 2, w / 2, 0, -a)
            LD_x, LD_y, _ = rotate_coordination(-l / 2, -w / 2, 0, -a)
            plt.plot([RU_x + x, RD_x + x], [RU_y + y, RD_y + y], color=color)
            plt.plot([RU_x + x, LU_x + x], [RU_y + y, LU_y + y], color=color)
            plt.plot([LD_x + x, RD_x + x], [LD_y + y, RD_y + y], color=color)
            plt.plot([LD_x + x, LU_x + x], [LD_y + y, LU_y + y], color=color)

        draw_rotate_rec(x, y, phi * 180 / np.pi, 4.8, 2.2)
        plt.text(x - range_x / 2 - 20, range_y / 2 - 3, 'time: {:.2f}s'.format(self.simulation_time))
        plt.text(x - range_x / 2 - 20, range_y / 2 - 3 * 2,
                 'x: {:.2f}, y: {:.2f}, path_y: {:.2f}, delta_y: {:.2f}m'.format(x, y, path_y, delta_y))
        plt.text(x - range_x / 2 - 20, range_y / 2 - 3 * 3,
                 r'phi: {:.2f}rad (${:.2f}\degree$), path_phi: {:.2f}rad (${:.2f}\degree$), delta_phi: {:.2f}rad (${:.2f}\degree$)'.format(
                     phi,
                     phi * 180 / np.pi,
                     path_phi,
                     path_phi * 180 / np.pi,
                     delta_phi, delta_phi * 180 / np.pi,
                     delta_phi, delta_phi * 180 / np.pi))

        plt.text(x - range_x / 2 - 20, range_y / 2 - 3 * 4,
                 'v_x: {:.2f}m/s, v_y: {:.2f}m/s, v: {:.2f}m/s (expected: {:.2f}m/s)'.format(v_x, v_y, np.sqrt(
                     v_x ** 2 + v_y ** 2), exp_v))
        plt.text(x - range_x / 2 - 20, range_y / 2 - 3 * 5, 'yaw_rate: {:.2f}rad/s'.format(r))
        if self.action is not None:
            delta, a_x = self.action
            plt.text(x - range_x / 2 - 20, range_y / 2 - 3 * 6,
                     r'$\delta$: {:.2f}rad (${:.2f}\degree$), a_x: {:.2f}m/s^2'.format(delta, delta * 180 / np.pi, a_x))

        plt.axis([x - range_x / 2, x + range_x / 2, -range_y / 2, range_y / 2])

        plt.pause(0.001)
        plt.show()


def test_path_tracking_env():
    from mixed_pg_learner import judge_is_nan
    env = PathTrackingEnv()
    obs = env.reset()
    print(obs)
    judge_is_nan([obs])
    action = np.array([0, 1])
    for _ in range(10):
        done = 0
        while not done:
            obs, reward, done, info = env.step(action)
            env.render()
        obs = env.reset()


if __name__ == '__main__':
    test_path_tracking_env()
