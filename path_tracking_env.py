from abc import ABC
import gym
from collections import OrderedDict
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque


def shift_coordination(orig_x, orig_y, coordi_shift_x, coordi_shift_y):
    '''
    :param orig_x: original x
    :param orig_y: original y
    :param coordi_shift_x: coordi_shift_x along x axis
    :param coordi_shift_y: coordi_shift_y along y axis
    :return: shifted_x, shifted_y
    '''
    shifted_x = orig_x - coordi_shift_x
    shifted_y = orig_y - coordi_shift_y
    return shifted_x, shifted_y


def rotate_coordination(orig_x, orig_y, orig_d, coordi_rotate_rad):
    """
    :param orig_x: original x
    :param orig_y: original y
    :param orig_d: original rad
    :param coordi_rotate_d: coordination rotation d, positive if anti-clockwise, unit: rad
    :return:
    transformed_x, transformed_y, transformed_d
    """

    transformed_x = orig_x * tf.cos(coordi_rotate_rad) + orig_y * tf.sin(coordi_rotate_rad)
    transformed_y = -orig_x * tf.sin(coordi_rotate_rad) + orig_y * tf.cos(coordi_rotate_rad)
    transformed_d = orig_d - coordi_rotate_rad
    return transformed_x, transformed_y, transformed_d


def shift_and_rotate_coordination(orig_x, orig_y, orig_d, coordi_shift_x, coordi_shift_y, coordi_rotate_d):
    shift_x, shift_y = shift_coordination(orig_x, orig_y, coordi_shift_x, coordi_shift_y)
    transformed_x, transformed_y, transformed_d \
        = rotate_coordination(shift_x, shift_y, orig_d, coordi_rotate_d)
    return transformed_x, transformed_y, transformed_d


class VehicleDynamics(object):
    def __init__(self, ):
        self.vehicle_params = OrderedDict(C_f=88000.,  # front wheel cornering stiffness [N/rad]
                                          C_r=94000.,  # rear wheel cornering stiffness [N/rad]
                                          a=1.14,  # distance from CG to front axle [m]
                                          b=1.40,  # distance from CG to rear axle [m]
                                          mass=1500.,  # mass [kg]
                                          I_z=2420.,  # Polar moment of inertia at CG [kg*m^2]
                                          miu=1.0,  # tire-road friction coefficient
                                          g=9.81,  # acceleration of gravity [m/s^2]
                                          )

    def f_xu(self, states, actions):  # states and actions are tensors, [[], [], ...]
        with tf.name_scope('f_xu') as scope:
            v_y, r, v_x, delta_phi, x, delta_y = states[:, 0], states[:, 1], states[:, 2], states[:, 3], states[:, 4],\
                                                 states[:, 5]
            delta, a_x = actions[:, 0], actions[:, 1]

            C_f = tf.convert_to_tensor(self.vehicle_params['C_f'], dtype=tf.float32)
            C_r = tf.convert_to_tensor(self.vehicle_params['C_r'], dtype=tf.float32)
            a = tf.convert_to_tensor(self.vehicle_params['a'], dtype=tf.float32)
            b = tf.convert_to_tensor(self.vehicle_params['b'], dtype=tf.float32)
            mass = tf.convert_to_tensor(self.vehicle_params['mass'], dtype=tf.float32)
            I_z = tf.convert_to_tensor(self.vehicle_params['I_z'], dtype=tf.float32)
            miu = tf.convert_to_tensor(self.vehicle_params['miu'], dtype=tf.float32)
            g = tf.convert_to_tensor(self.vehicle_params['g'], dtype=tf.float32)

            F_zf, F_zr = b * mass * g / (a + b), a * mass * g / (a + b)
            F_xf = tf.where(a_x < 0, mass * a_x / 2, tf.zeros_like(a_x, dtype=tf.float32))
            F_xr = tf.where(a_x < 0, mass * a_x / 2, mass * a_x)
            miu_f = tf.sqrt(tf.square(miu * F_zf) - tf.square(F_xf)) / F_zf
            miu_r = tf.sqrt(tf.square(miu * F_zr) - tf.square(F_xr)) / F_zr
            alpha_f = tf.atan((v_y + a * r) / v_x) - delta
            alpha_r = tf.atan((v_y - b * r) / v_x)

            Ff_w1 = tf.square(C_f) / (3 * F_zf * miu_f)
            Ff_w2 = tf.pow(C_f, 3) / (27 * tf.pow(F_zf * miu_f, 2))
            F_yf_max = F_zf * miu_f

            Fr_w1 = tf.square(C_r) / (3 * F_zr * miu_r)
            Fr_w2 = tf.pow(C_r, 3) / (27 * tf.pow(F_zr * miu_r, 2))
            F_yr_max = F_zr * miu_r

            F_yf = - C_f * tf.tan(alpha_f) + Ff_w1 * tf.tan(alpha_f) * tf.abs(
                tf.tan(alpha_f)) - Ff_w2 * tf.pow(tf.tan(alpha_f), 3)
            F_yr = - C_r * tf.tan(alpha_r) + Fr_w1 * tf.tan(alpha_r) * tf.abs(
                tf.tan(alpha_r)) - Fr_w2 * tf.pow(tf.tan(alpha_r), 3)

            F_yf = tf.minimum(F_yf, F_yf_max)
            F_yf = tf.maximum(F_yf, -F_yf_max)

            F_yr = tf.minimum(F_yr, F_yr_max)
            F_yr = tf.maximum(F_yr, -F_yr_max)

            # tmp_f = tf.square(C_f * tf.tan(alpha_f)) / (27 * tf.square(miu_f * F_zf)) - C_f * tf.abs(tf.tan(alpha_f)) / (
            #         3 * miu_f * F_zf) + 1
            # tmp_r = tf.square(C_r * tf.tan(alpha_r)) / (27 * tf.square(miu_r * F_zr)) - C_r * tf.abs(tf.tan(alpha_r)) / (
            #         3 * miu_r * F_zr) + 1
            #
            # F_yf = -tf.sign(alpha_f) * tf.minimum(tf.abs(C_f * tf.tan(alpha_f) * tmp_f), tf.abs(miu_f * F_zf))
            # F_yr = -tf.sign(alpha_r) * tf.minimum(tf.abs(C_r * tf.tan(alpha_r) * tmp_r), tf.abs(miu_r * F_zr))

            state_deriv = [(F_yf * tf.cos(delta) + F_yr) / mass - v_x * r,
                           (a * F_yf * tf.cos(delta) - b * F_yr) / I_z,
                           a_x + v_y * r,  # - F_yf * tf.sin(delta) / mass,
                           r,
                           v_x * tf.cos(delta_phi) + v_y * tf.sin(delta_phi),
                           v_x * tf.sin(delta_phi) + v_y * tf.cos(delta_phi),
                           ]
            state_deriv_stack = tf.stack(state_deriv, axis=1)
        return state_deriv_stack

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
        with tf.name_scope('veh_model_next_states') as scope:
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
        self.curve_list = [(7.5, 200, 0.), (2.5, 300., 0.), (-5., 400., 0.)]

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

    def compute_y(self, x, delta_y):
        y_ref = self.compute_path_y(x)
        return delta_y + y_ref

    def compute_delta_y(self, x, y):
        y_ref = self.compute_path_y(x)
        return y - y_ref

    def compute_phi(self, x, delta_phi):
        phi_ref = self.compute_path_phi(x)
        return delta_phi + phi_ref

    def compute_delta_phi(self, x, phi):
        phi_ref = self.compute_path_phi(x)
        return phi - phi_ref

    def compute_state_delta_y(self, x, y):
        path_phi, path_y, path_x = self.compute_path_phi(x), self.compute_path_y(x), x
        return (y - path_y - tf.tan(path_phi) * x + tf.tan(path_phi) * path_x) / tf.sqrt(
            1. + tf.square(tf.tan(path_phi)))


class EnvironmentModel(object):  # all tensors
    def __init__(self):
        self.vehicle_dynamics = VehicleDynamics()
        self.path = ReferencePath()
        self.expected_vs = None
        self.base_frequency = 40
        self.out_frequency = 40
        self.obses = None
        self.veh_states = None
        # v_xs, v_ys, rs, delta_ys, delta_phis, exp_vs
        self.observation_space = {'low': np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.pi, -np.inf], dtype=np.float32),
                                  'high': np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf], dtype=np.float32),
                                  }
        self.action_space = {'low': np.array([-np.pi / 6, -3], dtype=np.float32),
                             'high': np.array([np.pi / 6, 3], dtype=np.float32),
                             }
        # self.history_positions = deque(maxlen=100)
        # plt.ion()

    def reset(self, obses):
        # self.history_positions.clear()

        self.obses = tf.clip_by_value(obses, self.observation_space['low'], self.observation_space['high'])
        self.veh_states = self._get_states(self.obses)
        # self.history_positions.append((self.veh_states[0][4], self.veh_states[0][5]))

    def compute_rewards(self, obses, actions):  # obses and actions are tensors
        with tf.name_scope('compute_reward') as scope:
            v_xs, v_ys, rs, delta_ys, delta_phis, exp_vs = obses[:, 0], obses[:, 1], obses[:, 2], obses[:, 3], \
                                                           obses[:, 4], obses[:, 5]
            deltas, a_xs = actions[:, 0], actions[:, 1]

            devi_v = -tf.square(v_xs - exp_vs)
            devi_y = -tf.square(delta_ys)
            devi_phi = -tf.square(delta_phis)
            punish_yaw_rate = -tf.square(rs)
            punish_delta = -tf.square(deltas)
            punish_a_x = -tf.square(a_xs)
            rewards = 0.001 * devi_v + 0.04 * devi_y + 0.1 * devi_phi + 0.02 * punish_yaw_rate + 0.1 * punish_delta + 0.001 * punish_a_x
            # rewards = 0.001 * devi_v + 0.04 * devi_y

        return rewards

    def _get_states(self, obses):  # obses are tensors
        with tf.name_scope('get_states') as scope:
            v_xs, v_ys, rs, delta_ys, delta_phis, exp_vs = obses[:, 0], obses[:, 1], obses[:, 2], obses[:, 3], \
                                                           obses[:, 4], obses[:, 5]
            self.expected_vs = exp_vs
            xs = tf.zeros_like(v_xs)
            veh_states = tf.stack([v_ys, rs, v_xs, delta_phis, xs, delta_ys], axis=1)
        return veh_states

    def _get_obses(self, veh_states):
        with tf.name_scope('get_obses') as scope:
            v_ys, rs, v_xs, delta_phis, xs, delta_ys = veh_states[:, 0], veh_states[:, 1], veh_states[:, 2], veh_states[:, 3], \
                                                       veh_states[:, 4], veh_states[:, 5]
            obses = tf.stack([v_xs, v_ys, rs, delta_ys, delta_phis, self.expected_vs], axis=1)
            obses = tf.clip_by_value(obses, self.observation_space['low'], self.observation_space['high'])
        return obses

    def rollout_out(self, actions):  # obses and actions are tensors, think of actions are in range [-1, 1]
        with tf.name_scope('model_step') as scope:
            deltas_norm, a_xs_norm = actions[:, 0], actions[:, 1]
            actions = tf.stack([deltas_norm * np.pi / 6, a_xs_norm * 3], axis=1)
            actions = tf.clip_by_value(actions, self.action_space['low'], self.action_space['high'])
            rewards = self.compute_rewards(self.obses, actions)
            self.veh_states = self.vehicle_dynamics.model_next_states(self.veh_states, actions,
                                                                      self.base_frequency, self.out_frequency)
            self.obses = self._get_obses(self.veh_states)
            # self.history_positions.append((self.veh_states[0][4], self.veh_states[0][5]))

        return self.obses, rewards

    # def render(self, mode='human'):
    #     plt.cla()
    #     v_y, r, v_x, phi, x, y = self.veh_states[0]
    #     v_x, v_y, r, delta_y, delta_phi, exp_v= self.obses[0]
    #     path_y, path_phi = self.path.compute_path_y(x).numpy(), self.path.compute_path_phi(x).numpy()
    #
    #     plt.title("Demo")
    #     range_x, range_y = 100, 100
    #     plt.axis('equal')
    #     path_xs = np.linspace(x - range_x / 2, x + range_x / 2, 1000)
    #     path_ys = np.zeros_like(path_xs)
    #     plt.plot(path_xs, path_ys)
    #
    #     history_positions = list(self.history_positions)
    #     history_xs = np.array(list(map(lambda x: x[0], history_positions)))
    #     history_ys = np.array(list(map(lambda x: x[1], history_positions)))
    #     plt.plot(history_xs, history_ys, 'g')
    #
    #     def draw_rotate_rec(x, y, a, l, w, color='black'):
    #         RU_x, RU_y, _ = rotate_coordination(l / 2, w / 2, 0, -a)
    #         RD_x, RD_y, _ = rotate_coordination(l / 2, -w / 2, 0, -a)
    #         LU_x, LU_y, _ = rotate_coordination(-l / 2, w / 2, 0, -a)
    #         LD_x, LD_y, _ = rotate_coordination(-l / 2, -w / 2, 0, -a)
    #         plt.plot([RU_x + x, RD_x + x], [RU_y + y, RD_y + y], color=color)
    #         plt.plot([RU_x + x, LU_x + x], [RU_y + y, LU_y + y], color=color)
    #         plt.plot([LD_x + x, RD_x + x], [LD_y + y, RD_y + y], color=color)
    #         plt.plot([LD_x + x, LU_x + x], [LD_y + y, LU_y + y], color=color)
    #
    #     draw_rotate_rec(x, y, phi, 4.8, 2.2)
    #     plt.text(x - range_x / 2 - 20, range_y / 2 - 3 * 2,
    #              'x: {:.2f}, y: {:.2f}, path_y: {:.2f}, delta_y: {:.2f}m'.format(x, y, path_y, delta_y))
    #     plt.text(x - range_x / 2 - 20, range_y / 2 - 3 * 3,
    #              r'phi: {:.2f}rad (${:.2f}\degree$), path_phi: {:.2f}rad (${:.2f}\degree$), delta_phi: {:.2f}rad (${:.2f}\degree$)'.format(
    #                  phi,
    #                  phi * 180 / np.pi,
    #                  path_phi,
    #                  path_phi * 180 / np.pi,
    #                  delta_phi, delta_phi * 180 / np.pi,
    #                  delta_phi, delta_phi * 180 / np.pi))
    #
    #     plt.text(x - range_x / 2 - 20, range_y / 2 - 3 * 4,
    #              'v_x: {:.2f}m/s, v_y: {:.2f}m/s, v: {:.2f}m/s (expected: {:.2f}m/s)'.format(v_x, v_y, np.sqrt(
    #                  v_x ** 2 + v_y ** 2), exp_v))
    #     plt.text(x - range_x / 2 - 20, range_y / 2 - 3 * 5, 'yaw_rate: {:.2f}rad/s'.format(r))
    #     plt.axis([x - range_x / 2, x + range_x / 2, -range_y / 2, range_y / 2])
    #
    #     plt.pause(0.001)
    #     plt.show()


class PathTrackingEnv(gym.Env, ABC):
    def __init__(self):
        self.vehicle_dynamics = VehicleDynamics()
        self.veh_state = None
        self.obs = None
        self.path = ReferencePath()
        self.expected_v = 20
        self.x = None
        self.y = None
        self.phi = None
        self.action = None
        self.base_frequency = 200
        self.simulation_frequency = 40
        self.simulation_time = 0
        # obs [v_x, v_y, r, delta_y, delta_phi, self.expected_v]
        self.observation_space = gym.spaces.Box(low=np.array([0, -10., -10., -np.inf, -10, 0]),
                                                high=np.array([35, 10., 10., np.inf, 10, 35]),
                                                dtype=np.float64)
        self.action_space = gym.spaces.Box(low=np.array([-np.pi / 6, -3]),
                                           high=np.array([np.pi / 6, 3]),
                                           dtype=np.float64)
        self.history_positions = deque(maxlen=100)
        plt.ion()

    def reset(self):
        self.history_positions.clear()
        self.simulation_time = 0
        self.expected_v = 20
        self.x = init_x = np.random.uniform(0, 600)

        init_delta_y = np.random.normal(0, 1)
        self.y = init_y = self.path.compute_y(init_x, init_delta_y).numpy()

        init_delta_phi = np.random.normal(0, np.pi / 9)
        self.phi = init_phi = self.path.compute_phi(init_x, init_delta_phi)

        init_v_x = np.random.uniform(20, 25)
        beta = np.random.normal(0, 0.15)
        init_v_y = init_v_x * np.tan(beta)
        init_r = np.random.normal(0, 0.3)
        self.veh_state = np.array([init_v_y, init_r, init_v_x, init_phi, init_x, init_y])
        self.obs = self._get_obs()
        self.history_positions.append((self.veh_state[4], self.veh_state[5]))
        return self.obs

    def _get_obs(self,):
        v_y, r, v_x, phi, x, y = self.veh_state
        delta_y = self.path.compute_delta_y(x, y).numpy()
        delta_phi = self.path.compute_delta_phi(x, phi).numpy()

        obs = np.array([v_x, v_y, r, delta_y, delta_phi, self.expected_v])
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        return obs

    def step(self, action):  # think of action is in range [-1, 1]
        delta_norm, a_x_norm = action
        action = delta_norm * np.pi / 6, a_x_norm * 3
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.simulation_time += 1 / self.simulation_frequency
        self.action = action
        reward = self.compute_reward(self.obs, self.action)

        veh_state_tensor = tf.convert_to_tensor(self.veh_state[np.newaxis, :], dtype=tf.float32)
        action_tensor = tf.convert_to_tensor(self.action[np.newaxis, :], dtype=tf.float32)
        self.veh_state = self.vehicle_dynamics.next_states(veh_state_tensor, action_tensor,
                                                           base_frequency=self.base_frequency,
                                                           out_frequency=self.simulation_frequency).numpy()[0]
        self.history_positions.append((self.veh_state[4], self.veh_state[5]))
        self.obs = self._get_obs()
        done = self.judge_done(self.obs)
        info = {}
        return self.obs, reward, done, info

    def compute_reward(self, obs, action):
        v_x, v_y, r, delta_y, delta_phi, exp_v = obs
        delta, a_x = action

        devi_v = -pow(v_x - exp_v, 2)
        devi_y = -pow(delta_y, 2)
        devi_phi = -pow(delta_phi, 2)
        punish_yaw_rate = -pow(r, 2)
        punish_delta = -pow(delta, 2)
        punish_a_x = -pow(a_x, 2)

        rewards = np.array([devi_v, devi_y, devi_phi, punish_yaw_rate, punish_delta, punish_a_x])
        coeffis = np.array([0.001, 0.04, 0.1, 0.02, 0.1, 0.001])
        # coeffis = np.array([0.001, 0.04, 0., 0., 0., 0.])

        reward = np.sum(rewards * coeffis)

        return reward

    def judge_done(self, obs):
        v_x, v_y, r, delta_y, delta_phi, exp_v = obs
        if abs(delta_y) > 3 or abs(delta_phi) > np.pi / 4. or v_x < 2 or abs(r) > 0.8:
            return 1
        else:
            return 0

    def render(self, mode='human'):
        plt.cla()
        v_y, r, v_x, phi, x, y = self.veh_state
        v_x, v_y, r, delta_y, delta_phi, exp_v= self.obs
        path_y, path_phi = self.path.compute_path_y(x).numpy(), self.path.compute_path_phi(x).numpy()

        plt.title("Demo")
        range_x, range_y = 100, 100
        plt.axis('equal')
        path_xs = np.linspace(x - range_x / 2, x + range_x / 2, 1000)
        path_ys = self.path.compute_path_y(path_xs).numpy()
        plt.plot(path_xs, path_ys)

        history_positions = list(self.history_positions)
        history_xs = np.array(list(map(lambda x: x[0], history_positions)))
        history_ys = np.array(list(map(lambda x: x[1], history_positions)))
        plt.plot(history_xs, history_ys, 'g')

        def draw_rotate_rec(x, y, a, l, w, color='black'):
            RU_x, RU_y, _ = rotate_coordination(l / 2, w / 2, 0, -a)
            RD_x, RD_y, _ = rotate_coordination(l / 2, -w / 2, 0, -a)
            LU_x, LU_y, _ = rotate_coordination(-l / 2, w / 2, 0, -a)
            LD_x, LD_y, _ = rotate_coordination(-l / 2, -w / 2, 0, -a)
            plt.plot([RU_x + x, RD_x + x], [RU_y + y, RD_y + y], color=color)
            plt.plot([RU_x + x, LU_x + x], [RU_y + y, LU_y + y], color=color)
            plt.plot([LD_x + x, RD_x + x], [LD_y + y, RD_y + y], color=color)
            plt.plot([LD_x + x, LU_x + x], [LD_y + y, LU_y + y], color=color)

        draw_rotate_rec(x, y, phi, 4.8, 2.2)
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


def test_path():
    path = ReferencePath()
    path_xs = np.linspace(500, 700, 1000)
    path_ys = path.compute_path_y(path_xs).numpy()
    plt.plot(path_xs, path_ys)
    plt.show()


def test_path_tracking_env():
    from mixed_pg_learner import judge_is_nan
    env = PathTrackingEnv()
    obs = env.reset()
    print(obs)
    judge_is_nan([obs])
    action = 2 * np.random.random((2,)) - 1
    action = [0, 1]
    for _ in range(10):
        done = 0
        while not done:
            obs, reward, done, info = env.step(action)
            action = 2 * np.random.random((2,)) - 1
            action = [0, 1]
            env.render()
        obs = env.reset()
        action = 2 * np.random.random((2,)) - 1
        action = [0, 1]

def test_environment_model():
    # v_x, v_y, r, delta_y, delta_phi, exp_v
    init_obs = tf.convert_to_tensor([[20, 0, 0, 20, -1, 20]], dtype=tf.float32)
    model = EnvironmentModel()
    model.reset(init_obs)
    action = tf.convert_to_tensor([[1, 0]], dtype=tf.float32)
    for i in range(100):
        obs, info = model.rollout_out(action)
        model.render()


if __name__ == '__main__':
    test_path_tracking_env()
