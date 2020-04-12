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
            # obs:       v_xs, v_ys, rs, delta_ys, delta_phis, steer, a_x
            # veh_state: v_ys, rs, v_xs, delta_phis, xs, delta_ys
            v_y, r, v_x, delta_phi, x, delta_y, steer, a_x = states[:, 0], states[:, 1], states[:, 2],\
                                                             states[:, 3], states[:, 4], states[:, 5], \
                                                             states[:, 6], states[:, 7]
            steer_rate, a_x_rate = actions[:, 0], actions[:, 1]

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
            alpha_f = tf.atan((v_y + a * r) / v_x) - steer
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

            state_deriv = [(F_yf * tf.cos(steer) + F_yr) / mass - v_x * r,
                           (a * F_yf * tf.cos(steer) - b * F_yr) / I_z,
                           a_x + v_y * r,  # - F_yf * tf.sin(delta) / mass,
                           r,
                           v_x * tf.cos(delta_phi) + v_y * tf.sin(delta_phi),
                           v_x * tf.sin(delta_phi) + v_y * tf.cos(delta_phi),
                           steer_rate,
                           a_x_rate
                           ]
            state_deriv_stack = tf.stack(state_deriv, axis=1)
        return state_deriv_stack

    def next_states(self, xs, us, base_frequency=200, interval_times=5):
        time = 0
        states = xs
        actions = us
        for _ in range(interval_times):
            delta_time = 1 / base_frequency
            time += delta_time
            state_deriv = self.f_xu(states, actions)
            delta_state = state_deriv * delta_time
            states = states + delta_state
            v_y, r, v_x, delta_phi, x, delta_y, steer, a_x = states[:, 0], states[:, 1], states[:, 2],\
                                                             states[:, 3], states[:, 4], states[:, 5],\
                                                             states[:, 6], states[:, 7]
            changed_phis = tf.where(delta_phi > np.pi, delta_phi - 2 * np.pi, delta_phi)
            changed_phis = tf.where(changed_phis <= -np.pi, changed_phis + 2 * np.pi, changed_phis)
            cliped_steer = tf.clip_by_value(steer, -1.2 * np.pi / 9, 1.2 * np.pi / 9)
            cliped_a_x = tf.clip_by_value(a_x, -4.4, 4.4)

            states = tf.stack([v_y, r, v_x, changed_phis, x, delta_y, cliped_steer, cliped_a_x], 1)

        return states

    def model_next_states(self, xs, us, base_frequency=40, interval_times=1):
        with tf.name_scope('veh_model_next_states') as scope:
            time = 0
            states = xs
            actions = us
            for _ in range(interval_times):
                delta_time = 1 / base_frequency
                time += delta_time
                state_deriv = self.f_xu(states, actions)
                delta_state = state_deriv * delta_time
                states = states + delta_state
                # v_y, r, v_x, delta_phi, x, delta_y, steer, a_x = states[:, 0], states[:, 1], states[:, 2], \
                #                                                  states[:, 3], states[:, 4], states[:, 5], \
                #                                                  states[:, 6], states[:, 7]

                # changed_phis = tf.where(delta_phi > np.pi, delta_phi - 2 * np.pi, delta_phi)
                # changed_phis = tf.where(changed_phis <= -np.pi, changed_phis + 2 * np.pi, changed_phis)
                # cliped_steer = tf.clip_by_value(steer, -1.2 * np.pi / 9, 1.2 * np.pi / 9)
                # cliped_a_x = tf.clip_by_value(a_x, -3., 3.)

                # states = tf.stack([v_y, r, v_x, changed_phis, x, delta_y, cliped_steer, cliped_a_x], 1)

        return states


class ReferencePath(object):
    def __init__(self):
        self.curve_list = [(7.5, 200, 0.), (2.5, 300., 0.), (-5., 400., 0.)]

    def compute_path_y(self, x):
        y = np.zeros_like(x, dtype=np.float32)
        for curve in self.curve_list:
            magnitude, T, shift = curve
            y += magnitude * np.sin((x - shift) * 2 * np.pi / T)
        return y

    def compute_path_phi(self, x):
        deriv = np.zeros_like(x, dtype=np.float32)
        for curve in self.curve_list:
            magnitude, T, shift = curve
            deriv += magnitude * 2 * np.pi / T * np.cos(
                (x - shift) * 2 * np.pi / T)
        return np.arctan(deriv)

    def compute_y(self, x, delta_y):
        y_ref = self.compute_path_y(x)
        return delta_y + y_ref

    def compute_delta_y(self, x, y):
        y_ref = self.compute_path_y(x)
        return y - y_ref

    def compute_phi(self, x, delta_phi):
        phi_ref = self.compute_path_phi(x)
        phi = delta_phi + phi_ref
        phi[phi > np.pi] -= 2 * np.pi
        phi[phi <= -np.pi] += 2 * np.pi
        return phi

    def compute_delta_phi(self, x, phi):
        phi_ref = self.compute_path_phi(x)
        delta_phi = phi - phi_ref
        delta_phi[delta_phi > np.pi] -= 2 * np.pi
        delta_phi[delta_phi <= -np.pi] += 2 * np.pi
        return delta_phi


class EnvironmentModel(object):  # all tensors
    def __init__(self):
        self.vehicle_dynamics = VehicleDynamics()
        self.expected_vs = 25
        self.base_frequency = 40
        self.interval_times = 1
        self.obses = None
        self.veh_states = None
        # obs:       v_xs, v_ys, rs, delta_ys, delta_phis, steer, a_x
        # veh_state: v_ys, rs, v_xs, delta_phis, xs, delta_ys, steer, a_x
        self.observation_space = {
            'low': np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf], dtype=np.float32),
            'high': np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf], dtype=np.float32),
        }
        self.action_space = {'low': np.array([-np.pi / 9, -2], dtype=np.float32),
                             'high': np.array([np.pi / 9, 2], dtype=np.float32),
                             }

    def reset(self, obses):
        # self.obses = tf.clip_by_value(obses, self.observation_space['low'], self.observation_space['high'])
        self.obses = obses
        self.veh_states = self._get_states(self.obses)

    def compute_rewards(self, obses, actions):  # obses and actions are tensors
        with tf.name_scope('compute_reward') as scope:
            v_xs, v_ys, rs, delta_ys, delta_phis, steers, a_xs = obses[:, 0], obses[:, 1], obses[:, 2], \
                                                                 obses[:, 3], obses[:, 4], obses[:, 5], \
                                                                 obses[:, 6]
            steer_rates, a_x_rates = actions[:, 0], actions[:, 1]

            devi_v = -tf.square(v_xs - self.expected_vs)
            devi_y = -tf.square(delta_ys)
            devi_phi = -tf.square(delta_phis)
            punish_yaw_rate = -tf.square(rs)
            punish_steer = -tf.square(steers)
            punish_a_x = -tf.square(a_xs)
            punish_steer_rate = -tf.square(steer_rates)
            punish_a_x_rate = -tf.square(a_x_rates)

            rewards = 0.001 * devi_v + 0.04 * devi_y + 0.1 * devi_phi + 0.02 * punish_yaw_rate + \
                      0.05 * punish_steer + 0.0005 * punish_a_x + 0.05 * punish_steer_rate + 0.0005 * punish_a_x_rate

        return rewards

    def _get_states(self, obses):  # obses are tensors
        with tf.name_scope('get_states') as scope:
            v_xs, v_ys, rs, delta_ys, delta_phis, steer, a_x = obses[:, 0], obses[:, 1], obses[:, 2], \
                                                               obses[:, 3], obses[:, 4], obses[:, 5], \
                                                               obses[:, 6]
            xs = tf.zeros_like(v_xs)
            veh_states = tf.stack([v_ys, rs, v_xs, delta_phis, xs, delta_ys, steer, a_x], axis=1)
        return veh_states

    def _get_obses(self, veh_states):
        with tf.name_scope('get_obses') as scope:
            v_ys, rs, v_xs, delta_phis, xs, delta_ys, steer, a_x = veh_states[:, 0], veh_states[:, 1], \
                                                                   veh_states[:, 2], veh_states[:, 3], \
                                                                   veh_states[:, 4], veh_states[:, 5], \
                                                                   veh_states[:, 6], veh_states[:, 7]
            obses = tf.stack([v_xs, v_ys, rs, delta_ys, delta_phis, steer, a_x], axis=1)
            # obses = tf.clip_by_value(obses, self.observation_space['low'], self.observation_space['high'])
        return obses

    def rollout_out(self, actions):  # obses and actions are tensors, think of actions are in range [-1, 1]
        with tf.name_scope('model_step') as scope:
            steer_rate_norm, a_xs_rate_norm = actions[:, 0], actions[:, 1]
            actions = tf.stack([steer_rate_norm * np.pi / 9, a_xs_rate_norm * 2], axis=1)
            # actions = tf.clip_by_value(actions, self.action_space['low'], self.action_space['high'])
            rewards = self.compute_rewards(self.obses, actions)
            self.veh_states = self.vehicle_dynamics.model_next_states(self.veh_states, actions,
                                                                      self.base_frequency, self.interval_times)
            self.obses = self._get_obses(self.veh_states)

        return self.obses, rewards


class PathTrackingEnv(gym.Env, ABC):
    def __init__(self, **kwargs):
        self.vehicle_dynamics = VehicleDynamics()
        self.veh_state = None
        self.obs = None
        self.simulation_time = 0
        self.path = ReferencePath()
        self.action = None
        self.num_agent = kwargs['num_agent']
        self.expected_v = 25 * np.ones((self.num_agent,))
        self.done = np.zeros((self.num_agent,), dtype=np.int)
        self.base_frequency = 200
        self.interval_times = 5
        # obs [v_x, v_y, r, delta_y, delta_phi, self.expected_v, steer, a_x]
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]),
            high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]),
            dtype=np.float64)
        self.action_space = gym.spaces.Box(low=np.array([-np.pi / 9, -2]),
                                           high=np.array([np.pi / 9, 2]),
                                           dtype=np.float64)
        self.history_positions = deque(maxlen=100)
        plt.ion()
        # obs:       v_xs, v_ys, rs, delta_ys, delta_phis, steer, a_x (dim 7)
        # veh_state: v_ys, rs, v_xs, phis, xs, ys, steer, a_x (dim 8)

    def reset(self):
        if self.done[0] == 1:
            self.history_positions.clear()
        self.simulation_time = 0
        init_x = np.random.uniform(0, 600, (self.num_agent,))

        init_delta_y = np.random.normal(0, 1, (self.num_agent,))
        init_y = self.path.compute_y(init_x, init_delta_y)

        init_delta_phi = np.random.normal(0, np.pi / 9, (self.num_agent,))
        init_phi = self.path.compute_phi(init_x, init_delta_phi)

        init_v_x = np.random.uniform(15, 25, (self.num_agent,))
        beta = np.random.normal(0, 0.15, (self.num_agent,))
        init_v_y = init_v_x * np.tan(beta)
        init_r = np.random.normal(0, 0.3, (self.num_agent,))
        init_steer = np.random.uniform(-np.pi / 18, np.pi / 18, (self.num_agent,))
        init_a_x = np.random.uniform(-2, 2, (self.num_agent,))

        init_state = np.stack([init_v_y, init_r, init_v_x, init_phi, init_x, init_y, init_steer, init_a_x], 1)
        if self.veh_state is None:
            self.veh_state = init_state
        else:
            for i, done in enumerate(self.done):
                self.veh_state[i, :] = np.where(done == 1, init_state[i, :], self.veh_state[i, :])

        self.obs = self._get_obs()
        self.history_positions.append((self.veh_state[0, 4], self.veh_state[0, 5]))
        return self.obs

    def _get_obs(self, ):
        v_y, r, v_x, phi, x, y, steer, a_x = self.veh_state[:, 0], self.veh_state[:, 1], self.veh_state[:, 2], \
                                             self.veh_state[:, 3], self.veh_state[:, 4], self.veh_state[:, 5], \
                                             self.veh_state[:, 6], self.veh_state[:, 7]
        delta_y = self.path.compute_delta_y(x, y)
        delta_phi = self.path.compute_delta_phi(x, phi)

        obs = np.stack([v_x, v_y, r, delta_y, delta_phi, steer, a_x], 1)
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        return obs

    def step(self, action):  # think of action is in range [-1, 1]
        steer_rate_norm, a_x_rate_norm = action[:, 0], action[:, 1]
        action = np.stack([steer_rate_norm * np.pi / 9, a_x_rate_norm * 2], 1)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.simulation_time += self.interval_times * 1 / self.base_frequency
        self.action = action
        reward = self.compute_reward(self.obs, self.action)

        veh_state_tensor = tf.convert_to_tensor(self.veh_state, dtype=tf.float32)
        action_tensor = tf.convert_to_tensor(self.action, dtype=tf.float32)
        self.veh_state = self.vehicle_dynamics.next_states(veh_state_tensor, action_tensor,
                                                           base_frequency=self.base_frequency,
                                                           interval_times=self.interval_times).numpy()
        self.history_positions.append((self.veh_state[0, 4], self.veh_state[0, 5]))
        self.obs = self._get_obs()
        self.done = self.judge_done(self.obs)
        info = {}
        return self.obs, reward, self.done, info

    def compute_reward(self, obs, action):
        v_x, v_y, r, delta_y, delta_phi, steer, a_x = obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3], \
                                                      obs[:, 4], obs[:, 5], obs[:, 6]
        steer_rate, a_x_rate = action[:, 0], action[:, 1]

        devi_v = -np.power(v_x - self.expected_v, 2)
        devi_y = -np.power(delta_y, 2)
        devi_phi = -np.power(delta_phi, 2)
        punish_yaw_rate = -np.power(r, 2)
        punish_steer = -np.power(steer, 2)
        punish_a_x = -np.power(a_x, 2)
        punish_steer_rate = -tf.square(steer_rate)
        punish_a_x_rate = -tf.square(a_x_rate)

        rewards = np.stack([devi_v, devi_y, devi_phi, punish_yaw_rate, punish_steer,
                            punish_a_x, punish_steer_rate, punish_a_x_rate], 1)
        coeffis = np.array([0.001, 0.04, 0.1, 0.02, 0.05, 0.0005, 0.05, 0.0005])

        reward = np.sum(rewards * coeffis, 1)

        return reward

    def judge_done(self, obs):
        v_x, v_y, r, delta_y, delta_phi, steer, a_x = obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3], obs[:, 4], \
                                                      obs[:, 5], obs[:, 6]
        done = (np.abs(delta_y) > 3) | (np.abs(delta_phi) > np.pi / 4.) | (v_x < 2) | \
               (np.abs(r) > 0.8) | (np.abs(steer) > 1.1 * np.pi / 9) | (np.abs(a_x) > 2.9)
        return done

    def render(self, mode='human'):
        plt.cla()
        v_y, r, v_x, phi, x, y, steer, a_x = self.veh_state[0, 0], self.veh_state[0, 1], self.veh_state[0, 2], \
                                             self.veh_state[0, 3], self.veh_state[0, 4], self.veh_state[0, 5], \
                                             self.veh_state[0, 6], self.veh_state[0, 7]
        v_x, v_y, r, delta_y, delta_phi, steer, a_x = self.obs[0, 0], self.obs[0, 1], self.obs[0, 2], self.obs[0, 3], \
                                                      self.obs[0, 4], self.obs[0, 5], self.obs[0, 6]
        path_y, path_phi = self.path.compute_path_y(x), self.path.compute_path_phi(x)

        plt.title("Demo")
        range_x, range_y = 100, 100
        plt.axis('equal')
        path_xs = np.linspace(x - range_x / 2, x + range_x / 2, 1000)
        path_ys = self.path.compute_path_y(path_xs)
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
                 ))

        plt.text(x - range_x / 2 - 20, range_y / 2 - 3 * 4,
                 'v_x: {:.2f}m/s (expected: {:.2f}m/s), v_y: {:.2f}m/s'.format(v_x, self.expected_v[0], v_y))
        plt.text(x - range_x / 2 - 20, range_y / 2 - 3 * 5, 'yaw_rate: {:.2f}rad/s'.format(r))
        plt.text(x - range_x / 2 - 20, range_y / 2 - 3 * 6,
                 r'steer: {:.2f}rad (${:.2f}\degree$), a_x: {:.2f}m/s^2'.format(steer, steer * 180 / np.pi, a_x))

        if self.action is not None:
            steer_rate, a_x_rate = self.action[0, 0], self.action[0, 1]
            plt.text(x - range_x / 2 - 20, range_y / 2 - 3 * 7,
                     r'steer_rate: {:.2f}rad/s (${:.2f}\degree$/s), a_x_rate: {:.2f}m/s^3'.format(steer_rate,
                                                                                                  steer_rate * 180 / np.pi,
                                                                                                  a_x_rate))

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
    env = PathTrackingEnv(num_agent=1)
    obs = env.reset()
    print(obs)
    judge_is_nan([obs])
    action = np.array([[0, 1]])
    for _ in range(1000):
        obs, reward, done, info = env.step(action)
        env.render()
        # env.reset()


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
