#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/8/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: evaluator.py
# =====================================

import copy
import logging
import os
from copy import deepcopy

import gym
import numpy as np

import dynamics
import safe_control_gym
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make

from preprocessor import Preprocessor
from utils.dummy_vec_env import DummyVecEnv
from utils.misc import TimerStat

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Evaluator(object):
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    def __init__(self, policy_cls, env_id, args):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        self.args = args
        kwargs = copy.deepcopy(vars(self.args))
        if self.args.env_id == 'PathTracking-v0':
            self.env = gym.make(self.args.env_id, num_agent=self.args.num_eval_agent, num_future_data=self.args.num_future_data)
        else:
            env = gym.make(self.args.env_id)
            self.env = DummyVecEnv(env)
        self.policy_with_value = policy_cls(**kwargs)
        self.iteration = 0
        if self.args.mode == 'training':
            self.log_dir = self.args.log_dir + '/evaluator'
        else:
            self.log_dir = self.args.test_log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.preprocessor = Preprocessor(**kwargs)

        self.writer = self.tf.summary.create_file_writer(self.log_dir)
        self.stats = {}
        self.eval_timer = TimerStat()
        self.eval_times = 0

    def get_stats(self):
        self.stats.update(dict(eval_time=self.eval_timer.mean))
        return self.stats

    def load_weights(self, load_dir, iteration):
        self.policy_with_value.load_weights(load_dir, iteration)

    def load_ppc_params(self, load_dir):
        self.preprocessor.load_params(load_dir)

    def evaluate_saved_model(self, model_load_dir, ppc_params_load_dir, iteration):
        self.load_weights(model_load_dir, iteration)
        self.load_ppc_params(ppc_params_load_dir)

    def run_an_episode(self, steps=None, render=True):
        obs_list = []
        action_list = []
        reward_list = []
        info_list = []
        done = 0
        obs = self.env.reset()
        if render: self.env.render()
        if steps is not None:
            for _ in range(steps):
                processed_obs = self.preprocessor.tf_process_obses(obs)
                action = self.policy_with_value.compute_mode(processed_obs)
                obs_list.append(obs[0])
                action_list.append(action[0])
                obs, reward, done, info = self.env.step(action.numpy())
                if render: self.env.render()
                reward_list.append(reward[0])
                info_list.append(info[0])
        else:
            while not done:
                processed_obs = self.preprocessor.tf_process_obses(obs)
                action = self.policy_with_value.compute_mode(processed_obs)
                obs_list.append(obs[0])
                action_list.append(action[0])
                obs, reward, done, info = self.env.step(action.numpy())
                if render: self.env.render()
                reward_list.append(reward[0])
                info_list.append(info[0])
        episode_return = sum(reward_list)
        episode_len = len(reward_list)
        info_dict = dict()
        for key in info_list[0].keys():
            info_key = list(map(lambda x: x[key], info_list))
            mean_key = sum(info_key) / len(info_key)
            info_dict.update({key: mean_key})
        info_dict.update(dict(obs_list=np.array(obs_list),
                              action_list=np.array(action_list),
                              reward_list=np.array(reward_list),
                              episode_return=episode_return,
                              episode_len=episode_len))
        return info_dict

    def run_n_episodes(self, n):
        metrics_list = []
        for _ in range(n):
            logger.info('logging {}-th episode'.format(_))
            episode_info = self.run_an_episode(self.args.fixed_steps, self.args.eval_render)
            metrics_list.append(self.metrics_for_an_episode(episode_info))
        out = {}
        for key in metrics_list[0].keys():
            value_list = list(map(lambda x: x[key], metrics_list))
            out.update({key: sum(value_list)/len(value_list)})
        return metrics_list, out

    def run_n_episodes_parallel(self, n):
        logger.info('logging {} episodes in parallel'.format(n))
        metrics_list = []
        obses_list = []
        actions_list = []
        rewards_list = []
        obses = self.env.reset()
        if self.args.eval_render: self.env.render()
        for _ in range(self.args.fixed_steps):
            processed_obses = self.preprocessor.tf_process_obses(obses)
            actions = self.policy_with_value.compute_mode(processed_obses)
            obses_list.append(obses)
            actions_list.append(actions)
            obses, rewards, dones, _ = self.env.step(actions.numpy())
            if self.args.eval_render: self.env.render()
            rewards_list.append(rewards)
        for i in range(n):
            obs_list = [obses[i] for obses in obses_list]
            action_list = [actions[i] for actions in actions_list]
            reward_list = [rewards[i] for rewards in rewards_list]
            episode_return = sum(reward_list)
            episode_len = len(reward_list)
            info_dict = dict()
            info_dict.update(dict(obs_list=np.array(obs_list),
                                  action_list=np.array(action_list),
                                  reward_list=np.array(reward_list),
                                  episode_return=episode_return,
                                  episode_len=episode_len))
            metrics_list.append(self.metrics_for_an_episode(info_dict))
        out = {}
        for key in metrics_list[0].keys():
            value_list = list(map(lambda x: x[key], metrics_list))
            out.update({key: sum(value_list) / len(value_list)})
        return metrics_list, out


    def metrics_for_an_episode(self, episode_info):  # user defined, transform episode info dict to metric dict
        key_list = ['episode_return', 'episode_len']
        episode_return = episode_info['episode_return']
        episode_len = episode_info['episode_len']
        value_list = [episode_return, episode_len]
        if self.args.env_id == 'PathTracking-v0':
            delta_v_list = list(map(lambda x: x[0], episode_info['obs_list']))
            delta_y_list = list(map(lambda x: x[3], episode_info['obs_list']))
            delta_phi_list = list(map(lambda x: x[4], episode_info['obs_list']))
            steer_list = list(map(lambda x: x[0]*1.2 * np.pi / 9, episode_info['action_list']))
            acc_list = list(map(lambda x: x[1]*3., episode_info['action_list']))

            rew_list = episode_info['reward_list']
            stationary_rew_mean = sum(rew_list[20:])/len(rew_list[20:])

            delta_y_mse = np.sqrt(np.mean(np.square(np.array(delta_y_list))))
            delta_phi_mse = np.sqrt(np.mean(np.square(np.array(delta_phi_list))))
            delta_v_mse = np.sqrt(np.mean(np.square(np.array(delta_v_list))))
            steer_mse = np.sqrt(np.mean(np.square(np.array(steer_list))))
            acc_mse = np.sqrt(np.mean(np.square(np.array(acc_list))))
            key_list.extend(['delta_y_mse', 'delta_phi_mse', 'delta_v_mse',
                             'stationary_rew_mean', 'steer_mse', 'acc_mse'])
            value_list.extend([delta_y_mse, delta_phi_mse, delta_v_mse,
                               stationary_rew_mean, steer_mse, acc_mse])
        elif self.args.env_id == 'InvertedPendulumConti-v0':
            x_list = list(map(lambda x: x[0], episode_info['obs_list']))
            theta_list = list(map(lambda x: x[1], episode_info['obs_list']))
            xdot_list = list(map(lambda x: x[2], episode_info['obs_list']))
            thetadot_list = list(map(lambda x: x[3], episode_info['obs_list']))
            x_mean, x_var = np.mean(np.array(x_list)), np.var(np.array(x_list))
            theta_mean, theta_var = np.mean(np.array(theta_list)), np.var(np.array(theta_list))
            xdot_mean, xdot_var = np.mean(np.array(xdot_list)), np.var(np.array(xdot_list))
            thetadot_mean, thetadot_var = np.mean(np.array(thetadot_list)), np.var(np.array(thetadot_list))
            x_mse, theta_mse = np.sqrt(np.mean(np.square(np.array(x_list)))),\
                               np.sqrt(np.mean(np.square(np.array(theta_list))))

            xdot_mse, thetadot_mse = np.sqrt(np.mean(np.square(np.array(xdot_list)))),\
                                     np.sqrt(np.mean(np.square(np.array(thetadot_list))))
            x_mse_25, theta_mse_25 = np.sqrt(np.mean(np.square(np.array(x_list)[:25]))), \
                                     np.sqrt(np.mean(np.square(np.array(theta_list)[:25])))
            xdot_mse_25, thetadot_mse_25 = np.sqrt(np.mean(np.square(np.array(xdot_list)[:25]))), \
                                     np.sqrt(np.mean(np.square(np.array(thetadot_list)[:25])))
            key_list.extend(['x_mean', 'x_var', 'theta_mean', 'theta_var',
                             'xdot_mean', 'xdot_var', 'thetadot_mean', 'thetadot_var',
                             'x_mse', 'theta_mse', 'xdot_mse', 'thetadot_mse',
                             'x_mse_25', 'theta_mse_25', 'xdot_mse_25', 'thetadot_mse_25'])
            value_list.extend([x_mean, x_var, theta_mean, theta_var,
                               xdot_mean, xdot_var, thetadot_mean, thetadot_var,
                               x_mse, theta_mse, xdot_mse, thetadot_mse,
                               x_mse_25, theta_mse_25, xdot_mse_25, thetadot_mse_25])

        return dict(zip(key_list, value_list))

    def set_weights(self, weights):
        self.policy_with_value.set_weights(weights)

    def set_ppc_params(self, params):
        self.preprocessor.set_params(params)

    def run_evaluation(self, iteration):
        with self.eval_timer:
            self.iteration = iteration
            if self.args.num_eval_agent == 1:
                n_metrics_list, mean_metric_dict = self.run_n_episodes(self.args.num_eval_episode)
            else:
                n_metrics_list, mean_metric_dict = self.run_n_episodes_parallel(self.args.num_eval_episode)
            with self.writer.as_default():
                for key, val in mean_metric_dict.items():
                    self.tf.summary.scalar("evaluation/{}".format(key), val, step=self.iteration)
                for key, val in self.get_stats().items():
                    self.tf.summary.scalar("evaluation/{}".format(key), val, step=self.iteration)
                self.writer.flush()
            np.save(self.log_dir + '/n_metrics_list_ite{}.npy'.format(iteration), np.array(n_metrics_list))
        if self.eval_times % self.args.eval_log_interval == 0:
            logger.info('Evaluator_info: {}, {}'.format(self.get_stats(), mean_metric_dict))
        self.eval_times += 1


class EvaluatorWithCost(object):
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    def __init__(self, policy_cls, env_id, args):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        self.args = args
        kwargs = copy.deepcopy(vars(self.args))
        if self.args.env_id == 'PathTracking-v0':
            self.env = gym.make(self.args.env_id, num_agent=self.args.num_eval_agent, num_future_data=self.args.num_future_data)
        elif self.args.env_id == 'quadrotor':
            env = make('quadrotor', **self.args.config_eval.quadrotor_config)
            self.env = DummyVecEnv(env)
        else:
            env = gym.make(self.args.env_id)
            self.env = DummyVecEnv(env)
        if isinstance(self.args.random_seed, int):
            self.set_seed(self.args.random_seed)
        self.policy_with_value = policy_cls(**kwargs)
        self.iteration = 0
        if self.args.mode == 'training':
            self.log_dir = self.args.log_dir + '/evaluator'
        else:
            self.log_dir = self.args.test_log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.preprocessor = Preprocessor(**kwargs)

        self.writer = self.tf.summary.create_file_writer(self.log_dir)
        self.stats = {}
        self.eval_timer = TimerStat()
        self.eval_times = 0
        self.eval_start_location = self.args.eval_start_location
        assert self.args.num_eval_episode % len(self.eval_start_location) == 0, \
            print('num_epi:', self.args.num_eval_episode, 'len(starting loc):', len(self.eval_start_location))

    def set_seed(self, seed):
        self.tf.random.set_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)

    def get_stats(self):
        self.stats.update(dict(eval_time=self.eval_timer.mean))
        return self.stats

    def load_weights(self, load_dir, iteration):
        self.policy_with_value.load_weights(load_dir, iteration)

    def load_ppc_params(self, load_dir):
        self.preprocessor.load_params(load_dir)

    def evaluate_saved_model(self, model_load_dir, ppc_params_load_dir, iteration):
        self.load_weights(model_load_dir, iteration)
        self.load_ppc_params(ppc_params_load_dir)

    def run_an_episode(self, steps=None, render=True, epi_idx=0):
        obs_list = []
        action_list = []
        reward_list = []
        info_list = []
        done = 0
        cost_list = []
        qc_list = []
        lam_list = []

        if self.args.env_id == 'quadrotor':
            config = deepcopy(self.args.config_eval)
            config.quadrotor_config['init_state']['init_x'] = self.eval_start_location[epi_idx][0]
            config.quadrotor_config['init_state']['init_z'] = self.eval_start_location[epi_idx][1]
            env = make('quadrotor', **config.quadrotor_config)
            self.env = DummyVecEnv(env)
            self.env.seed(self.args.random_seed + epi_idx)

        obs, info = self.env.reset()
        if render: self.env.render()
        if steps is not None:
            for _ in range(steps):
                processed_obs = self.preprocessor.tf_process_obses(obs)
                action = self.policy_with_value.compute_mode(processed_obs)
                if self.args.demo:
                    qc_val = self.policy_with_value.compute_QC1(processed_obs, action)
                    lam = self.policy_with_value.compute_lam(processed_obs)
                    # print("qc: {}".format(qc_val.numpy()))
                    # print("lam: {}".format(lam.numpy()))
                    qc_list.append(qc_val[0])
                    lam_list.append(lam[0])
                obs_list.append(obs[0])
                action_list.append(action[0])

                obs, reward, done, info = self.env.step(action.numpy())
                cost = np.max(info[0].get('constraint_values', 0))  # todo: scg: constraint_value; gym: cost
                if self.args.demo:
                    qc = np.abs(qc_val[0] - 3)
                    self.env.load_indicator(10 * lam[0] + 0.05 * qc)
                if render: self.env.render()
                reward_list.append(reward[0])
                info_list.append(info[0])
                cost_list.append(cost)
        else:
            while not done:
                processed_obs = self.preprocessor.tf_process_obses(obs)
                action = self.policy_with_value.compute_mode(processed_obs)
                # qc_val = self.policy_with_value.compute_QC1(processed_obs, action)
                # lam = self.policy_with_value.compute_lam(processed_obs)
                # qc_list.append(qc_val[0])
                # lam_list.append(lam[0])
                obs_list.append(obs[0])
                action_list.append(action[0])
                obs, reward, done, info = self.env.step(action.numpy())
                cost = np.max(info[0].get('constraint_values', 0))  # todo: scg: constraint_values; gym: cost
                if render: self.env.render()
                reward_list.append(reward[0])
                info_list.append(info[0])
                cost_list.append(cost)
        episode_return = sum(reward_list)
        episode_cost_sum = sum(cost_list)
        episode_len = len(reward_list)
        info_dict = dict()
        for key in info_list[0].keys():
            info_key = list(map(lambda x: max(x[key]) if key is 'constraint_values' else x[key], info_list))
            mean_key = sum(info_key) / len(info_key)
            info_dict.update({key: mean_key})
        info_dict.update(dict(action_list=np.array(action_list),
                              reward_list=np.array(reward_list),
                              obs_list=np.array(obs_list),
                              episode_cost_sum=episode_cost_sum,
                              episode_return=episode_return,
                              episode_len=episode_len))
        return info_dict

    def run_n_episodes(self, n):
        metrics_list = []
        vectors_list = []
        for i in range(n):
            logger.info('logging {}-th episode'.format(i))
            episode_info = self.run_an_episode(self.args.fixed_steps, self.args.eval_render, i)
            metrics_list.append(self.metrics_for_an_episode(episode_info))
            if self.args.mode == 'testing' and self.args.env_id == 'quadrotor':
                vectors_list.append({'x': episode_info['obs_list'][:, 0], 
                                     'x_dot': episode_info['obs_list'][:, 1], 
                                     'z': episode_info['obs_list'][:, 2],
                                     'z_dot': episode_info['obs_list'][:, 3]})

        if self.args.mode == 'testing' and self.args.env_id == 'quadrotor':
            np.save(self.log_dir + '/coordinates_x_z.npy', np.array((vectors_list)))

        out = {}
        for key in metrics_list[0].keys():
            value_list = list(map(lambda x: x[key], metrics_list))
            out.update({key: sum(value_list)/len(value_list)})

        if self.args.env_id == 'quadrotor':
            return_list = list(map(lambda x: x['episode_return'], metrics_list))
            out.update({'worst-case return': min(return_list)})

            violation_list = list(map(lambda x: x['episode_constraint_violation'], metrics_list))
            out.update({'worst-case violation': max(violation_list)})

        return metrics_list, out

    def run_n_episodes_parallel(self, n):
        logger.info('logging {} episodes in parallel'.format(n))
        metrics_list = []
        obses_list = []
        actions_list = []
        rewards_list = []
        obses, info = self.env.reset()
        if self.args.eval_render: self.env.render()
        for _ in range(self.args.fixed_steps):
            processed_obses = self.preprocessor.tf_process_obses(obses)
            actions = self.policy_with_value.compute_mode(processed_obses)
            obses_list.append(obses)
            actions_list.append(actions)
            obses, rewards, dones, _ = self.env.step(actions.numpy())
            if self.args.eval_render: self.env.render()
            rewards_list.append(rewards)
        for i in range(n):
            obs_list = [obses[i] for obses in obses_list]
            action_list = [actions[i] for actions in actions_list]
            reward_list = [rewards[i] for rewards in rewards_list]
            episode_return = sum(reward_list)
            episode_len = len(reward_list)
            info_dict = dict()
            info_dict.update(dict(obs_list=np.array(obs_list),
                                  action_list=np.array(action_list),
                                  reward_list=np.array(reward_list),
                                  episode_return=episode_return,
                                  episode_len=episode_len))
            metrics_list.append(self.metrics_for_an_episode(info_dict))
        out = {}
        for key in metrics_list[0].keys():
            value_list = list(map(lambda x: x[key], metrics_list))
            out.update({key: sum(value_list) / len(value_list)})
        return metrics_list, out

    def metrics_for_an_episode(self, episode_info):  # user defined, transform episode info dict to metric dict
        key_list = ['episode_return', 'episode_len']
        episode_return = episode_info['episode_return']
        episode_len = episode_info['episode_len']
        value_list = [episode_return, episode_len]
        if self.args.env_id == 'PathTracking-v0':
            delta_v_list = list(map(lambda x: x[0], episode_info['obs_list']))
            delta_y_list = list(map(lambda x: x[3], episode_info['obs_list']))
            delta_phi_list = list(map(lambda x: x[4], episode_info['obs_list']))
            steer_list = list(map(lambda x: x[0]*1.2 * np.pi / 9, episode_info['action_list']))
            acc_list = list(map(lambda x: x[1]*3., episode_info['action_list']))

            rew_list = episode_info['reward_list']
            stationary_rew_mean = sum(rew_list[20:])/len(rew_list[20:])

            delta_y_mse = np.sqrt(np.mean(np.square(np.array(delta_y_list))))
            delta_phi_mse = np.sqrt(np.mean(np.square(np.array(delta_phi_list))))
            delta_v_mse = np.sqrt(np.mean(np.square(np.array(delta_v_list))))
            steer_mse = np.sqrt(np.mean(np.square(np.array(steer_list))))
            acc_mse = np.sqrt(np.mean(np.square(np.array(acc_list))))
            key_list.extend(['delta_y_mse', 'delta_phi_mse', 'delta_v_mse',
                             'stationary_rew_mean', 'steer_mse', 'acc_mse'])
            value_list.extend([delta_y_mse, delta_phi_mse, delta_v_mse,
                               stationary_rew_mean, steer_mse, acc_mse])
        elif self.args.env_id == 'InvertedPendulumConti-v0':
            x_list = list(map(lambda x: x[0], episode_info['obs_list']))
            theta_list = list(map(lambda x: x[1], episode_info['obs_list']))
            xdot_list = list(map(lambda x: x[2], episode_info['obs_list']))
            thetadot_list = list(map(lambda x: x[3], episode_info['obs_list']))
            x_mean, x_var = np.mean(np.array(x_list)), np.var(np.array(x_list))
            theta_mean, theta_var = np.mean(np.array(theta_list)), np.var(np.array(theta_list))
            xdot_mean, xdot_var = np.mean(np.array(xdot_list)), np.var(np.array(xdot_list))
            thetadot_mean, thetadot_var = np.mean(np.array(thetadot_list)), np.var(np.array(thetadot_list))
            x_mse, theta_mse = np.sqrt(np.mean(np.square(np.array(x_list)))),\
                               np.sqrt(np.mean(np.square(np.array(theta_list))))

            xdot_mse, thetadot_mse = np.sqrt(np.mean(np.square(np.array(xdot_list)))),\
                                     np.sqrt(np.mean(np.square(np.array(thetadot_list))))
            x_mse_25, theta_mse_25 = np.sqrt(np.mean(np.square(np.array(x_list)[:25]))), \
                                     np.sqrt(np.mean(np.square(np.array(theta_list)[:25])))
            xdot_mse_25, thetadot_mse_25 = np.sqrt(np.mean(np.square(np.array(xdot_list)[:25]))), \
                                     np.sqrt(np.mean(np.square(np.array(thetadot_list)[:25])))
            key_list.extend(['x_mean', 'x_var', 'theta_mean', 'theta_var',
                             'xdot_mean', 'xdot_var', 'thetadot_mean', 'thetadot_var',
                             'x_mse', 'theta_mse', 'xdot_mse', 'thetadot_mse',
                             'x_mse_25', 'theta_mse_25', 'xdot_mse_25', 'thetadot_mse_25'])
            value_list.extend([x_mean, x_var, theta_mean, theta_var,
                               xdot_mean, xdot_var, thetadot_mean, thetadot_var,
                               x_mse, theta_mse, xdot_mse, thetadot_mse,
                               x_mse_25, theta_mse_25, xdot_mse_25, thetadot_mse_25])

        elif self.args.env_id[:4] == 'Safe':
            episode_cost = episode_info['episode_cost']
            ep_cost_rate = episode_info['ep_cost_rate']
            key_list.extend(['episode_cost', 'ep_cost_rate'])
            value_list.extend([episode_cost, ep_cost_rate])

        elif self.args.env_id == 'quadrotor':
            episode_cost_sum = episode_info['episode_cost_sum']
            episode_mse = episode_info['mse']
            episode_mse_speed = episode_info['mse_speed']
            episode_mse_angle = episode_info['mse_angle']
            episode_mse_angle_speed = episode_info['mse_angle_speed']
            episode_constraint_violation = episode_info['constraint_violation']
            key_list.extend(['episode_cost_sum', 'episode_mse', 'mse_speed', 'mse_angle',
                             'mse_angle_speed', 'episode_constraint_violation'])
            value_list.extend([episode_cost_sum, episode_mse, episode_mse_speed, episode_mse_angle,
                               episode_mse_angle_speed, episode_constraint_violation])

        return dict(zip(key_list, value_list))

    def set_weights(self, weights):
        self.policy_with_value.set_weights(weights)

    def set_ppc_params(self, params):
        self.preprocessor.set_params(params)

    def run_evaluation(self, iteration):
        with self.eval_timer:
            self.iteration = iteration
            if self.args.num_eval_agent == 1:
                n_metrics_list, mean_metric_dict = self.run_n_episodes(self.args.num_eval_episode)
            else:
                n_metrics_list, mean_metric_dict = self.run_n_episodes_parallel(self.args.num_eval_episode)
            with self.writer.as_default():
                for key, val in mean_metric_dict.items():
                    self.tf.summary.scalar("evaluation/{}".format(key), val, step=self.iteration)
                for key, val in self.get_stats().items():
                    self.tf.summary.scalar("evaluation/{}".format(key), val, step=self.iteration)
                self.writer.flush()
            np.save(self.log_dir + '/n_metrics_list_ite{}.npy'.format(iteration), np.array(n_metrics_list))
        if self.eval_times % self.args.eval_log_interval == 0:
            logger.info('Evaluator_info: {}, {}'.format(self.get_stats(), mean_metric_dict))
        self.eval_times += 1
        over_cost_lim = mean_metric_dict['episode_return'] > -6000 # todo
        return over_cost_lim

    def run_evaluation_demo(self, iteration):
        with self.eval_timer:
            self.iteration = iteration
            if self.args.num_eval_agent == 1:
                n_metrics_list, mean_metric_dict = self.run_n_episodes(self.args.num_eval_episode)
            else:
                n_metrics_list, mean_metric_dict = self.run_n_episodes_parallel(self.args.num_eval_episode)
            with self.writer.as_default():
                for key, val in mean_metric_dict.items():
                    self.tf.summary.scalar("evaluation/{}".format(key), val, step=self.iteration)
                for key, val in self.get_stats().items():
                    self.tf.summary.scalar("evaluation/{}".format(key), val, step=self.iteration)
                self.writer.flush()
            np.save(self.log_dir + '/n_metrics_list_ite{}.npy'.format(iteration), np.array(n_metrics_list))
        if self.eval_times % self.args.eval_log_interval == 0:
            logger.info('Evaluator_info: {}, {}'.format(self.get_stats(), mean_metric_dict))
        self.eval_times += 1


def test_trained_model(model_dir, ppc_params_dir, iteration):
    from train_script import built_mixedpg_parser
    from policy import PolicyWithQs
    args = built_mixedpg_parser()
    evaluator = Evaluator(PolicyWithQs, args.env_id, args)
    evaluator.load_weights(model_dir, iteration)
    evaluator.load_ppc_params(ppc_params_dir)
    return evaluator.metrics(1000, render=False, reset=False)

def test_evaluator():
    from train_script import built_SAC_parser
    from policy import PolicyWithQs
    args = built_SAC_parser()
    evaluator = Evaluator(PolicyWithQs, args.env_id, args)
    evaluator.run_evaluation(3)

def read_metrics():
    metrics = np.load('/home/mahaitong/PycharmProjects/mpg/results/FSAC/PointGoal/PointGoal2-2021-04-28-01-16-21/logs/tester/test-2021-05-04-01-26-19/n_metrics_list_ite3000000.npy'
                      , allow_pickle=True)
    print(metrics)
    ep_cost = []
    ep_ret = []
    for metric in metrics:
        if 0 < metric['episode_cost']<300 :
            ep_cost.append(metric['episode_cost'])
            ep_ret.append(metric['episode_return'])
    mean = np.mean(ep_cost)
    std = np.std(ep_cost)
    upquant = np.quantile(ep_cost, 0.75)
    print('cost mean: {}, cost std: {}, quant: {}'.format(mean, std, upquant))
    mean = np.mean(ep_ret)
    std = np.std(ep_ret)
    upquant = np.quantile(ep_ret, 0.75)
    print('return mean: {}, return std: {}, quant: {}'.format(mean, std, upquant))



if __name__ == '__main__':
    read_metrics()
