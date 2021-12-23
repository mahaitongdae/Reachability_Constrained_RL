#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/6/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: worker.py
# =====================================

import logging

import gym
import numpy as np

import dynamics
import safe_control_gym
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make

from preprocessor import Preprocessor
from utils.dummy_vec_env import DummyVecEnv
from utils.misc import judge_is_nan

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# logger.setLevel(logging.INFO)


class OffPolicyWorker(object):
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    """just for sample"""

    def __init__(self, policy_cls, env_id, args, worker_id):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        self.worker_id = worker_id
        self.args = args
        self.num_agent = self.args.num_agent
        if self.args.env_id == 'PathTracking-v0':
            self.env = gym.make(self.args.env_id, num_agent=self.num_agent, num_future_data=self.args.num_future_data)
        else:
            env = gym.make(self.args.env_id)
            self.env = DummyVecEnv(env)
        if isinstance(self.args.random_seed, int):
            self.set_seed(self.args.random_seed)
        self.policy_with_value = policy_cls(**vars(self.args))
        self.batch_size = self.args.batch_size
        self.obs = self.env.reset()
        self.done = False
        self.preprocessor = Preprocessor(**vars(self.args))

        self.explore_sigma = self.args.explore_sigma
        self.iteration = 0
        self.num_sample = 0
        self.sample_times = 0
        self.stats = {}
        logger.info('Worker initialized')

    def set_seed(self, seed):
        self.tf.random.set_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)

    def get_stats(self):
        self.stats.update(dict(worker_id=self.worker_id,
                               num_sample=self.num_sample,
                               # ppc_params=self.get_ppc_params()
                               )
                          )
        return self.stats

    def save_weights(self, save_dir, iteration):
        self.policy_with_value.save_weights(save_dir, iteration)

    def load_weights(self, load_dir, iteration):
        self.policy_with_value.load_weights(load_dir, iteration)

    def get_weights(self):
        return self.policy_with_value.get_weights()

    def set_weights(self, weights):
        return self.policy_with_value.set_weights(weights)

    def apply_gradients(self, iteration, grads):
        self.iteration = iteration
        self.policy_with_value.apply_gradients(self.tf.constant(iteration, dtype=self.tf.int32), grads)

    def get_ppc_params(self):
        return self.preprocessor.get_params()

    def set_ppc_params(self, params):
        self.preprocessor.set_params(params)

    def save_ppc_params(self, save_dir):
        self.preprocessor.save_params(save_dir)

    def load_ppc_params(self, load_dir):
        self.preprocessor.load_params(load_dir)

    def sample(self):
        batch_data = []
        for _ in range(int(self.batch_size/self.num_agent)):
            processed_obs = self.preprocessor.process_obs(self.obs)
            judge_is_nan([processed_obs])
            action, logp = self.policy_with_value.compute_action(self.tf.constant(processed_obs))
            if self.explore_sigma is not None:
                action += np.random.normal(0, self.explore_sigma, np.shape(action))
            try:
                judge_is_nan([action])
            except ValueError:
                print('processed_obs', processed_obs)
                print('preprocessor_params', self.preprocessor.get_params())
                print('policy_weights', self.policy_with_value.policy.trainable_weights)
                action, logp = self.policy_with_value.compute_action(processed_obs)
                judge_is_nan([action])
                raise ValueError
            obs_tp1, reward, self.done, info = self.env.step(action.numpy())
            processed_rew = self.preprocessor.process_rew(reward, self.done)
            for i in range(self.num_agent):
                batch_data.append((self.obs[i].copy(), action[i].numpy(), reward[i], obs_tp1[i].copy(), self.done[i]))
            self.obs = self.env.reset()

        if self.worker_id == 1 and self.sample_times % self.args.worker_log_interval == 0:
            logger.info('Worker_info: {}'.format(self.get_stats()))

        self.num_sample += len(batch_data)
        self.sample_times += 1
        return batch_data

    def sample_with_count(self):
        batch_data = self.sample()
        return batch_data, len(batch_data)


class OffPolicyWorkerWithCost(object):
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    """just for sample"""

    def __init__(self, policy_cls, env_id, args, worker_id):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        self.worker_id = worker_id
        self.args = args
        self.num_agent = self.args.num_agent
        if self.args.env_id == 'PathTracking-v0':
            self.env = gym.make(self.args.env_id, num_agent=self.num_agent, num_future_data=self.args.num_future_data)
        elif self.args.env_id == 'quadrotor':
            env = make('quadrotor', **self.args.config.quadrotor_config)
            self.env = DummyVecEnv(env)
        else:
            env = gym.make(self.args.env_id)
            self.env = DummyVecEnv(env)
        if isinstance(self.args.random_seed, int):
            self.set_seed(self.args.random_seed)
        self.policy_with_value = policy_cls(**vars(self.args))
        self.batch_size = self.args.batch_size
        self.obs, self.info = self.env.reset()
        self.done = False
        self.preprocessor = Preprocessor(**vars(self.args))

        self.explore_sigma = self.args.explore_sigma
        self.iteration = 0
        self.num_sample = 0
        self.sample_times = 0
        self.num_costs = 0
        self.stats = {}
        logger.info('Worker initialized')

    def set_seed(self, seed):
        self.tf.random.set_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)

    def get_stats(self):
        cost_rate = self.num_costs / self.num_sample if self.num_sample!=0 else 0
        self.stats.update(dict(worker_id=self.worker_id,
                               num_sample=self.num_sample,
                               # ppc_params=self.get_ppc_params()
                               num_costs=self.num_costs,
                               cost_rate=cost_rate
                               )
                          )
        return self.stats

    def save_weights(self, save_dir, iteration):
        self.policy_with_value.save_weights(save_dir, iteration)

    def load_weights(self, load_dir, iteration):
        self.policy_with_value.load_weights(load_dir, iteration)

    def get_weights(self):
        return self.policy_with_value.get_weights()

    def set_weights(self, weights):
        return self.policy_with_value.set_weights(weights)

    def apply_gradients(self, iteration, grads):
        self.iteration = iteration
        qc_grad, lam_grad = self.policy_with_value.apply_gradients(self.tf.constant(iteration, dtype=self.tf.int32), grads)
        return qc_grad, lam_grad

    def apply_ascent_gradients(self, iteration, qc_grad, lam_grad):
        self.iteration = iteration
        self.policy_with_value.apply_ascent_gradients(self.tf.constant(iteration, dtype=self.tf.int32), qc_grad, lam_grad)

    def get_ppc_params(self):
        return self.preprocessor.get_params()

    def set_ppc_params(self, params):
        self.preprocessor.set_params(params)

    def save_ppc_params(self, save_dir):
        self.preprocessor.save_params(save_dir)

    def load_ppc_params(self, load_dir):
        self.preprocessor.load_params(load_dir)

    def sample(self):
        batch_data = []
        self.sampled_costs = 0
        for _ in range(int(self.batch_size/self.num_agent)):
            processed_obs = self.preprocessor.process_obs(self.obs)
            judge_is_nan([processed_obs])
            action, logp = self.policy_with_value.compute_action(self.tf.constant(processed_obs))
            if self.explore_sigma is not None:
                action += np.random.normal(0, self.explore_sigma, np.shape(action))
            try:
                judge_is_nan([action])
            except ValueError:
                print('processed_obs', processed_obs)
                print('preprocessor_params', self.preprocessor.get_params())
                print('policy_weights', self.policy_with_value.policy.trainable_weights)
                action, logp = self.policy_with_value.compute_action(processed_obs)
                judge_is_nan([action])
                raise ValueError
            obs_tp1, reward, self.done, info = self.env.step(action.numpy())
            cost = np.max(info[0].get('constraint_values', 0))  # todo: scg: constraint_values; gym: cost
            self.sampled_costs += cost
            processed_rew = self.preprocessor.process_rew(reward, self.done)
            for i in range(self.num_agent):
                batch_data.append((self.obs[i].copy(), action[i].numpy(), reward[i], obs_tp1[i].copy(), self.done[i], cost))
            self.obs, self.info = self.env.reset()

        if self.worker_id == 1 and self.sample_times % self.args.worker_log_interval == 0:
            logger.info('Worker_info: {}'.format(self.get_stats()))

        self.num_sample += len(batch_data)
        self.sample_times += 1
        self.num_costs += int(self.sampled_costs)
        return batch_data, int(self.sampled_costs)

    def random_sample(self, policy='default'):
        batch_data = []
        self.sampled_costs = 0
        for _ in range(int(self.batch_size/self.num_agent)):
            processed_obs = self.preprocessor.process_obs(self.obs)
            judge_is_nan([processed_obs])
            # action, logp = self.policy_with_value.compute_action(self.tf.constant(processed_obs))
            action = self.env.action_space.sample()
            if self.explore_sigma is not None:
                action += np.random.normal(0, self.explore_sigma, np.shape(action))
            try:
                judge_is_nan([action])
            except ValueError:
                print('processed_obs', processed_obs)
                print('preprocessor_params', self.preprocessor.get_params())
                print('policy_weights', self.policy_with_value.policy.trainable_weights)
                action, logp = self.policy_with_value.compute_action(processed_obs)
                judge_is_nan([action])
                raise ValueError
            obs_tp1, reward, self.done, info = self.env.step(action)
            cost = np.max(info[0].get('constraint_values', 0))  # todo: scg: constraint_values; gym: cost
            self.sampled_costs += cost
            processed_rew = self.preprocessor.process_rew(reward, self.done)
            for i in range(self.num_agent):
                batch_data.append((self.obs[i].copy(), action[i], reward[i], obs_tp1[i].copy(), self.done[i], cost))
            self.obs = self.env.reset()

        if self.worker_id == 1 and self.sample_times % self.args.worker_log_interval == 0:
            logger.info('Worker_info: {}'.format(self.get_stats()))

        self.num_sample += len(batch_data)
        self.sample_times += 1
        self.num_costs += int(self.sampled_costs)
        return batch_data, int(self.sampled_costs)

    def sample_with_count(self):
        batch_data, costs_count = self.sample()
        return batch_data, len(batch_data), costs_count

    def random_sample_with_count(self):
        batch_data, costs_count = self.sample()
        return batch_data, len(batch_data), costs_count
