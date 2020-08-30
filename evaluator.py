#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/8/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: evaluator.py
# =====================================

import logging
import os

import gym
import numpy as np

from preprocessor import Preprocessor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# logger.setLevel(logging.INFO)


class Evaluator(object):
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], 'GPU')

    def __init__(self, policy_cls, env_id, args):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        self.args = args
        self.env = gym.make(env_id, num_agent=1, num_future_data=self.args.num_future_data)
        self.policy_with_value = policy_cls(self.env.observation_space, self.env.action_space, self.args)
        self.iteration = 0
        self.log_dir = self.args.log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.preprocessor = Preprocessor(self.env.observation_space, self.args.obs_preprocess_type, self.args.reward_preprocess_type,
                                         self.args.obs_scale_factor, self.args.reward_scale_factor,
                                         gamma=self.args.gamma, num_agent=1)

        self.writer = self.tf.summary.create_file_writer(self.log_dir + '/evaluator')
        self.stats = {}

    def load_weights(self, load_dir, iteration):
        self.policy_with_value.load_weights(load_dir, iteration)

    def load_ppc_params(self, load_dir):
        self.preprocessor.load_params(load_dir)

    def evaluate_saved_model(self, model_load_dir, ppc_params_load_dir, iteration):
        self.load_weights(model_load_dir, iteration)
        self.load_ppc_params(ppc_params_load_dir)

    def run_an_episode(self):
        obs_list = []
        action_list = []
        reward_list = []
        done = 0
        obs = self.env.reset()
        while not done:
            processed_obs = self.preprocessor.tf_process_obses(obs)
            action, neglogp = self.policy_with_value.compute_action(processed_obs)
            obs_list.append(obs[0])
            action_list.append(action[0])
            obs, reward, done, info = self.env.step(action.numpy())
            reward_list.append(reward[0])
        episode_return = sum(reward_list)
        episode_len = len(reward_list)

        return dict(obs_list=np.array(obs_list),
                    action_list=np.array(action_list),
                    reward_list=np.array(reward_list),
                    episode_return=episode_return,
                    episode_len=episode_len)

    def run_n_episodes(self, n):
        metrics_list = []
        for _ in range(n):
            episode_info = self.run_an_episode()
            metrics_list.append(self.metrics_for_an_episode(episode_info))
        out = {}
        for key in metrics_list[0].keys():
            value_list = list(map(lambda x: x[key], metrics_list))
            out.update({key: sum(value_list)/len(value_list)})
        return out


    def metrics_for_an_episode(self, episode_info):
        key_list = ['episode_return', 'episode_len', 'delta_y_mse', 'delta_phi_mse', 'delta_v_mse']
        episode_return = episode_info['episode_return']
        episode_len = episode_info['episode_len']
        delta_v_list = list(map(lambda x: x[0]-20, episode_info['obs_list']))
        delta_y_list = list(map(lambda x: x[3], episode_info['obs_list']))
        delta_phi_list = list(map(lambda x: x[4], episode_info['obs_list']))

        delta_y_mse = np.sqrt(np.mean(np.square(np.array(delta_y_list))))
        delta_phi_mse = np.sqrt(np.mean(np.square(np.array(delta_phi_list))))
        delta_v_mse = np.sqrt(np.mean(np.square(np.array(delta_v_list))))
        value_list = [episode_return, episode_len, delta_y_mse, delta_phi_mse, delta_v_mse]
        return dict(zip(key_list, value_list))

    def set_weights(self, weights):
        self.policy_with_value.set_weights(weights)

    def set_ppc_params(self, params):
        self.preprocessor.set_params(params)

    def run_evaluation(self, iteration):
        self.iteration = iteration
        metrics = self.run_n_episodes(self.args.num_eval_episode)
        logger.info(metrics)
        with self.writer.as_default():
            for key, value in metrics.items():
                self.tf.summary.scalar("evaluation/{}".format(key), value, step=self.iteration)
            self.writer.flush()


def test_trained_model(model_dir, ppc_params_dir, iteration):
    from train_script import built_mixedpg_parser
    from policy import PolicyWithQs
    args = built_mixedpg_parser()
    evaluator = Evaluator(PolicyWithQs, args.env_id, args)
    evaluator.load_weights(model_dir, iteration)
    evaluator.load_ppc_params(ppc_params_dir)
    return evaluator.metrics(1000, render=False, reset=False)


if __name__ == '__main__':
    model_dir = './results/mixed_pg/experiment-2020-04-22-14-01-37/models'
    # model_dir = './results/mixed_pg/experiment-2020-04-22-15-02-12/models'
    print(test_trained_model(model_dir, model_dir, 20))
