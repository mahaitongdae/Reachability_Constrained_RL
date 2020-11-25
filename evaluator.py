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
from utils.misc import TimerStat

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Evaluator(object):
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], 'GPU')

    def __init__(self, policy_cls, env_id, args):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        self.args = args
        self.env = gym.make(env_id, num_agent=self.args.num_eval_agent, num_future_data=self.args.num_future_data)
        self.policy_with_value = policy_cls(self.env.observation_space, self.env.action_space, self.args)
        self.iteration = 0
        if self.args.mode == 'training':
            self.log_dir = self.args.log_dir + '/evaluator'
        else:
            self.log_dir = self.args.test_log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.preprocessor = Preprocessor(self.env.observation_space, self.args.obs_preprocess_type, self.args.reward_preprocess_type,
                                         self.args.obs_scale, self.args.reward_scale, self.args.reward_shift,
                                         gamma=self.args.gamma, num_agent=self.args.num_eval_agent)

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
                info_list.append(info)
        else:
            while not done:
                processed_obs = self.preprocessor.tf_process_obses(obs)
                action = self.policy_with_value.compute_mode(processed_obs)
                obs_list.append(obs[0])
                action_list.append(action[0])
                obs, reward, done, info = self.env.step(action.numpy())
                if render: self.env.render()
                reward_list.append(reward[0])
                info_list.append(info)
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
            obses, rewards, dones, infos = self.env.step(actions.numpy())
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
        key_list = ['episode_return', 'episode_len', 'delta_y_mse', 'delta_phi_mse', 'delta_v_mse',
                    'stationary_rew_mean', 'steer_mse', 'acc_mse']
        episode_return = episode_info['episode_return']
        episode_len = episode_info['episode_len']
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

        value_list = [episode_return, episode_len, delta_y_mse, delta_phi_mse, delta_v_mse,
                      stationary_rew_mean, steer_mse, acc_mse]
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

if __name__ == '__main__':
    test_evaluator()
