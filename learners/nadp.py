#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/9/12
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: nadp.py
# =====================================

import logging

import gym
import numpy as np
from envs_and_models import NAME2MODELCLS

from preprocessor import Preprocessor
from utils.misc import TimerStat

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class NADPLearner(object):
    import tensorflow as tf

    def __init__(self, policy_cls, args):
        self.args = args
        self.batch_size = self.args.replay_batch_size
        self.policy_with_value = policy_cls(**vars(self.args))
        self.batch_data = {}
        self.M = self.args.M
        self.num_rollout_list_for_policy_update = self.args.num_rollout_list_for_policy_update
        self.num_rollout_list_for_q_estimation = self.args.num_rollout_list_for_q_estimation

        self.model = NAME2MODELCLS[self.args.env_id](**vars(self.args))
        self.preprocessor = Preprocessor(self.args.obs_dim, self.args.obs_ptype, self.args.rew_ptype,
                                         self.args.obs_scale, self.args.rew_scale, self.args.rew_shift,
                                         gamma=self.args.gamma)
        self.policy_gradient_timer = TimerStat()
        self.q_gradient_timer = TimerStat()
        self.stats = {}
        self.info_for_buffer = {}
        self.counter = 0
        self.num_batch_reuse = self.args.num_batch_reuse

    def get_stats(self):
        return self.stats

    def get_info_for_buffer(self):
        return self.info_for_buffer

    def get_batch_data(self, batch_data, rb, indexes):
        self.batch_data = {'batch_obs': batch_data[0].astype(np.float32),
                           'batch_actions': batch_data[1].astype(np.float32),
                           'batch_rewards': batch_data[2].astype(np.float32),
                           'batch_obs_tp1': batch_data[3].astype(np.float32),
                           'batch_dones': batch_data[4].astype(np.float32),
                           }

        if self.args.buffer_type != 'normal':
            self.info_for_buffer.update(dict(td_error=self.compute_td_error(),
                                             rb=rb,
                                             indexes=indexes))

    def compute_td_error(self):
        processed_obs = self.preprocessor.tf_process_obses(self.batch_data['batch_obs']).numpy()  # n_step*obs_dim
        processed_rewards = self.preprocessor.tf_process_rewards(self.batch_data['batch_rewards']).numpy()
        processed_obs_tp1 = self.preprocessor.tf_process_obses(self.batch_data['batch_obs_tp1']).numpy()

        values_t = self.policy_with_value.compute_Q1(processed_obs, self.batch_data['batch_actions']).numpy()
        target_act_tp1, _ = self.policy_with_value.compute_target_action(processed_obs_tp1)
        target_Q1_of_tp1 = self.policy_with_value.compute_Q1_target(processed_obs_tp1, target_act_tp1).numpy()
        td_error = processed_rewards + self.args.gamma * target_Q1_of_tp1 - values_t
        return td_error

    def get_weights(self):
        return self.policy_with_value.get_weights()

    def set_weights(self, weights):
        return self.policy_with_value.set_weights(weights)

    def set_ppc_params(self, params):
        self.preprocessor.set_params(params)

    def model_rollout_for_q_estimation(self, start_obses, start_actions):
        obses_tile = self.tf.tile(start_obses, [self.M, 1])
        actions_tile = self.tf.tile(start_actions, [self.M, 1])

        processed_obses_tile = self.preprocessor.tf_process_obses(obses_tile)
        processed_obses_tile_list = [processed_obses_tile]
        actions_tile_list = [actions_tile]
        rewards_sum_tile = self.tf.zeros((obses_tile.shape[0],))
        rewards_sum_list = [rewards_sum_tile]
        gammas_list = [self.tf.ones((obses_tile.shape[0],))]

        self.model.reset(obses_tile)
        max_num_rollout = max(self.num_rollout_list_for_q_estimation)
        if max_num_rollout > 0:
            for ri in range(max_num_rollout):
                obses_tile, rewards = self.model.rollout_out(actions_tile)
                processed_obses_tile = self.preprocessor.tf_process_obses(obses_tile)
                processed_rewards = self.preprocessor.tf_process_rewards(rewards)
                rewards_sum_tile += self.tf.pow(self.args.gamma, ri) * processed_rewards
                rewards_sum_list.append(rewards_sum_tile)
                actions_tile, _ = self.policy_with_value.compute_action(processed_obses_tile)
                processed_obses_tile_list.append(processed_obses_tile)
                actions_tile_list.append(actions_tile)
                gammas_list.append(self.tf.pow(self.args.gamma, ri + 1) * self.tf.ones((obses_tile.shape[0],)))

        with self.tf.name_scope('compute_all_model_returns') as scope:
            all_Qs = self.policy_with_value.compute_Q1_target(
                self.tf.concat(processed_obses_tile_list, 0), self.tf.concat(actions_tile_list, 0))
            all_rewards_sums = self.tf.concat(rewards_sum_list, 0)
            all_gammas = self.tf.concat(gammas_list, 0)
            all_targets = all_rewards_sums + all_gammas * all_Qs

            final = self.tf.reshape(all_targets, (max_num_rollout + 1, self.M, -1))
            all_model_returns = self.tf.reduce_mean(final, axis=1)
        selected_model_returns = []
        for num_rollout in self.num_rollout_list_for_q_estimation:
            selected_model_returns.append(all_model_returns[num_rollout])

        selected_model_returns_flatten = self.tf.concat(selected_model_returns, 0)
        return self.tf.stop_gradient(selected_model_returns_flatten)

    def model_rollout_for_policy_update(self, start_obses):
        max_num_rollout = max(self.num_rollout_list_for_policy_update)

        obses_tile = self.tf.tile(start_obses, [self.M, 1])
        processed_obses_tile = self.preprocessor.tf_process_obses(obses_tile)
        actions_tile, _ = self.policy_with_value.compute_action(processed_obses_tile)

        processed_obses_tile_list = [processed_obses_tile]
        actions_tile_list = [actions_tile]
        rewards_sum_tile = self.tf.zeros((obses_tile.shape[0],))
        rewards_sum_list = [rewards_sum_tile]
        gammas_list = [self.tf.ones((obses_tile.shape[0],))]

        self.model.reset(obses_tile)
        if max_num_rollout > 0:
            for ri in range(max_num_rollout):
                obses_tile, rewards = self.model.rollout_out(actions_tile)
                processed_obses_tile = self.preprocessor.tf_process_obses(obses_tile)
                processed_rewards = self.preprocessor.tf_process_rewards(rewards)
                rewards_sum_tile += self.tf.pow(self.args.gamma, ri) * processed_rewards
                rewards_sum_list.append(rewards_sum_tile)
                actions_tile, _ = self.policy_with_value.compute_action(processed_obses_tile)
                processed_obses_tile_list.append(processed_obses_tile)
                actions_tile_list.append(actions_tile)
                gammas_list.append(self.tf.pow(self.args.gamma, ri + 1) * self.tf.ones((obses_tile.shape[0],)))

        with self.tf.name_scope('compute_all_model_returns') as scope:
            all_Qs = self.policy_with_value.compute_Q1(
                self.tf.concat(processed_obses_tile_list, 0), self.tf.concat(actions_tile_list, 0))
            all_rewards_sums = self.tf.concat(rewards_sum_list, 0)
            all_gammas = self.tf.concat(gammas_list, 0)

            final = self.tf.reshape(all_rewards_sums + all_gammas * all_Qs, (max_num_rollout + 1, self.M, -1))
            # final [[[time0+traj0], [time0+traj1], ..., [time0+trajn]],
            #        [[time1+traj0], [time1+traj1], ..., [time1+trajn]],
            #        ...
            #        [[timen+traj0], [timen+traj1], ..., [timen+trajn]],
            #        ]
            all_model_returns = self.tf.reduce_mean(final, axis=1)

        reduced_model_returns = self.tf.reduce_mean(all_model_returns, axis=1)
        value_mean = reduced_model_returns[0]
        policy_loss = -reduced_model_returns[self.num_rollout_list_for_policy_update[0]]
        return policy_loss, value_mean

    @tf.function
    def q_forward_and_backward(self, mb_obs, mb_actions):
        processed_mb_obs = self.preprocessor.tf_process_obses(mb_obs)
        model_targets = self.model_rollout_for_q_estimation(mb_obs, mb_actions)
        with self.tf.GradientTape() as tape:
            with self.tf.name_scope('q_loss') as scope:
                q_pred = self.policy_with_value.compute_Q1(processed_mb_obs, mb_actions)
                q_loss = 0.5 * self.tf.reduce_mean(self.tf.square(q_pred - model_targets))
        with self.tf.name_scope('q_gradient') as scope:
            q_gradient = tape.gradient(q_loss, self.policy_with_value.Q1.trainable_weights)

        return q_loss, q_gradient

    @tf.function
    def policy_forward_and_backward(self, mb_obs):
        with self.tf.GradientTape(persistent=True) as tape:
            policy_loss, value_mean = self.model_rollout_for_policy_update(mb_obs)

        with self.tf.name_scope('policy_jacobian') as scope:
            policy_gradient = tape.gradient(policy_loss,
                                            self.policy_with_value.policy.trainable_weights)
            return policy_loss, policy_gradient, value_mean

    def export_graph(self, writer):
        mb_obs = self.batch_data['batch_obs']
        mb_actions = self.batch_data['batch_actions']
        self.tf.summary.trace_on(graph=True, profiler=False)
        self.q_forward_and_backward(mb_obs, mb_actions)
        with writer.as_default():
            self.tf.summary.trace_export(name="q_forward_and_backward", step=0)

        self.tf.summary.trace_on(graph=True, profiler=False)
        self.policy_forward_and_backward(mb_obs)
        with writer.as_default():
            self.tf.summary.trace_export(name="policy_forward_and_backward", step=0)

    def compute_gradient(self, batch_data, rb, indexes, iteration):  # compute gradient
        if self.counter % self.num_batch_reuse == 0:
            self.get_batch_data(batch_data, rb, indexes)
        self.counter += 1
        if self.args.buffer_type != 'normal':
            self.info_for_buffer.update(dict(td_error=self.compute_td_error()))
        mb_obs = self.batch_data['batch_obs']
        mb_actions = self.batch_data['batch_actions']

        with self.q_gradient_timer:
            q_loss, q_gradient = self.q_forward_and_backward(mb_obs, mb_actions)
            q_gradient, q_gradient_norm = self.tf.clip_by_global_norm(q_gradient, self.args.gradient_clip_norm)

        with self.policy_gradient_timer:
            policy_loss, policy_gradient, value_mean = self.policy_forward_and_backward(mb_obs)
            final_policy_gradient, policy_gradient_norm = self.tf.clip_by_global_norm(policy_gradient,
                                                                                  self.args.gradient_clip_norm)

        self.stats.update(dict(
            iteration=iteration,
            q_timer=self.q_gradient_timer.mean,
            pg_time=self.policy_gradient_timer.mean,
            q_loss=q_loss.numpy(),
            policy_loss=policy_loss.numpy(),
            value_mean=value_mean.numpy(),
            q_gradient_norm=q_gradient_norm.numpy(),
            policy_gradient_norm=policy_gradient_norm.numpy(),
            num_rollout_list_for_policy=self.num_rollout_list_for_policy_update,
            num_rollout_list_for_q=self.num_rollout_list_for_q_estimation,
        ))

        gradient_tensor = q_gradient + final_policy_gradient  # q_gradient + final_policy_gradient
        return list(map(lambda x: x.numpy(), gradient_tensor))


if __name__ == '__main__':
    pass
