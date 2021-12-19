#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2021/12/13
# @Author  : Haitong Ma, Dongjie Yu (Tsinghua Univ.)
# @FileName: naive_reach.py
# =====================================

import logging

import gym
import numpy as np
# from gym.envs.user_defined.EmerBrake.models import EmBrakeModel
from dynamics.models import *

from preprocessor import Preprocessor
from utils.misc import TimerStat, args2envkwargs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CONSTRAINTS_CLIP_MINUS = -1.0  # TODO: why -1


class CstrReachLearner(object):
    import tensorflow as tf
    tf.config.optimizer.set_experimental_options({'constant_folding': True,
                                                  'arithmetic_optimization': True,
                                                  'dependency_optimization': True,
                                                  'loop_optimization': True,
                                                  'function_optimization': True,
                                                  })
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    def __init__(self, policy_cls, args):
        self.args = args
        self.policy_with_value = policy_cls(self.args)
        self.batch_data = {}
        self.all_data = {}
        self.M = self.args.M
        self.num_rollout_list_for_policy_update = self.args.num_rollout_list_for_policy_update
        self.fea_gamma = self.args.fea_gamma
        self.gamma = self.args.gamma

        brake_model = EmBrakeModel()
        double_intergrator_model = UpperTriangleModel()
        air3d_model = Air3dModel()
        model_dict = {"UpperTriangle": double_intergrator_model,
                      "Air3d": air3d_model}
        self.model = model_dict.get(args.env_id.split("-")[0])
        self.preprocessor = Preprocessor((self.args.obs_dim,), self.args.obs_preprocess_type,
                                         self.args.reward_preprocess_type,
                                         self.args.obs_scale, self.args.reward_scale, self.args.reward_shift,
                                         gamma=self.args.gamma)
        self.grad_timer = TimerStat()
        self.fea_grad_timer = TimerStat()
        self.stats = {}
        self.info_for_buffer = {}
        # self.constraint_total_dim = args.num_rollout_list_for_policy_update[0] * self.model.constraints_num

    def get_stats(self):
        return self.stats

    def get_info_for_buffer(self):
        return self.info_for_buffer

    def get_batch_data(self, batch_data, rb, indexes):
        self.batch_data = {'batch_obs': batch_data[0].astype(np.float32),
                           'batch_actions': batch_data[1].astype(np.float32),
                           'batch_rewards': batch_data[2].astype(np.float32),
                           'batch_obs_tp1': batch_data[3].astype(np.float32),
                           'batch_dones': batch_data[4].astype(np.float32)
                           }

    def get_weights(self):
        return self.policy_with_value.get_weights()

    def set_weights(self, weights):
        return self.policy_with_value.set_weights(weights)

    def set_ppc_params(self, params):
        self.preprocessor.set_params(params)

    # TODO: feasibility v learning
    @tf.function
    def feasibility_forward_and_backward(self, mb_obs, mb_done):
        with self.tf.GradientTape(persistent=True) as tape1:
            tape1.watch(mb_obs)
            constraints = self.model.compute_constraints(mb_obs)
            fea_v_obses = self.policy_with_value.compute_fea_v(mb_obs)

        d_fea_v_d_obses = tape1.gradient(fea_v_obses, mb_obs)  # shape: (B, 2)

        with self.tf.GradientTape(persistent=True) as tape2:
            fea_v_obses = self.policy_with_value.compute_fea_v(mb_obs)
            signed_obj = self.tf.matmul(d_fea_v_d_obses, self.model.g_x(mb_obs))  # shape: (B, 1)
            u_safe = self.tf.where(signed_obj >= 0., -self.model.action_range,
                                   self.model.action_range)  # action_range: tf.constant, necessary?
            obses_tp1 = self.model.f_xu(mb_obs, u_safe)
            fea_v_obses_tp1_minimum = self.policy_with_value.compute_fea_v(obses_tp1)
            assert fea_v_obses.shape == fea_v_obses_tp1_minimum.shape == constraints.shape, print(fea_v_obses.shape)

            fea_v_terminal = constraints
            fea_v_non_terminal = (1 - self.fea_gamma) * constraints + \
                                 self.fea_gamma * self.tf.maximum(constraints, fea_v_obses_tp1_minimum)
            fea_v_target = self.tf.stop_gradient(self.tf.where(mb_done, fea_v_terminal, fea_v_non_terminal))

            fea_loss = 0.5 * self.tf.reduce_mean(self.tf.square(fea_v_target - fea_v_obses))

        with self.tf.name_scope('fea_gradient') as scope:
            fea_v_gradient = tape2.gradient(fea_loss, self.policy_with_value.fea_v.trainable_weights)

            return fea_v_gradient, fea_loss

    def model_rollout_for_update(self, start_obses):
        start_obses = self.tf.tile(start_obses, [self.M, 1])
        self.model.reset(start_obses)
        rewards_sum = self.tf.zeros((start_obses.shape[0],))
        obses = start_obses
        # for step in range(self.num_rollout_list_for_policy_update[0]):
        processed_obses = self.preprocessor.tf_process_obses(obses)
        actions, _ = self.policy_with_value.compute_action(processed_obses)
        obses, rewards, constraints, done = self.model.rollout_out(actions)
        rewards_sum += self.preprocessor.tf_process_rewards(rewards)
        processed_obses_t = self.preprocessor.tf_process_obses(start_obses)
        processed_obses_tp1 = self.preprocessor.tf_process_obses(obses)

        # value part
        v_t = self.policy_with_value.compute_obj_v(processed_obses_t)
        v_tp1 = self.policy_with_value.compute_obj_v(processed_obses_tp1)
        v_target = self.tf.stop_gradient(self.gamma * v_tp1 + rewards_sum)
        # assert v_target.shape == v_tp1.shape
        obj_loss = 0.5 * self.tf.reduce_mean(self.tf.square(v_target - v_t))

        # policy part
        mu = self.tf.squeeze(self.policy_with_value.compute_mu(processed_obses_t), axis=1)
        fea_v_tp1 = self.policy_with_value.compute_fea_v(processed_obses_tp1)
        # assert mu.shape == fea_v_tp1.shape, print(mu.shape, fea_v_tp1.shape)
        punish_terms = self.tf.reduce_mean(self.tf.multiply(self.tf.stop_gradient(mu), fea_v_tp1))
        pg_loss = -self.tf.reduce_mean(v_tp1 + rewards_sum) + punish_terms

        # mu part
        complementary_slackness = self.tf.reduce_mean(
            self.tf.multiply(mu, self.tf.stop_gradient(fea_v_tp1)))
        mu_loss = - complementary_slackness

        return obj_loss, pg_loss, mu_loss, fea_v_tp1, mu, punish_terms, constraints, done

    @tf.function
    def forward_and_backward(self, mb_obs):
        with self.tf.GradientTape(persistent=True) as tape:
            obj_loss, pg_loss, mu_loss, fea_v_tp1, mu, punish_terms, constraints, mb_done = self.model_rollout_for_update(mb_obs)

        with self.tf.name_scope('policy_gradient') as scope:
            obj_v_grad = tape.gradient(obj_loss, self.policy_with_value.obj_v.trainable_weights)
            pg_grad = tape.gradient(pg_loss, self.policy_with_value.policy.trainable_weights)
            mu_grad = tape.gradient(mu_loss, self.policy_with_value.mu.trainable_weights)

        return obj_v_grad, pg_grad, mu_grad, obj_loss, pg_loss, mu_loss, \
               fea_v_tp1, mu, punish_terms, constraints, mb_done

    def compute_gradient(self, samples, rb, indexs, iteration):
        self.get_batch_data(samples, rb, indexs)
        mb_obs = self.tf.constant(self.batch_data['batch_obs'])
        iteration = self.tf.convert_to_tensor(iteration, self.tf.int32)

        with self.grad_timer:
            obj_v_grad, pg_grad, mu_grad, obj_loss, pg_loss, mu_loss, \
                fea_v_tp1, mu, punish_terms, constraints, mb_done = self.forward_and_backward(mb_obs)

            obj_v_grad, obj_v_grad_norm = self.tf.clip_by_global_norm(obj_v_grad, self.args.gradient_clip_norm)
            pg_grad, pg_grad_norm = self.tf.clip_by_global_norm(pg_grad, self.args.gradient_clip_norm)
            mu_grad, mu_grad_norm = self.tf.clip_by_global_norm(mu_grad, self.args.gradient_clip_norm)
        
        with self.fea_grad_timer:
            fea_v_grad, fea_loss = self.feasibility_forward_and_backward(mb_obs, mb_done)
            fea_v_grad, fea_v_grad_norm = self.tf.clip_by_global_norm(fea_v_grad, self.args.gradient_clip_norm)

        self.stats.update(dict(
            iteration=iteration,
            grad_time=self.grad_timer.mean,
            fea_grad_time=self.fea_grad_timer.mean,
            obj_loss=obj_loss.numpy(),
            fea_loss=fea_loss.numpy(),
            punish_terms=punish_terms.numpy(),
            mu_loss=mu_loss.numpy(),
            pg_loss=pg_loss.numpy(),
            obj_v_grad_norm=obj_v_grad_norm.numpy(),
            fea_v_grad_norm=fea_v_grad_norm.numpy(),
            pg_grad_norm=pg_grad_norm.numpy(),
            mu_grad_norm=mu_grad_norm.numpy(),
            mu_mean=np.mean(mu.numpy()),
            mu_max=np.max(mu.numpy()),
            mu_min=np.min(mu.numpy()),
            fea_v_mean=np.mean(fea_v_tp1.numpy()),
            fea_v_max=np.max(fea_v_tp1.numpy()),
            fea_v_min=np.min(fea_v_tp1.numpy()),
            constraints_mean=np.mean(constraints.numpy())
        ))

        grads = obj_v_grad + fea_v_grad + pg_grad + mu_grad

        return list(map(lambda x: x.numpy(), grads))


if __name__ == '__main__':
    pass
