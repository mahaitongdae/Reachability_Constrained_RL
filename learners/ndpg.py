#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/9/12
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: ndpg.py
# =====================================

import logging

import gym
import numpy as np

from preprocessor import Preprocessor
from utils.misc import TimerStat
from utils.dummy_vec_env import DummyVecEnv

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class NDPGLearner(object):
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    def __init__(self, policy_cls, args):
        self.args = args
        self.sample_num_in_learner = self.args.sample_num_in_learner
        self.batch_size = self.args.replay_batch_size
        if self.args.env_id == 'PathTracking-v0':
            self.env = gym.make(self.args.env_id, num_agent=self.batch_size, num_future_data=self.args.num_future_data)
        else:
            env = gym.make(self.args.env_id)
            self.env = DummyVecEnv(env)
        self.policy_with_value = policy_cls(**vars(self.args))
        self.batch_data = {}
        self.counter = 0
        self.num_batch_reuse = self.args.num_batch_reuse
        self.preprocessor = Preprocessor(self.args.obs_dim, self.args.obs_ptype, self.args.rew_ptype,
                                         self.args.obs_scale, self.args.rew_scale, self.args.rew_shift,
                                         gamma=self.args.gamma)
        self.policy_gradient_timer = TimerStat()
        self.q_gradient_timer = TimerStat()
        self.target_timer = TimerStat()
        self.stats = {}
        self.info_for_buffer = {}

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

        with self.target_timer:
            target = self.compute_n_step_target()

        self.batch_data.update(dict(batch_targets=target,))
        if self.args.buffer_type != 'normal':
            self.info_for_buffer.update(dict(td_error=self.compute_td_error(),
                                             rb=rb,
                                             indexes=indexes))

        # print(self.batch_data['batch_obs'].shape)  # batch_size * obs_dim
        # print(self.batch_data['batch_actions'].shape)  # batch_size * act_dim

    def sample(self, start_obs, start_action):
        if self.env.num_agent == 1:
            all_obs = []
            all_actions = []
            all_rewards = []
            all_obs_tp1 = []
            for i in range(len(start_obs)):
                obs = start_obs[i][np.newaxis, :]
                self.env.reset(init_obs=obs)
                for t in range(self.sample_num_in_learner):
                    processed_obs = self.preprocessor.tf_process_obses(obs).numpy()
                    action, _ = self.policy_with_value.compute_action(processed_obs)
                    action = self.tf.constant(start_action[i][np.newaxis, :]) if t == 0 else action
                    obs_tp1, reward, _, _ = self.env.step(action.numpy())
                    all_obs.append(obs.copy()[0])
                    all_actions.append(action.numpy()[0])
                    all_rewards.append(reward[0])
                    all_obs_tp1.append(obs_tp1.copy()[0])
                    obs = obs_tp1.copy()
            return {'all_rewards': np.swapaxes(np.array(all_rewards, dtype=np.float32).reshape((self.batch_size, self.sample_num_in_learner)), 0, 1),
                    'all_obs_tp1': np.swapaxes(np.array(all_obs_tp1, dtype=np.float32).reshape((self.batch_size, self.sample_num_in_learner, -1)), 0, 1),
                    }
        else:
            assert self.env.num_agent == self.batch_size
            batch_data = []
            obs = start_obs
            self.env.reset(init_obs=obs)
            for t in range(self.sample_num_in_learner):
                processed_obs = self.preprocessor.tf_process_obses(obs).numpy()
                action, _ = self.policy_with_value.compute_action(processed_obs)
                action = self.tf.constant(start_action) if t == 0 else action
                obs_tp1, reward, _, _ = self.env.step(action.numpy())
                batch_data.append((obs.copy(), action.numpy(), reward, obs_tp1.copy()))
                obs = obs_tp1.copy()

            return {'all_rewards': np.asarray(list(map(lambda x: x[2], batch_data)), dtype=np.float32),
                    'all_obs_tp1': np.asarray(list(map(lambda x: x[3], batch_data)), dtype=np.float32),
                    }

    def compute_td_error(self):
        processed_obs = self.preprocessor.tf_process_obses(self.batch_data['batch_obs']).numpy()  # n_step*obs_dim
        processed_rewards = self.preprocessor.tf_process_rewards(self.batch_data['batch_rewards']).numpy()
        processed_obs_tp1 = self.preprocessor.tf_process_obses(self.batch_data['batch_obs_tp1']).numpy()

        values_t = self.policy_with_value.compute_Q1(processed_obs, self.batch_data['batch_actions']).numpy()
        target_act_tp1, _ = self.policy_with_value.compute_target_action(processed_obs_tp1)
        target_Q1_of_tp1 = self.policy_with_value.compute_Q1_target(processed_obs_tp1, target_act_tp1).numpy()
        td_error = processed_rewards + self.args.gamma * target_Q1_of_tp1 - values_t
        return td_error

    def compute_n_step_target(self):
        if self.sample_num_in_learner is None:
            processed_rewards = self.preprocessor.tf_process_rewards(self.batch_data['batch_rewards']).numpy()
            processed_obs_tp1 = self.preprocessor.tf_process_obses(self.batch_data['batch_obs_tp1']).numpy()
            target_act_tp1, _ = self.policy_with_value.compute_target_action(processed_obs_tp1)
            target_Q1_of_tp1 = self.policy_with_value.compute_Q1_target(processed_obs_tp1, target_act_tp1).numpy()
            return processed_rewards + self.args.gamma * target_Q1_of_tp1
        else:
            rollouts = self.sample(self.batch_data['batch_obs'], self.batch_data['batch_actions'])
            processed_all_obs_tp1 = self.preprocessor.tf_process_obses(rollouts['all_obs_tp1']).numpy()
            processed_all_rewards = self.preprocessor.tf_process_rewards(rollouts['all_rewards']).numpy()
            act_tp1, _ = self.policy_with_value.compute_target_action(
                processed_all_obs_tp1.reshape(self.sample_num_in_learner * self.batch_size, -1))
            all_values_tp1 = \
                self.policy_with_value.compute_Q1_target(
                    processed_all_obs_tp1.reshape(self.sample_num_in_learner * self.batch_size, -1),
                    act_tp1.numpy()).numpy().reshape(self.sample_num_in_learner, self.batch_size)

            if self.args.env_id == 'InvertedPendulumConti-v0':  # todo
                all_values_tp1 = self.tf.clip_by_value(all_values_tp1, -0.5, 0.)
            n_step_target = np.zeros((self.batch_size,), dtype=np.float32)
            for t in range(self.sample_num_in_learner):
                n_step_target += np.power(self.args.gamma, t) * processed_all_rewards[t]
            n_step_target += np.power(self.args.gamma, self.sample_num_in_learner) * all_values_tp1[-1, :]
            return n_step_target

    def get_weights(self):
        return self.policy_with_value.get_weights()

    def set_weights(self, weights):
        return self.policy_with_value.set_weights(weights)

    def set_ppc_params(self, params):
        self.preprocessor.set_params(params)

    @tf.function
    def q_forward_and_backward(self, mb_obs, mb_actions, mb_targets):
        processed_mb_obs = self.preprocessor.tf_process_obses(mb_obs)
        with self.tf.GradientTape() as tape:
            with self.tf.name_scope('q_loss') as scope:
                q_pred = self.policy_with_value.compute_Q1(processed_mb_obs, mb_actions)
                q_loss = 0.5 * self.tf.reduce_mean(self.tf.square(q_pred - mb_targets))

        with self.tf.name_scope('q_gradient') as scope:
            q_gradient = tape.gradient(q_loss, self.policy_with_value.Q1.trainable_weights)
        return q_loss, q_gradient

    @tf.function
    def policy_forward_and_backward(self, mb_obs):
        with self.tf.GradientTape(persistent=True) as tape:
            processed_obses = self.preprocessor.tf_process_obses(mb_obs)
            actions, _ = self.policy_with_value.compute_action(processed_obses)
            all_Qs = self.policy_with_value.compute_Q1(processed_obses, actions)
            policy_loss = -self.tf.reduce_mean(all_Qs)
            value_var = self.tf.math.reduce_variance(all_Qs)
            value_mean = -policy_loss

        with self.tf.name_scope('policy_gradient') as scope:
            policy_gradient = tape.gradient(policy_loss, self.policy_with_value.policy.trainable_weights,)
            return policy_loss, policy_gradient, value_mean, value_var

    def export_graph(self, writer):
        mb_obs = self.batch_data['batch_obs']
        mb_targets = self.batch_data['batch_targets']
        mb_actions = self.batch_data['batch_actions']
        self.tf.summary.trace_on(graph=True, profiler=False)
        self.q_forward_and_backward(mb_obs, mb_actions, mb_targets)
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
        mb_targets = self.batch_data['batch_targets']

        with self.q_gradient_timer:
            q_loss, q_gradient = self.q_forward_and_backward(mb_obs, mb_actions, mb_targets)
            q_gradient, q_gradient_norm = self.tf.clip_by_global_norm(q_gradient, self.args.gradient_clip_norm)

        with self.policy_gradient_timer:
            policy_loss, policy_gradient, value_mean, value_var = self.policy_forward_and_backward(mb_obs)

        final_policy_gradient, policy_gradient_norm = self.tf.clip_by_global_norm(policy_gradient,
                                                                                  self.args.gradient_clip_norm)

        self.stats.update(dict(
            iteration=iteration,
            q_timer=self.q_gradient_timer.mean,
            pg_time=self.policy_gradient_timer.mean,
            target_time=self.target_timer.mean,
            q_loss=q_loss.numpy(),
            policy_loss=policy_loss.numpy(),
            mb_targets_mean=np.mean(mb_targets),
            value_mean=value_mean.numpy(),
            value_var=value_var.numpy(),
            q_gradient_norm=q_gradient_norm.numpy(),
            policy_gradient_norm=policy_gradient_norm.numpy(),
        ))

        gradient_tensor = q_gradient + final_policy_gradient  # q_gradient + final_policy_gradient
        return list(map(lambda x: x.numpy(), gradient_tensor))


if __name__ == '__main__':
    pass
