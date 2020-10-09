#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/8/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: mpg_learner.py
# =====================================

import logging

import gym
import numpy as np
from gym.envs.user_defined.path_tracking_env import EnvironmentModel

from preprocessor import Preprocessor
from utils.misc import TimerStat

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MPGLearner(object):
    import tensorflow as tf

    def __init__(self, policy_cls, args):
        self.args = args
        self.sample_num_in_learner = self.args.sample_num_in_learner
        self.batch_size = self.args.replay_batch_size
        self.env = gym.make(self.args.env_id, num_agent=self.batch_size, num_future_data=self.args.num_future_data)
        obs_space, act_space = self.env.observation_space, self.env.action_space
        self.policy_with_value = policy_cls(obs_space, act_space, self.args)
        self.batch_data = {}
        self.counter = 0
        self.num_batch_reuse = self.args.num_batch_reuse
        self.policy_for_rollout = policy_cls(obs_space, act_space, self.args)
        self.M = self.args.M
        self.num_rollout_list_for_policy_update = self.args.num_rollout_list_for_policy_update
        self.num_rollout_list_for_q_estimation = self.args.num_rollout_list_for_q_estimation

        self.model = EnvironmentModel(num_future_data=self.args.num_future_data)  # TODO
        self.preprocessor = Preprocessor(obs_space, self.args.obs_preprocess_type, self.args.reward_preprocess_type,
                                         self.args.obs_scale_factor, self.args.reward_scale_factor,
                                         gamma=self.args.gamma)
        self.policy_gradient_timer = TimerStat()
        self.q_gradient_timer = TimerStat()
        self.target_timer = TimerStat()
        self.stats = {}
        self.info_for_buffer = {}
        self.ws_old = self.tf.convert_to_tensor([0.]+[1./(len(self.num_rollout_list_for_policy_update)-1)
                                for _ in range(len(self.num_rollout_list_for_policy_update)-1)], dtype=self.tf.float32)

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
            if self.args.learner_version == 'MPG-v1' or self.args.learner_version == 'MPG-v3':
                target = self.compute_n_step_target()
            elif self.args.learner_version == 'MPG-v2':
                target = self.compute_clipped_double_q_target()
            else:
                raise ValueError

        self.batch_data.update(dict(batch_targets=target,))
        if self.args.buffer_type != 'normal':
            self.info_for_buffer.update(dict(td_error=self.compute_td_error(),
                                             rb=rb,
                                             indexes=indexes))

    def sample(self, start_obs, start_action):
        batch_data = []
        obs = start_obs
        self.env.reset(init_obs=obs)
        for t in range(self.sample_num_in_learner):
            if t == 0:
                action = self.tf.convert_to_tensor(start_action)
            else:
                processed_obs = self.preprocessor.tf_process_obses(obs).numpy()
                action, logp = self.policy_with_value.compute_action(processed_obs)
            obs_tp1, reward, _, info = self.env.step(action.numpy())

            done = np.zeros((self.batch_size,), dtype=np.int)
            batch_data.append((obs, action.numpy(), reward, obs_tp1, done))
            obs = obs_tp1.copy()

        return batch_data

    def compute_clipped_double_q_target(self):
        processed_rewards = self.preprocessor.tf_process_rewards(self.batch_data['batch_rewards']).numpy()
        processed_obs_tp1 = self.preprocessor.tf_process_obses(self.batch_data['batch_obs_tp1']).numpy()
        target_act_tp1, _ = self.policy_with_value.compute_target_action(processed_obs_tp1)
        target_Q1_of_tp1 = self.policy_with_value.compute_Q1_target(processed_obs_tp1, target_act_tp1).numpy()[:, 0]
        target_Q2_of_tp1 = self.policy_with_value.compute_Q2_target(processed_obs_tp1, target_act_tp1).numpy()[:, 0]
        clipped_double_q_target = processed_rewards + self.args.gamma * np.minimum(target_Q1_of_tp1, target_Q2_of_tp1)
        return clipped_double_q_target

    def compute_td_error(self):
        processed_obs = self.preprocessor.tf_process_obses(self.batch_data['batch_obs']).numpy()  # n_step*obs_dim
        processed_rewards = self.preprocessor.tf_process_rewards(self.batch_data['batch_rewards']).numpy()
        processed_obs_tp1 = self.preprocessor.tf_process_obses(self.batch_data['batch_obs_tp1']).numpy()

        values_t = self.policy_with_value.compute_Q1(processed_obs, self.batch_data['batch_actions']).numpy()[:, 0]
        target_act_tp1, _ = self.policy_with_value.compute_target_action(processed_obs_tp1)
        target_Q1_of_tp1 = self.policy_with_value.compute_Q1_target(processed_obs_tp1, target_act_tp1).numpy()[:, 0]
        td_error = processed_rewards + self.args.gamma * target_Q1_of_tp1 - values_t
        return td_error

    def compute_n_step_target(self):
        rollouts = self.sample(self.batch_data['batch_obs'], self.batch_data['batch_actions'])
        tmp = {'all_rewards': np.asarray(list(map(lambda x: x[2], rollouts)), dtype=np.float32),
               'all_obs_tp1': np.asarray(list(map(lambda x: x[3], rollouts)), dtype=np.float32),
               }
        processed_all_obs_tp1 = self.preprocessor.tf_process_obses(tmp['all_obs_tp1']).numpy()
        processed_all_rewards = self.preprocessor.tf_process_rewards(tmp['all_rewards']).numpy()
        act_tp1, _ = self.policy_with_value.compute_target_action(
            processed_all_obs_tp1.reshape(self.sample_num_in_learner * self.batch_size, -1))
        all_values_tp1 = \
            self.policy_with_value.compute_Q1_target(
                processed_all_obs_tp1.reshape(self.sample_num_in_learner * self.batch_size, -1),
                act_tp1.numpy()).numpy().reshape(self.sample_num_in_learner, self.batch_size)

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

    def model_rollout_for_q_estimation(self, start_obses, start_actions):  # only use to calculate heuristic bias
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
                self.tf.concat(processed_obses_tile_list, 0), self.tf.concat(actions_tile_list, 0))[:, 0]
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
                actions_tile, _ = self.policy_for_rollout.compute_action(processed_obses_tile) if not \
                    self.args.deriv_interval_policy else self.policy_with_value.compute_action(processed_obses_tile)
                processed_obses_tile_list.append(processed_obses_tile)
                actions_tile_list.append(actions_tile)
                gammas_list.append(self.tf.pow(self.args.gamma, ri + 1) * self.tf.ones((obses_tile.shape[0],)))

        with self.tf.name_scope('compute_all_model_returns') as scope:
            all_rewards_sums = self.tf.concat(rewards_sum_list, 0)
            all_gammas = self.tf.concat(gammas_list, 0)
            if self.args.learner_version == 'MPG-v2':
                all_Q1s = self.policy_with_value.compute_Q1(
                    self.tf.concat(processed_obses_tile_list, 0), self.tf.concat(actions_tile_list, 0))[:, 0]
                all_Q2s = self.policy_with_value.compute_Q2(
                    self.tf.concat(processed_obses_tile_list, 0), self.tf.concat(actions_tile_list, 0))[:, 0]
                all_min_Qs = self.tf.reduce_min((all_Q1s, all_Q2s), axis=0)
                final = self.tf.reshape(all_rewards_sums + all_gammas * all_min_Qs, (max_num_rollout + 1, self.M, -1))
            else:
                all_Qs = self.policy_with_value.compute_Q1(
                    self.tf.concat(processed_obses_tile_list, 0), self.tf.concat(actions_tile_list, 0))[:, 0]
                final = self.tf.reshape(all_rewards_sums + all_gammas * all_Qs, (max_num_rollout + 1, self.M, -1))
            # final [[[time0+traj0], [time0+traj1], ..., [time0+trajn]],
            #        [[time1+traj0], [time1+traj1], ..., [time1+trajn]],
            #        ...
            #        [[timen+traj0], [timen+traj1], ..., [timen+trajn]],
            #        ]
            all_model_returns = self.tf.reduce_mean(final, axis=1)
        all_reduced_model_returns = self.tf.reduce_mean(all_model_returns, axis=1)
        all_model_returns_var = self.tf.math.reduce_variance(all_model_returns, axis=1)

        selected_model_returns, selected_model_returns_var, minus_selected_reduced_model_returns = [], [], []
        for num_rollout in self.num_rollout_list_for_policy_update:
            # selected_model_returns.append(all_model_returns[num_rollout])
            selected_model_returns_var.append(all_model_returns_var[num_rollout])
            minus_selected_reduced_model_returns.append(-all_reduced_model_returns[num_rollout])

        # selected_model_returns_flatten = self.tf.concat(selected_model_returns, 0)
        selected_model_returns_var = self.tf.stack(selected_model_returns_var, 0)
        minus_selected_reduced_model_returns = self.tf.stack(minus_selected_reduced_model_returns, 0)
        value_mean = self.tf.reduce_mean(all_model_returns[0])
        return selected_model_returns_var, minus_selected_reduced_model_returns, value_mean

    @tf.function
    def q_forward_and_backward(self, mb_obs, mb_actions, mb_targets):
        processed_mb_obs = self.preprocessor.tf_process_obses(mb_obs)
        model_mse_list = []
        if self.args.learner_version == 'MPG-v1':
            model_targets = self.model_rollout_for_q_estimation(mb_obs, mb_actions)
            for i, num_rollout in enumerate(self.num_rollout_list_for_q_estimation):
                model_target_i = model_targets[i * self.batch_size:
                                               (i + 1) * self.batch_size]
                model_mse_list.append(self.tf.reduce_mean(self.tf.square(model_target_i - mb_targets)))

        if self.args.learner_version == 'MPG-v1' or self.args.learner_version == 'MPG-v3':
            with self.tf.GradientTape() as tape:
                with self.tf.name_scope('q_loss') as scope:
                    q_pred1 = self.policy_with_value.compute_Q1(processed_mb_obs, mb_actions)[:, 0]
                    q_loss1 = 0.5 * self.tf.reduce_mean(self.tf.square(q_pred1 - mb_targets))

            with self.tf.name_scope('q_gradient') as scope:
                q_gradient1 = tape.gradient(q_loss1, self.policy_with_value.Q1.trainable_weights)

            return q_loss1, q_gradient1, self.tf.convert_to_tensor(model_mse_list)

        else:
            with self.tf.GradientTape(persistent=True) as tape:
                with self.tf.name_scope('q_loss') as scope:
                    q_pred1 = self.policy_with_value.compute_Q1(processed_mb_obs, mb_actions)[:, 0]
                    q_loss1 = 0.5 * self.tf.reduce_mean(self.tf.square(q_pred1 - mb_targets))

                    q_pred2 = self.policy_with_value.compute_Q2(processed_mb_obs, mb_actions)[:, 0]
                    q_loss2 = 0.5 * self.tf.reduce_mean(self.tf.square(q_pred2 - mb_targets))

            with self.tf.name_scope('q_gradient') as scope:
                q_gradient1 = tape.gradient(q_loss1, self.policy_with_value.Q1.trainable_weights)
                q_gradient2 = tape.gradient(q_loss2, self.policy_with_value.Q2.trainable_weights)

            return q_loss1, q_loss2, q_gradient1, q_gradient2, self.tf.convert_to_tensor(model_mse_list)

    @tf.function
    def policy_forward_and_backward(self, mb_obs, ite, model_mse_tensor, ws_old):
        with self.tf.GradientTape() as tape:
            model_returns_var, minus_reduced_model_returns, value_mean = self.model_rollout_for_policy_update(mb_obs)
            if self.args.learner_version == 'MPG-v2' or self.args.learner_version == 'MPG-v3':
                ws = ws_new = self.rule_based_weights(ite, self.args.rule_based_bias_total_ite, self.args.eta)
            else:
                ws, ws_new = self.heuristic_weights(ws_old, model_mse_tensor)
            total_loss = self.tf.reduce_sum(self.tf.stop_gradient(ws)*minus_reduced_model_returns)
        with self.tf.name_scope('policy_gradient') as scope:
            policy_gradient = tape.gradient(total_loss,
                                            self.policy_with_value.policy.trainable_weights)
            return policy_gradient, total_loss, value_mean, ws, ws_new, minus_reduced_model_returns

    def export_graph(self, writer):
        mb_obs = self.batch_data['batch_obs']
        mb_targets = self.batch_data['batch_targets']
        mb_actions = self.batch_data['batch_actions']
        self.tf.summary.trace_on(graph=True, profiler=False)
        self.q_forward_and_backward(mb_obs, mb_actions, mb_targets)
        with writer.as_default():
            self.tf.summary.trace_export(name="q_forward_and_backward", step=0)

        self.tf.summary.trace_on(graph=True, profiler=False)
        self.policy_forward_and_backward(mb_obs,
                                         self.tf.constant(0, dtype=self.tf.int32),
                                         self.tf.ones(len(self.num_rollout_list_for_policy_update)),
                                         self.ws_old)
        with writer.as_default():
            self.tf.summary.trace_export(name="policy_forward_and_backward", step=0)

    def rule_based_weights(self, ite, total_ite, eta):
        start = self.tf.convert_to_tensor(1. - eta, dtype=self.tf.float32)
        slope = self.tf.convert_to_tensor(2. * eta / total_ite, dtype=self.tf.float32)
        lam = start + slope * ite
        lam = self.tf.clip_by_value(lam, 0, 1.5)
        if lam < 1.:
            biases = self.tf.convert_to_tensor([self.tf.pow(lam, i) for i in self.num_rollout_list_for_policy_update],
                                               dtype=self.tf.float32)
        else:
            max_index = max(self.num_rollout_list_for_policy_update)
            biases = self.tf.convert_to_tensor([self.tf.pow(2-lam, max_index-i) for i in self.num_rollout_list_for_policy_update],
                                               dtype=self.tf.float32)
        epsilon = 1e-8
        bias_inverses = 1. / (biases + epsilon)
        ws = self.tf.nn.softmax(bias_inverses)
        return ws

    def heuristic_weights(self, ws_old, model_mse_tensor):
        epsilon = 1e-8
        mse_inverse = 1. / (model_mse_tensor + epsilon)
        if ws_old[0] < 0.98:
            ws_new = (1. / (model_mse_tensor + epsilon)) / self.tf.reduce_sum(mse_inverse)

            ws = ws_old + self.args.w_moving_rate * (ws_new - ws_old)
            return ws, ws_new
        else:
            ws = ws_new = self.tf.convert_to_tensor([1., 0])
            return ws, ws_new

    def compute_gradient(self, batch_data, rb, indexes, iteration):  # compute gradient
        if self.counter % self.num_batch_reuse == 0:
            self.get_batch_data(batch_data, rb, indexes)
        self.counter += 1
        if self.args.buffer_type != 'normal':
            self.info_for_buffer.update(dict(td_error=self.compute_td_error()))
        mb_obs = self.batch_data['batch_obs']
        mb_actions = self.batch_data['batch_actions']
        mb_targets = self.batch_data['batch_targets']
        rewards_mean = np.abs(np.mean(self.preprocessor.np_process_rewards(self.batch_data['batch_rewards'])))

        with self.q_gradient_timer:
            if self.args.learner_version == 'MPG-v1' or self.args.learner_version == 'MPG-v3':
                q_loss1, q_gradient1, model_mse_tensor = self.q_forward_and_backward(mb_obs, mb_actions, mb_targets)
                q_gradient1, q_gradient_norm1 = self.tf.clip_by_global_norm(q_gradient1, self.args.gradient_clip_norm)
            else:
                q_loss1, q_loss2, q_gradient1, q_gradient2, model_mse_tensor = self.q_forward_and_backward(mb_obs, mb_actions, mb_targets)
                q_gradient1, q_gradient_norm1 = self.tf.clip_by_global_norm(q_gradient1, self.args.gradient_clip_norm)
                q_gradient2, q_gradient_norm2 = self.tf.clip_by_global_norm(q_gradient2, self.args.gradient_clip_norm)

        with self.policy_gradient_timer:
            self.policy_for_rollout.set_weights(self.policy_with_value.get_weights())
            policy_gradient, total_loss, value_mean, ws, ws_new, all_losses = \
                self.policy_forward_and_backward(mb_obs,
                                                 self.tf.convert_to_tensor(iteration, dtype=self.tf.float32),
                                                 model_mse_tensor,
                                                 self.ws_old)
            self.ws_old = ws

        policy_gradient, policy_gradient_norm = self.tf.clip_by_global_norm(policy_gradient,
                                                                            self.args.gradient_clip_norm)

        self.stats.update(dict(
            iteration=iteration,
            q_timer=self.q_gradient_timer.mean,
            pg_time=self.policy_gradient_timer.mean,
            target_time=self.target_timer.mean,
            value_mean=value_mean.numpy(),
            policy_total_loss=total_loss.numpy(),
            policy_gradient_norm=policy_gradient_norm.numpy(),
            q_loss1=q_loss1.numpy(),
            q_gradient_norm1=q_gradient_norm1.numpy(),
            num_rollout_list=self.num_rollout_list_for_policy_update,
            w_list_new=list(ws_new.numpy()),
            w_list=list(ws.numpy()),
            all_losses=list(all_losses.numpy())
        ))
        if self.args.learner_version == 'MPG-v1' or self.args.learner_version == 'MPG-v3':
            gradient_tensor = q_gradient1 + policy_gradient
        else:
            self.stats.update(dict(q_loss2=q_loss2.numpy(),
                                   q_gradient_norm2=q_gradient_norm2.numpy(),))
            gradient_tensor = q_gradient1 + q_gradient2 + policy_gradient

        return list(map(lambda x: x.numpy(), gradient_tensor))


def test_rule_based_weights():
    import matplotlib.pyplot as plt
    import tensorflow as tf
    num_rollout_list_for_policy_update = [i for i in range(0,20,2)]

    def rule_based_bias(ite, total_ite, eta):
        start = 1 - eta
        slope = 2 * eta / total_ite
        lam = start + slope * ite
        lam = np.clip(lam, 0, 1.5)
        if lam < 1:
            bias_list = [np.power(lam, i) for i in num_rollout_list_for_policy_update]
        else:
            max_index = max(num_rollout_list_for_policy_update)
            bias_list = [np.power(2-lam, max_index-i) for i in num_rollout_list_for_policy_update]
        bias_inverse = list(map(lambda x: 1. / (x + 1e-8), bias_list))
        bias_inverse_sum = sum(bias_inverse)
        w_bias_list = list(map(lambda x: (1. / (x + 1e-8)) / bias_inverse_sum, bias_list))
        w_sm = tf.nn.softmax(bias_inverse).numpy()
        return bias_list, w_sm

    # plot i-th elem of all iters
    ite = list(range(30000))
    i = 0
    elem_i_weights1 = list(map(lambda x_: rule_based_bias(x_, 20000, 0.1)[1][i], ite))
    # elem_i_weights2 = list(map(lambda x_: rule_based_bias(x_, 10000, 0.2)[1][i], ite))
    # elem_i_weights3 = list(map(lambda x_: rule_based_bias(x_, 10000, 0.3)[1][i], ite))

    plt.figure('first elem of all iters')
    plt.plot(elem_i_weights1, 'r')
    # plt.plot(elem_i_weights2, 'g')
    # plt.plot(elem_i_weights3, 'b')
    #
    # plot all elems in some iter
    ite = 30000
    _, w_bias_list1 = rule_based_bias(ite, 20000, 0.1)
    # _, w_bias_list2 = rule_based_bias(ite, 20000, 0.2)
    # _, w_bias_list3 = rule_based_bias(ite, 20000, 0.3)

    plt.figure('all elems in some iter')
    plt.plot(w_bias_list1, 'r')
    # plt.plot(w_bias_list2, 'g')
    # plt.plot(w_bias_list3, 'b')

    # plt.plot(np.tanh([x for x in range(10)]))

    plt.show()

def test_moving_weights():
    w_old = 0.
    for i in range(100):
        w_new = 0.8
        w = w_old + 0.2 * (w_new - w_old)
        w_old = w
        print(w)


if __name__ == '__main__':
    test_moving_weights()
