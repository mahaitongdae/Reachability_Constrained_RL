#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/8/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: policy.py
# =====================================

import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers.schedules import PolynomialDecay

from model import MLPNet, AlphaModel, LamModel

NAME2MODELCLS = dict([('MLP', MLPNet),])


class PolicyWithMu(tf.Module):
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    tfb = tfp.bijectors
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    def __init__(self, obs_dim, act_dim,
                 value_model_cls, value_num_hidden_layers, value_num_hidden_units,
                 value_hidden_activation, value_lr_schedule, cost_value_lr_schedule,
                 policy_model_cls, policy_num_hidden_layers, policy_num_hidden_units, policy_hidden_activation,
                 policy_out_activation, policy_lr_schedule,
                 alpha, alpha_lr_schedule,
                 policy_only, double_Q, target, tau, delay_update,
                 deterministic_policy, action_range, lam_lr_schedule, dual_ascent_interval=1, **kwargs):
        super().__init__()
        self.policy_only = policy_only
        self.double_Q = double_Q
        self.target = target
        self.tau = tau
        self.delay_update = delay_update
        self.deterministic_policy = deterministic_policy
        self.action_range = action_range
        self.alpha = alpha
        self.dual_ascent_interval = dual_ascent_interval
        self.constrained = kwargs.get('constrained')
        self.mlp_lam = kwargs.get('mlp_lam')
        self.double_QC = kwargs.get('double_QC')
        self.penalty_start = kwargs.get('penalty_start')
        self.mu_upperbound = kwargs.get('mu_upperbound')

        value_model_cls, policy_model_cls = NAME2MODELCLS[value_model_cls], \
                                            NAME2MODELCLS[policy_model_cls]
        self.policy = policy_model_cls(obs_dim, policy_num_hidden_layers, policy_num_hidden_units,
                                       policy_hidden_activation, act_dim * 2, name='policy',
                                       output_activation=policy_out_activation)
        self.policy_target = policy_model_cls(obs_dim, policy_num_hidden_layers, policy_num_hidden_units,
                                              policy_hidden_activation, act_dim * 2, name='policy_target',
                                              output_activation=policy_out_activation)
        policy_lr = PolynomialDecay(*policy_lr_schedule)
        self.policy_optimizer = self.tf.keras.optimizers.Adam(policy_lr, name='policy_adam_opt')

        self.Q1 = value_model_cls(obs_dim + act_dim, value_num_hidden_layers, value_num_hidden_units,
                                  value_hidden_activation, 1, name='Q1')
        self.Q1_target = value_model_cls(obs_dim + act_dim, value_num_hidden_layers, value_num_hidden_units,
                                         value_hidden_activation, 1, name='Q1_target')
        self.Q1_target.set_weights(self.Q1.get_weights())
        value_lr = PolynomialDecay(*value_lr_schedule)
        self.Q1_optimizer = self.tf.keras.optimizers.Adam(value_lr, name='Q1_adam_opt')

        self.Q2 = value_model_cls(obs_dim + act_dim, value_num_hidden_layers, value_num_hidden_units,
                                  value_hidden_activation, 1, name='Q2')
        self.Q2_target = value_model_cls(obs_dim + act_dim, value_num_hidden_layers, value_num_hidden_units,
                                         value_hidden_activation, 1, name='Q2_target')
        self.Q2_target.set_weights(self.Q2.get_weights())
        self.Q2_optimizer = self.tf.keras.optimizers.Adam(value_lr, name='Q2_adam_opt')

        cost_value_lr = PolynomialDecay(*cost_value_lr_schedule)
        self.QC1 = value_model_cls(obs_dim + act_dim, value_num_hidden_layers, value_num_hidden_units,
                                   value_hidden_activation, 1, name='QC1')
        self.QC1_target = value_model_cls(obs_dim + act_dim, value_num_hidden_layers, value_num_hidden_units,
                                          value_hidden_activation, 1, name='QC1_target')
        self.QC1_target.set_weights(self.QC1.get_weights())
        self.QC1_optimizer = self.tf.keras.optimizers.Adam(cost_value_lr, name='QC1_adam_opt')

        if self.double_QC:
            self.QC2 = value_model_cls(obs_dim + act_dim, value_num_hidden_layers, value_num_hidden_units,
                                      value_hidden_activation, 1, name='QC2')
            # output_bias=kwargs.get('cost_bias')
            self.QC2_target = value_model_cls(obs_dim + act_dim, value_num_hidden_layers, value_num_hidden_units,
                                             value_hidden_activation, 1, name='QC2_target')
            self.QC2_target.set_weights(self.QC2.get_weights())
            self.QC2_optimizer = self.tf.keras.optimizers.Adam(cost_value_lr, name='QC2_adam_opt')


        if self.mlp_lam:
            lam_lr = PolynomialDecay(*lam_lr_schedule)
            self.Lam = value_model_cls(obs_dim, value_num_hidden_layers, value_num_hidden_units,
                                       value_hidden_activation, 1,
                                       name='Lam', output_activation='softplus', output_bias=0.)
            self.Lam_optimizer = self.tf.keras.optimizers.Adam(lam_lr, name='lam_opt')
        else:
            lam_lr = 3e-4
            self.Lam = LamModel(name='Lam')
            self.Lam_optimizer = self.tf.keras.optimizers.Adam(lam_lr, name='lam_opt')



        if self.policy_only:
            self.target_models = ()
            self.models = (self.policy,)
            self.optimizers = (self.policy_optimizer,)
        else:
            if self.double_Q:
                if self.double_QC:
                    assert self.target
                    self.target_models = (self.Q1_target, self.Q2_target, self.QC1_target, self.QC2_target,
                                          self.policy_target,)
                    self.models = (self.Q1, self.Q2, self.QC1, self.QC2, self.policy,self.Lam,)
                    self.optimizers = (self.Q1_optimizer, self.Q2_optimizer, self.QC1_optimizer, self.QC2_optimizer,
                                       self.policy_optimizer,self.Lam_optimizer,)
                else:
                    self.target_models = (self.Q1_target, self.Q2_target, self.QC1_target,
                                          self.policy_target,)
                    self.models = (self.Q1, self.Q2, self.QC1, self.policy, self.Lam,)
                    self.optimizers = (self.Q1_optimizer, self.Q2_optimizer, self.QC1_optimizer,
                                       self.policy_optimizer, self.Lam_optimizer,)
            elif self.target:
                self.target_models = (self.Q1_target, self.policy_target,)
                self.models = (self.Q1, self.policy,)
                self.optimizers = (self.Q1_optimizer, self.policy_optimizer,)
            else:
                self.target_models = ()
                self.models = (self.Q1, self.policy,)
                self.optimizers = (self.Q1_optimizer, self.policy_optimizer,)


        if self.alpha == 'auto':
            self.alpha_model = AlphaModel(name='alpha')
            alpha_lr = self.tf.keras.optimizers.schedules.PolynomialDecay(*alpha_lr_schedule)
            self.alpha_optimizer = self.tf.keras.optimizers.Adam(alpha_lr, name='alpha_adam_opt')
            self.models += (self.alpha_model,)
            self.optimizers += (self.alpha_optimizer,)

    def save_weights(self, save_dir, iteration):
        model_pairs = [(model.name, model) for model in self.models]
        target_model_pairs = [(target_model.name, target_model) for target_model in self.target_models]
        optimizer_pairs = [(optimizer._name, optimizer) for optimizer in self.optimizers]
        ckpt = self.tf.train.Checkpoint(**dict(model_pairs + target_model_pairs + optimizer_pairs))
        ckpt.save(save_dir + '/ckpt_ite' + str(iteration))

    def load_weights(self, load_dir, iteration):
        model_pairs = [(model.name, model) for model in self.models]
        target_model_pairs = [(target_model.name, target_model) for target_model in self.target_models]
        optimizer_pairs = [(optimizer._name, optimizer) for optimizer in self.optimizers]
        ckpt = self.tf.train.Checkpoint(**dict(model_pairs + target_model_pairs + optimizer_pairs))
        ckpt.restore(load_dir + '/ckpt_ite' + str(iteration) + '-1')

    def get_weights(self):
        return [model.get_weights() for model in self.models] + \
               [model.get_weights() for model in self.target_models]

    def set_weights(self, weights):
        for i, weight in enumerate(weights):
            if i < len(self.models):
                self.models[i].set_weights(weight)
            else:
                self.target_models[i-len(self.models)].set_weights(weight)

    @tf.function
    def apply_gradients(self, iteration, grads):
        if self.policy_only:
            policy_grad = grads
            self.policy_optimizer.apply_gradients(zip(policy_grad, self.policy.trainable_weights))
        else:
            if self.double_Q:
                q_weights_len = len(self.Q1.trainable_weights)
                policy_weights_len = len(self.policy.trainable_weights)
                lam_weights_len = len(self.Lam.trainable_weights)
                q1_grad, q2_grad, qc1_grad, qc2_grad, policy_grad =\
                    grads[:q_weights_len], \
                    grads[q_weights_len:2*q_weights_len],\
                    grads[2*q_weights_len:3*q_weights_len], \
                    grads[3 * q_weights_len:4 * q_weights_len], \
                    grads[4 * q_weights_len:4 * q_weights_len + policy_weights_len]
                lam_grad = grads[
                           4 * q_weights_len + policy_weights_len: 4 * q_weights_len + policy_weights_len + lam_weights_len]
                self.Q1_optimizer.apply_gradients(zip(q1_grad, self.Q1.trainable_weights))
                self.Q2_optimizer.apply_gradients(zip(q2_grad, self.Q2.trainable_weights))
                self.QC1_optimizer.apply_gradients(zip(qc1_grad, self.QC1.trainable_weights))
                if self.double_QC:
                    self.QC2_optimizer.apply_gradients(zip(qc2_grad, self.QC2.trainable_weights))
                if iteration % self.delay_update == 0:
                    self.policy_optimizer.apply_gradients(zip(policy_grad, self.policy.trainable_weights))
                    self.update_policy_target()
                    self.update_all_Q_target()
                    if self.alpha == 'auto':
                        alpha_grad = grads[-1:]
                        self.alpha_optimizer.apply_gradients(zip(alpha_grad, self.alpha_model.trainable_weights))
            else:
                q_weights_len = len(self.Q1.trainable_weights)
                policy_weights_len = len(self.policy.trainable_weights)
                q1_grad, policy_grad = grads[:q_weights_len], grads[q_weights_len:q_weights_len+policy_weights_len]
                self.Q1_optimizer.apply_gradients(zip(q1_grad, self.Q1.trainable_weights))
                if iteration % self.delay_update == 0:
                    self.policy_optimizer.apply_gradients(zip(policy_grad, self.policy.trainable_weights))
                    if self.alpha == 'auto':
                        alpha_grad = grads[-1:]
                        self.alpha_optimizer.apply_gradients(zip(alpha_grad, self.alpha_model.trainable_weights))
                    if self.target:
                        self.update_policy_target()
                        self.update_Q1_target()
        return qc1_grad, lam_grad

    @tf.function
    def apply_ascent_gradients(self, iteration, qc_grad, lam_grad):
        assert self.double_Q
        if iteration % self.dual_ascent_interval == 0 and self.constrained:
            self.Lam_optimizer.apply_gradients(zip(lam_grad, self.Lam.trainable_weights))

    def update_all_Q_target(self):
        self.update_Q1_target()
        self.update_Q2_target()
        self.update_QC1_target()
        if self.double_QC:
            self.update_QC2_target()

    def update_Q1_target(self):
        tau = self.tau
        for source, target in zip(self.Q1.trainable_weights, self.Q1_target.trainable_weights):
            target.assign(tau * source + (1.0 - tau) * target)

    def update_Q2_target(self):
        tau = self.tau
        for source, target in zip(self.Q2.trainable_weights, self.Q2_target.trainable_weights):
            target.assign(tau * source + (1.0 - tau) * target)

    def update_QC1_target(self):
        tau = self.tau
        for source, target in zip(self.QC1.trainable_weights, self.QC1_target.trainable_weights):
            target.assign(tau * source + (1.0 - tau) * target)

    def update_QC2_target(self):
        tau = self.tau
        for source, target in zip(self.QC2.trainable_weights, self.QC2_target.trainable_weights):
            target.assign(tau * source + (1.0 - tau) * target)

    def update_policy_target(self):
        tau = self.tau
        for source, target in zip(self.policy.trainable_weights, self.policy_target.trainable_weights):
            target.assign(tau * source + (1.0 - tau) * target)

    @tf.function
    def compute_mode(self, obs):
        logits = self.policy(obs)
        mean, _ = self.tf.split(logits, num_or_size_splits=2, axis=-1)
        return self.action_range * self.tf.tanh(mean) if self.action_range is not None else mean

    def _logits2dist(self, logits):
        mean, log_std = self.tf.split(logits, num_or_size_splits=2, axis=-1)
        log_std = tf.clip_by_value(log_std, -5., 1.)
        act_dist = self.tfd.MultivariateNormalDiag(mean, self.tf.exp(log_std))
        if self.action_range is not None:
            act_dist = (
                self.tfp.distributions.TransformedDistribution(
                    distribution=act_dist,
                    bijector=self.tfb.Chain(
                        [self.tfb.Affine(scale_identity_multiplier=self.action_range),
                         self.tfb.Tanh()])
                ))
        return act_dist

    @tf.function
    def compute_action(self, obs):
        with self.tf.name_scope('compute_action') as scope:
            logits = self.policy(obs)
            if self.deterministic_policy:
                mean, _ = self.tf.split(logits, num_or_size_splits=2, axis=-1)
                return self.action_range * self.tf.tanh(mean) if self.action_range is not None else mean, 0.
            else:
                act_dist = self._logits2dist(logits)
                actions = act_dist.sample()
                logps = act_dist.log_prob(actions)
                return actions, logps

    @tf.function
    def compute_target_action(self, obs):
        with self.tf.name_scope('compute_target_action') as scope:
            logits = self.policy_target(obs)
            if self.deterministic_policy:
                mean, _ = self.tf.split(logits, num_or_size_splits=2, axis=-1)
                return self.action_range * self.tf.tanh(mean) if self.action_range is not None else mean, 0.
            else:
                act_dist = self._logits2dist(logits)
                actions = act_dist.sample()
                logps = act_dist.log_prob(actions)
                return actions, logps

    @tf.function
    def compute_Q1(self, obs, act):
        with self.tf.name_scope('compute_Q1') as scope:
            Q_inputs = self.tf.concat([obs, act], axis=-1)
            return tf.squeeze(self.Q1(Q_inputs), axis=1)

    @tf.function
    def compute_Q2(self, obs, act):
        with self.tf.name_scope('compute_Q2') as scope:
            Q_inputs = self.tf.concat([obs, act], axis=-1)
            return tf.squeeze(self.Q2(Q_inputs), axis=1)

    @tf.function
    def compute_QC1(self, obs, act):
        with self.tf.name_scope('compute_QC1') as scope:
            Q_inputs = self.tf.concat([obs, act], axis=-1)
            return tf.squeeze(self.QC1(Q_inputs), axis=1)

    @tf.function
    def compute_QC2(self, obs, act):
        with self.tf.name_scope('compute_QC2') as scope:
            Q_inputs = self.tf.concat([obs, act], axis=-1)
            return tf.squeeze(self.QC2(Q_inputs), axis=1)

    @tf.function
    def compute_Q1_target(self, obs, act):
        with self.tf.name_scope('compute_Q1_target') as scope:
            Q_inputs = self.tf.concat([obs, act], axis=-1)
            return tf.squeeze(self.Q1_target(Q_inputs), axis=1)

    @tf.function
    def compute_Q2_target(self, obs, act):
        with self.tf.name_scope('compute_Q2_target') as scope:
            Q_inputs = self.tf.concat([obs, act], axis=-1)
            return tf.squeeze(self.Q2_target(Q_inputs), axis=1)

    @tf.function
    def compute_QC1_target(self, obs, act):
        with self.tf.name_scope('compute_QC1_target') as scope:
            Q_inputs = self.tf.concat([obs, act], axis=-1)
            return tf.squeeze(self.QC1_target(Q_inputs), axis=1)

    @tf.function
    def compute_QC2_target(self, obs, act):
        with self.tf.name_scope('compute_QC2_target') as scope:
            Q_inputs = self.tf.concat([obs, act], axis=-1)
            return tf.squeeze(self.QC2_target(Q_inputs), axis=1)

    @tf.function
    def compute_lam(self, obs):
        with self.tf.name_scope('compute_lam') as scope:
            # Q_inputs = self.tf.concat([obs], axis=-1)
            if self.mu_upperbound is not None:
                return tf.clip_by_value(tf.squeeze(self.Lam(obs), axis=1),
                                        clip_value_min=-1.,
                                        clip_value_max=self.mu_upperbound)
            else:
                return tf.squeeze(self.Lam(obs), axis=1)

    @property
    def log_alpha(self):
        return self.alpha_model.log_alpha

    @property
    def log_lam(self):
        if self.mu_upperbound is not None:
            return tf.clip_by_value(tf.nn.softplus(self.Lam.var),
                                    clip_value_min=-1.,
                                    clip_value_max=self.mu_upperbound)
        else:
            return tf.nn.softplus(self.Lam.var)


class PolicyWithQs(tf.Module):
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    tfb = tfp.bijectors
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    def __init__(self, obs_dim, act_dim,
                 value_model_cls, value_num_hidden_layers, value_num_hidden_units,
                 value_hidden_activation, value_lr_schedule,
                 policy_model_cls, policy_num_hidden_layers, policy_num_hidden_units, policy_hidden_activation,
                 policy_out_activation, policy_lr_schedule,
                 alpha, alpha_lr_schedule,
                 policy_only, double_Q, target, tau, delay_update,
                 deterministic_policy, action_range, **kwargs):
        super().__init__()
        self.policy_only = policy_only
        self.double_Q = double_Q
        self.target = target
        self.tau = tau
        self.delay_update = delay_update
        self.deterministic_policy = deterministic_policy
        self.action_range = action_range
        self.alpha = alpha

        value_model_cls, policy_model_cls = NAME2MODELCLS[value_model_cls], \
                                            NAME2MODELCLS[policy_model_cls]
        self.policy = policy_model_cls(obs_dim, policy_num_hidden_layers, policy_num_hidden_units,
                                       policy_hidden_activation, act_dim * 2, name='policy',
                                       output_activation=policy_out_activation)
        self.policy_target = policy_model_cls(obs_dim, policy_num_hidden_layers, policy_num_hidden_units,
                                              policy_hidden_activation, act_dim * 2, name='policy_target',
                                              output_activation=policy_out_activation)
        policy_lr = PolynomialDecay(*policy_lr_schedule)
        self.policy_optimizer = self.tf.keras.optimizers.Adam(policy_lr, name='policy_adam_opt')

        self.Q1 = value_model_cls(obs_dim + act_dim, value_num_hidden_layers, value_num_hidden_units,
                                  value_hidden_activation, 1, name='Q1')
        self.Q1_target = value_model_cls(obs_dim + act_dim, value_num_hidden_layers, value_num_hidden_units,
                                         value_hidden_activation, 1, name='Q1_target')
        self.Q1_target.set_weights(self.Q1.get_weights())
        value_lr = PolynomialDecay(*value_lr_schedule)
        self.Q1_optimizer = self.tf.keras.optimizers.Adam(value_lr, name='Q1_adam_opt')

        self.Q2 = value_model_cls(obs_dim + act_dim, value_num_hidden_layers, value_num_hidden_units,
                                  value_hidden_activation, 1, name='Q2')
        self.Q2_target = value_model_cls(obs_dim + act_dim, value_num_hidden_layers, value_num_hidden_units,
                                         value_hidden_activation, 1, name='Q2_target')
        self.Q2_target.set_weights(self.Q2.get_weights())
        self.Q2_optimizer = self.tf.keras.optimizers.Adam(value_lr, name='Q2_adam_opt')

        if self.policy_only:
            self.target_models = ()
            self.models = (self.policy,)
            self.optimizers = (self.policy_optimizer,)
        else:
            if self.double_Q:
                assert self.target
                self.target_models = (self.Q1_target, self.Q2_target, self.policy_target,)
                self.models = (self.Q1, self.Q2, self.policy,)
                self.optimizers = (self.Q1_optimizer, self.Q2_optimizer, self.policy_optimizer,)
            elif self.target:
                self.target_models = (self.Q1_target, self.policy_target,)
                self.models = (self.Q1, self.policy,)
                self.optimizers = (self.Q1_optimizer, self.policy_optimizer,)
            else:
                self.target_models = ()
                self.models = (self.Q1, self.policy,)
                self.optimizers = (self.Q1_optimizer, self.policy_optimizer,)

        if self.alpha == 'auto':
            self.alpha_model = AlphaModel(name='alpha')
            alpha_lr = self.tf.keras.optimizers.schedules.PolynomialDecay(*alpha_lr_schedule)
            self.alpha_optimizer = self.tf.keras.optimizers.Adam(alpha_lr, name='alpha_adam_opt')
            self.models += (self.alpha_model,)
            self.optimizers += (self.alpha_optimizer,)

    def save_weights(self, save_dir, iteration):
        model_pairs = [(model.name, model) for model in self.models]
        target_model_pairs = [(target_model.name, target_model) for target_model in self.target_models]
        optimizer_pairs = [(optimizer._name, optimizer) for optimizer in self.optimizers]
        ckpt = self.tf.train.Checkpoint(**dict(model_pairs + target_model_pairs + optimizer_pairs))
        ckpt.save(save_dir + '/ckpt_ite' + str(iteration))

    def load_weights(self, load_dir, iteration):
        model_pairs = [(model.name, model) for model in self.models]
        target_model_pairs = [(target_model.name, target_model) for target_model in self.target_models]
        optimizer_pairs = [(optimizer._name, optimizer) for optimizer in self.optimizers]
        ckpt = self.tf.train.Checkpoint(**dict(model_pairs + target_model_pairs + optimizer_pairs))
        ckpt.restore(load_dir + '/ckpt_ite' + str(iteration) + '-1')

    def get_weights(self):
        return [model.get_weights() for model in self.models] + \
               [model.get_weights() for model in self.target_models]

    def set_weights(self, weights):
        for i, weight in enumerate(weights):
            if i < len(self.models):
                self.models[i].set_weights(weight)
            else:
                self.target_models[i-len(self.models)].set_weights(weight)

    @tf.function
    def apply_gradients(self, iteration, grads):
        if self.policy_only:
            policy_grad = grads
            self.policy_optimizer.apply_gradients(zip(policy_grad, self.policy.trainable_weights))
        else:
            if self.double_Q:
                q_weights_len = len(self.Q1.trainable_weights)
                policy_weights_len = len(self.policy.trainable_weights)
                q1_grad, q2_grad, policy_grad =\
                    grads[:q_weights_len], \
                    grads[q_weights_len:2*q_weights_len],\
                    grads[2*q_weights_len:2*q_weights_len + policy_weights_len]
                self.Q1_optimizer.apply_gradients(zip(q1_grad, self.Q1.trainable_weights))
                self.Q2_optimizer.apply_gradients(zip(q2_grad, self.Q2.trainable_weights))
                if iteration % self.delay_update == 0:
                    self.policy_optimizer.apply_gradients(zip(policy_grad, self.policy.trainable_weights))
                    self.update_policy_target()
                    self.update_all_Q_target()
                    if self.alpha == 'auto':
                        alpha_grad = grads[-1:]
                        self.alpha_optimizer.apply_gradients(zip(alpha_grad, self.alpha_model.trainable_weights))
            else:
                q_weights_len = len(self.Q1.trainable_weights)
                policy_weights_len = len(self.policy.trainable_weights)
                q1_grad, policy_grad = grads[:q_weights_len], grads[q_weights_len:q_weights_len+policy_weights_len]
                self.Q1_optimizer.apply_gradients(zip(q1_grad, self.Q1.trainable_weights))
                if iteration % self.delay_update == 0:
                    self.policy_optimizer.apply_gradients(zip(policy_grad, self.policy.trainable_weights))
                    if self.alpha == 'auto':
                        alpha_grad = grads[-1:]
                        self.alpha_optimizer.apply_gradients(zip(alpha_grad, self.alpha_model.trainable_weights))
                    if self.target:
                        self.update_policy_target()
                        self.update_Q1_target()

    def update_all_Q_target(self):
        self.update_Q1_target()
        self.update_Q2_target()

    def update_Q1_target(self):
        tau = self.tau
        for source, target in zip(self.Q1.trainable_weights, self.Q1_target.trainable_weights):
            target.assign(tau * source + (1.0 - tau) * target)

    def update_Q2_target(self):
        tau = self.tau
        for source, target in zip(self.Q2.trainable_weights, self.Q2_target.trainable_weights):
            target.assign(tau * source + (1.0 - tau) * target)

    def update_policy_target(self):
        tau = self.tau
        for source, target in zip(self.policy.trainable_weights, self.policy_target.trainable_weights):
            target.assign(tau * source + (1.0 - tau) * target)

    @tf.function
    def compute_mode(self, obs):
        logits = self.policy(obs)
        mean, _ = self.tf.split(logits, num_or_size_splits=2, axis=-1)
        return self.action_range * self.tf.tanh(mean) if self.action_range is not None else mean

    def _logits2dist(self, logits):
        mean, log_std = self.tf.split(logits, num_or_size_splits=2, axis=-1)
        log_std = tf.clip_by_value(log_std, -5., 1.)
        act_dist = self.tfd.MultivariateNormalDiag(mean, self.tf.exp(log_std))
        if self.action_range is not None:
            act_dist = (
                self.tfp.distributions.TransformedDistribution(
                    distribution=act_dist,
                    bijector=self.tfb.Chain(
                        [self.tfb.Affine(scale_identity_multiplier=self.action_range),
                         self.tfb.Tanh()])
                ))
        return act_dist

    @tf.function
    def compute_action(self, obs):
        with self.tf.name_scope('compute_action') as scope:
            logits = self.policy(obs)
            if self.deterministic_policy:
                mean, _ = self.tf.split(logits, num_or_size_splits=2, axis=-1)
                return self.action_range * self.tf.tanh(mean) if self.action_range is not None else mean, 0.
            else:
                act_dist = self._logits2dist(logits)
                actions = act_dist.sample()
                logps = act_dist.log_prob(actions)
                return actions, logps

    @tf.function
    def compute_target_action(self, obs):
        with self.tf.name_scope('compute_target_action') as scope:
            logits = self.policy_target(obs)
            if self.deterministic_policy:
                mean, _ = self.tf.split(logits, num_or_size_splits=2, axis=-1)
                return self.action_range * self.tf.tanh(mean) if self.action_range is not None else mean, 0.
            else:
                act_dist = self._logits2dist(logits)
                actions = act_dist.sample()
                logps = act_dist.log_prob(actions)
                return actions, logps

    @tf.function
    def compute_Q1(self, obs, act):
        with self.tf.name_scope('compute_Q1') as scope:
            Q_inputs = self.tf.concat([obs, act], axis=-1)
            return tf.squeeze(self.Q1(Q_inputs), axis=1)

    @tf.function
    def compute_Q2(self, obs, act):
        with self.tf.name_scope('compute_Q2') as scope:
            Q_inputs = self.tf.concat([obs, act], axis=-1)
            return tf.squeeze(self.Q2(Q_inputs), axis=1)

    @tf.function
    def compute_Q1_target(self, obs, act):
        with self.tf.name_scope('compute_Q1_target') as scope:
            Q_inputs = self.tf.concat([obs, act], axis=-1)
            return tf.squeeze(self.Q1_target(Q_inputs), axis=1)

    @tf.function
    def compute_Q2_target(self, obs, act):
        with self.tf.name_scope('compute_Q2_target') as scope:
            Q_inputs = self.tf.concat([obs, act], axis=-1)
            return tf.squeeze(self.Q2_target(Q_inputs), axis=1)

    @property
    def log_alpha(self):
        return self.alpha_model.log_alpha