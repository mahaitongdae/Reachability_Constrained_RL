#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/9/1
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: policy.py
# =====================================

import tensorflow as tf
import numpy as np
from gym import spaces
from tensorflow.keras.optimizers.schedules import PolynomialDecay

from model import MLPNet

NAME2MODELCLS = dict([('MLP', MLPNet),])


class Policy4Toyota(tf.Module):
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    tfb = tfp.bijectors
    tf.config.experimental.set_visible_devices([], 'GPU')

    def __init__(self, args):
        super().__init__()
        self.args = args
        obs_dim, act_dim = self.args.obs_dim, self.args.act_dim
        n_hiddens, n_units, hidden_activation = self.args.num_hidden_layers, self.args.num_hidden_units, self.args.hidden_activation
        policy_model_cls = NAME2MODELCLS[self.args.policy_model_cls]
        self.policy = policy_model_cls(obs_dim, n_hiddens, n_units, hidden_activation, act_dim * 2, name='policy',
                                       output_activation=self.args.policy_out_activation)
        policy_lr_schedule = PolynomialDecay(*self.args.policy_lr_schedule)
        self.policy_optimizer = self.tf.keras.optimizers.Adam(policy_lr_schedule, name='adam_opt')

        self.models = (self.policy,)
        self.optimizers = (self.policy_optimizer,)

    def save_weights(self, save_dir, iteration):
        model_pairs = [(model.name, model) for model in self.models]
        optimizer_pairs = [(optimizer._name, optimizer) for optimizer in self.optimizers]
        ckpt = self.tf.train.Checkpoint(**dict(model_pairs + optimizer_pairs))
        ckpt.save(save_dir + '/ckpt_ite' + str(iteration))

    def load_weights(self, load_dir, iteration):
        model_pairs = [(model.name, model) for model in self.models]
        optimizer_pairs = [(optimizer._name, optimizer) for optimizer in self.optimizers]
        ckpt = self.tf.train.Checkpoint(**dict(model_pairs + optimizer_pairs))
        ckpt.restore(load_dir + '/ckpt_ite' + str(iteration) + '-1')

    def get_weights(self):
        return [model.get_weights() for model in self.models]

    def set_weights(self, weights):
        for i, weight in enumerate(weights):
            self.models[i].set_weights(weight)

    @tf.function
    def apply_gradients(self, iteration, grads):
        self.policy_optimizer.apply_gradients(zip(grads, self.policy.trainable_weights))

    @tf.function
    def compute_mode(self, obs):
        logits = self.policy(obs)
        mean, _ = self.tf.split(logits, num_or_size_splits=2, axis=-1)
        return self.args.action_range * self.tf.tanh(mean) if self.args.action_range is not None else mean

    @tf.function
    def compute_action(self, obs):
        with self.tf.name_scope('compute_action') as scope:
            logits = self.policy(obs)
            if self.args.deterministic_policy:
                mean, log_std = self.tf.split(logits, num_or_size_splits=2, axis=-1)
                return self.args.action_range * self.tf.tanh(mean) if self.args.action_range is not None else mean, 0.

class Policy4Lagrange(tf.Module):
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    tfb = tfp.bijectors
    tf.config.experimental.set_visible_devices([], 'GPU')

    def __init__(self, args):
        super().__init__()
        self.args = args
        obs_dim, act_dim = self.args.obs_dim, self.args.act_dim
        mu_dim = self.args.con_dim
        n_hiddens, n_units, hidden_activation = self.args.num_hidden_layers, self.args.num_hidden_units, self.args.hidden_activation
        value_model_cls, policy_model_cls = NAME2MODELCLS[self.args.value_model_cls], \
                                            NAME2MODELCLS[self.args.policy_model_cls]
        mu_model_cls = NAME2MODELCLS[self.args.mu_model_cls]
        self.policy = policy_model_cls(obs_dim, n_hiddens, n_units, hidden_activation, act_dim * 2, name='policy',
                                       output_activation=self.args.policy_out_activation)
        policy_lr_schedule = PolynomialDecay(*self.args.policy_lr_schedule)
        self.policy_optimizer = self.tf.keras.optimizers.Adam(policy_lr_schedule, name='adam_opt')

        self.obj_v = value_model_cls(obs_dim, n_hiddens, n_units, hidden_activation, 1, name='obj_v')
        self.con_v = value_model_cls(obs_dim, n_hiddens, n_units, hidden_activation, 1, name='con_v')

        self.mu = mu_model_cls(obs_dim, n_hiddens, n_units, hidden_activation, mu_dim, name='mu', output_activation=
                               self.args.mu_out_activation)

        mu_value_lr_schedule = PolynomialDecay(*self.args.value_lr_schedule)
        self.mu_optimizer = self.tf.optimizers.Adam(mu_value_lr_schedule, name='mu_adam_opt')

        # obj_value_lr_schedule = PolynomialDecay(*self.args.value_lr_schedule)
        # self.obj_value_optimizer = self.tf.keras.optimizers.Adam(obj_value_lr_schedule, name='objv_adam_opt')
        #
        # con_value_lr_schedule = PolynomialDecay(*self.args.value_lr_schedule)
        # self.con_value_optimizer = self.tf.keras.optimizers.Adam(con_value_lr_schedule, name='conv_adam_opt')

        self.models = (self.policy, self.mu)
        self.optimizers = (self.policy_optimizer, self.mu) # self.obj_value_optimizer, self.con_value_optimizer,

    def save_weights(self, save_dir, iteration):
        model_pairs = [(model.name, model) for model in self.models]
        optimizer_pairs = [(optimizer._name, optimizer) for optimizer in self.optimizers]
        ckpt = self.tf.train.Checkpoint(**dict(model_pairs + optimizer_pairs))
        ckpt.save(save_dir + '/ckpt_ite' + str(iteration))

    def load_weights(self, load_dir, iteration):
        model_pairs = [(model.name, model) for model in self.models]
        optimizer_pairs = [(optimizer._name, optimizer) for optimizer in self.optimizers]
        ckpt = self.tf.train.Checkpoint(**dict(model_pairs + optimizer_pairs))
        ckpt.restore(load_dir + '/ckpt_ite' + str(iteration) + '-1')

    def get_weights(self):
        return [model.get_weights() for model in self.models]

    def set_weights(self, weights):
        for i, weight in enumerate(weights):
            self.models[i].set_weights(weight)

    @tf.function
    def apply_gradients(self, iteration, grads):
        policy_len = len(self.policy.trainable_weights)
        policy_grad, mu_grad = grads[:policy_len], grads[policy_len:]
        self.policy_optimizer.apply_gradients(zip(policy_grad, self.policy.trainable_weights))
        self.mu_optimizer.apply_gradients(zip(mu_grad, self.mu.trainable_weights))

    @tf.function
    def compute_mode(self, obs):
        logits = self.policy(obs)
        mean, _ = self.tf.split(logits, num_or_size_splits=2, axis=-1)
        return self.args.action_range * self.tf.tanh(mean) if self.args.action_range is not None else mean

    def _logits2dist(self, logits):
        mean, log_std = self.tf.split(logits, num_or_size_splits=2, axis=-1)
        act_dist = self.tfd.MultivariateNormalDiag(mean, self.tf.exp(log_std))
        if self.args.action_range is not None:
            act_dist = (
                self.tfp.distributions.TransformedDistribution(
                    distribution=act_dist,
                    bijector=self.tfb.Chain(
                        [self.tfb.Affine(scale_identity_multiplier=self.args.action_range),
                         self.tfb.Tanh()])
                ))
        return act_dist

    @tf.function
    def compute_action(self, obs):
        with self.tf.name_scope('compute_action') as scope:
            logits = self.policy(obs)
            if self.args.deterministic_policy:
                mean, log_std = self.tf.split(logits, num_or_size_splits=2, axis=-1)
                return self.args.action_range * self.tf.tanh(mean) if self.args.action_range is not None else mean, 0.
            else:
                act_dist = self._logits2dist(logits)
                actions = act_dist.sample()
                logps = act_dist.log_prob(actions)
                return actions, logps

    # @tf.function
    # def compute_obj_v(self, obs):
    #     with self.tf.name_scope('compute_obj_v') as scope:
    #         return tf.squeeze(self.obj_v(obs), axis=1)
    #
    # @tf.function
    # def compute_con_v(self, obs):
    #     with self.tf.name_scope('compute_con_v') as scope:
    #         return tf.squeeze(self.con_v(obs), axis=1)

    @tf.function
    def compute_mu(self, obs):
        with self.tf.name_scope('compute_mu') as scope:
            return self.mu(obs)

def test_policy():
    import gym
    from train_script import built_mixedpg_parser
    args = built_mixedpg_parser()
    print(args.obs_dim, args.act_dim)
    env = gym.make('PathTracking-v0')
    policy = PolicyWithQs(env.observation_space, env.action_space, args)
    obs = np.random.random((128, 6))
    act = np.random.random((128, 2))
    Qs = policy.compute_Qs(obs, act)
    print(Qs)

def test_policy2():
    from train_script import built_mixedpg_parser
    import gym
    args = built_mixedpg_parser()
    env = gym.make('Pendulum-v0')
    policy_with_value = PolicyWithQs(env.observation_space, env.action_space, args)

def test_policy_with_Qs():
    from train_script import built_mixedpg_parser
    import gym
    import numpy as np
    import tensorflow as tf
    args = built_mixedpg_parser()
    args.obs_dim = 3
    env = gym.make('Pendulum-v0')
    policy_with_value = PolicyWithQs(env.observation_space, env.action_space, args)
    # print(policy_with_value.policy.trainable_weights)
    # print(policy_with_value.Qs[0].trainable_weights)
    obses = np.array([[1., 2., 3.], [3., 4., 5.]], dtype=np.float32)

    with tf.GradientTape() as tape:
        acts, _ = policy_with_value.compute_action(obses)
        Qs = policy_with_value.compute_Qs(obses, acts)[0]
        print(Qs)
        loss = tf.reduce_mean(Qs)

    gradient = tape.gradient(loss, policy_with_value.policy.trainable_weights)
    print(gradient)

def test_mlp():
    import tensorflow as tf
    import numpy as np
    policy = tf.keras.Sequential([tf.keras.layers.Dense(128, input_shape=(3,), activation='elu'),
                                  tf.keras.layers.Dense(128, input_shape=(3,), activation='elu'),
                                  tf.keras.layers.Dense(1, activation='elu')])
    value = tf.keras.Sequential([tf.keras.layers.Dense(128, input_shape=(4,), activation='elu'),
                                  tf.keras.layers.Dense(128, input_shape=(3,), activation='elu'),
                                  tf.keras.layers.Dense(1, activation='elu')])
    print(policy.trainable_variables)
    print(value.trainable_variables)
    with tf.GradientTape() as tape:
        obses = np.array([[1., 2., 3.], [3., 4., 5.]], dtype=np.float32)
        obses = tf.convert_to_tensor(obses)
        acts = policy(obses)
        a = tf.reduce_mean(acts)
        print(acts)
        Qs = value(tf.concat([obses, acts], axis=-1))
        print(Qs)
        loss = tf.reduce_mean(Qs)

    gradient = tape.gradient(loss, policy.trainable_weights)
    print(gradient)

def test_policy_for_Lag():
    from train_script import built_AMPC_parser
    import gym
    import numpy as np
    import tensorflow as tf
    args = built_AMPC_parser()
    # env = gym.make('CrossroadEnd2end-v4')
    args.obs_dim = 3
    args.act_dim = 2
    policy_with_value = Policy4Lagrange(args)
    # print(policy_with_value.policy.trainable_weights)
    # print(policy_with_value.Qs[0].trainable_weights)
    obses = np.array([[1., 2., 3.], [3., 4., 5.]], dtype=np.float32)

    with tf.GradientTape() as tape:
        acts, _ = policy_with_value.compute_action(obses)
        mu = policy_with_value.compute_mu(obses)
        print(mu)
        loss = tf.reduce_mean(mu)

    gradient = tape.gradient(loss, policy_with_value.mu.trainable_weights)
    print(gradient)



if __name__ == '__main__':
    test_policy_for_Lag()
