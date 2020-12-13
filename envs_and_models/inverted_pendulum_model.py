#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/12/8
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: inverted_pendulum_model.py
# =====================================
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import matplotlib.pyplot as plt
import numpy as np


class Dynamics(object):
    def __init__(self, if_model=False):
        self.mass_cart = 9.42
        self.mass_rod1 = 4.89
        self.mass_rod2 = 0.
        self.l_rod1 = 0.6
        self.l_rod2 = 0.
        self.g = 9.81
        self.damping_cart = 0.
        self.damping_rod1 = 0.
        self.if_model = if_model

    def f_xu(self, states, actions, tau):
        m, m1, m2 = tf.constant(self.mass_cart, dtype=tf.float32),\
                    tf.constant(self.mass_rod1, dtype=tf.float32), \
                    tf.constant(self.mass_rod2, dtype=tf.float32),
        l1, l2 = tf.constant(self.l_rod1, dtype=tf.float32), tf.constant(self.l_rod2, dtype=tf.float32)
        g = tf.constant(self.g, dtype=tf.float32)
        p, theta1, pdot, theta1dot = states[:, 0], states[:, 1], states[:, 2], states[:, 3],
        u = actions[:, 0]
        ones = tf.ones_like(p, dtype=tf.float32)
        d1 = m+m1+m2
        d2 = (0.5*m1+m2)*l1
        d3 = 0.5*m2*l2
        d4 = (1./3*m1+m2)*l1**2
        d5 = 0.5*m2*l1*l2
        d6 = 1./3*m2*l2**2
        f1 = (0.5*m1+m2)*l1*g
        f2 = 0.5*m2*l2*g
        D = tf.reshape(tf.stack([d1*ones, d2*tf.cos(theta1),
                                 d2*tf.cos(theta1), d4*ones,
                                ], axis=1),
                       shape=(-1, 2, 2))
        f = tf.reshape(tf.stack([d2*tf.sin(theta1)*tf.square(theta1dot)+u,
                                 f1*tf.sin(theta1),
                                 ], axis=1),
                       shape=(-1, 2, 1))
        D_inv = tf.linalg.inv(D)
        tmp = tf.squeeze(tf.matmul(D_inv, f), axis=-1)

        deriv = tf.concat([states[:, 2:], tmp], axis=-1)
        next_states = states + tau * deriv
        next_p, next_theta1, next_pdot, next_theta1dot = next_states[:, 0], next_states[:, 1],\
                                                         next_states[:, 2], next_states[:, 3]
        if self.if_model:
            next_theta1dot += tfd.Normal(0.05*tf.ones_like(next_p), 0.005).sample()
        next_states = tf.stack([next_p, next_theta1, next_pdot, next_theta1dot], axis=1)

        return next_states

    def compute_rewards(self, states):  # obses and actions are tensors
        with tf.name_scope('compute_reward') as scope:
            p, theta1, pdot, theta1dot = states[:, 0], states[:, 1], states[:, 2], states[:, 3]
            tip_x = p + self.l_rod1 * tf.sin(theta1)
            tip_y = self.l_rod1 * tf.cos(theta1)
            dist_penalty = 0.01 * tf.square(tip_x) + 0.04*tf.square(tip_y - 0.6)
            vel_penalty = 1e-3 * tf.square(pdot) + 1e-3 * tf.square(theta1dot)
            rewards = -dist_penalty-vel_penalty

        return rewards


class InvertedPendulumModel(object):  # all tensors
    def __init__(self, **kwargs):
        # obs: p, theta1, pdot, theta1dot
        self.dynamics = Dynamics(if_model=True)
        self.obses = None
        self.actions = None
        self.tau = 0.02
        plt.ion()

    def reset(self, obses):
        self.obses = obses

    def rollout_out(self, actions):  # action [-3,3]
        with tf.name_scope('model_step') as scope:
            self.actions = self.action_trans(actions)
            for i in range(2):
                self.obses = self.dynamics.f_xu(self.obses, self.actions, self.tau)
            rewards = self.dynamics.compute_rewards(self.obses)
        return self.obses, rewards

    def action_trans(self, actions):
        return tf.constant(100.) * actions

    def render(self, mode='human'):
        plt.cla()
        obs = self.obses.numpy()
        p, theta1, pdot, theta1dot = obs[0, 0], obs[0, 1], obs[0, 2], obs[0, 3],
        point0x, point0y = p, 0
        point1x, point1y = point0x + self.dynamics.l_rod1 * np.sin(theta1), \
                           point0y + self.dynamics.l_rod1 * np.cos(theta1)

        plt.title("Demo_model")
        ax = plt.axes(xlim=(-2.5, 2.5),
                      ylim=(-2.5, 2.5))
        ax.add_patch(plt.Rectangle((-2.5, -2.5),
                                   5, 5, edgecolor='black',
                                   facecolor='none'))
        ax.axis('equal')
        plt.axis('off')
        ax.plot([-2.5, 2.5], [0, 0], 'k')
        ax.plot([-1, -1], [-2.5, 2.5], 'k')
        ax.plot([1, 1], [-2.5, 2.5], 'k')
        ax.plot(point0x, point0y, 'b.')
        ax.plot([point0x, point1x], [point0y, point1y], color='b')
        ax.plot(point1x, point1y, 'y.')
        text_x, text_y_start = -4, 2
        ge = iter(range(0, 1000, 1))
        scale = 0.3
        plt.text(text_x, text_y_start - scale*next(ge), 'position: {:.2f}'.format(p))
        plt.text(text_x, text_y_start - scale*next(ge), r'theta1: {:.2f}rad (${:.2f}\degree$)'.format(theta1, theta1*180/np.pi))
        plt.text(text_x, text_y_start - scale*next(ge), 'theta1dot: {:.2f}rad/s'.format(theta1dot))
        if self.actions is not None:
            actions = self.actions.numpy()
            plt.text(text_x, text_y_start - scale*next(ge), 'action: {:.2f}N'.format(actions[0,0]))

        plt.pause(0.001)
        plt.show()


def testModel():
    model = InvertedPendulumModel()
    # p, sintheta1, costheta1, pdot, theta1dot, frc1, frc2,
    init_obs = np.array([[0., 0., 1., 0., 0., 0.]])
    for _ in range(10):
        print('reset')
        model.reset(tf.constant(init_obs, dtype=tf.float32))
        for i in range(50):
            print('step {}'.format(i))
            actions = tf.random.normal((1,1), dtype=tf.float32)
            model.rollout_out(actions)
            model.render()


def testModel2():
    import time
    import gym
    from policy import PolicyWithQs
    from train_scripts.train_script4mujoco import built_AMPC_parser
    args = built_AMPC_parser()
    env = gym.make('InvertedPendulum-v2')
    model = InvertedPendulumModel()
    args.obs_dim, args.act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    # policy = PolicyWithQs(**vars(args))
    # policy.load_weights()
    for _ in range(100):
        print('reset')
        env_obs = env.reset()
        done = 0
        model_obs = np.array([env_obs], dtype=np.float32)
        model.reset(model_obs)
        while not done:
            actions = tf.random.normal((1, 1), dtype=tf.float32)
            model_actions = actions
            # actions, _ = policy.compute_action(np.array([env_obs], dtype=np.float32))
            # model_actions, _ = policy.compute_action(model_obs)
            env_obs, env_rew, done, _ = env.step(actions.numpy()[0])
            env.render()
            model_obs, model_rew = model.rollout_out(model_actions)
            model_obs, model_rew = model_obs.numpy()[0], model_rew.numpy()[0]

            print('env_obs', env_obs, env_rew, done)
            print('model_obs', model_obs, model_rew)
            model.render()


def testInvPen():
    import gym
    env = gym.make('InvertedPendulum-v2')
    env.reset()


if __name__ == '__main__':
    testModel2()

