import numpy as np
import tensorflow as tf


class EmBrakeModel(object):
    def __init__(self):
        self.constraints_num = 1

    def rollout_out(self, actions):
        with tf.name_scope('model_step') as scope:
            self.actions = self._action_transformation_for_end2end(actions)
            rewards, constraints = self.compute_rewards(self.obses, self.actions)
            self.obses = self.f_xu(self.obses, self.actions)
            # self.reward_info.update({'final_rew': rewards.numpy()[0]})

            return self.obses, rewards, constraints

    def compute_rewards(self, obses, actions):
        # rewards = -0.01 * tf.square(actions[:, 0]) # tf.square(obses[:, 0]) + tf.square(obses[:, 1]) +
        rewards = -0.01 * tf.square(obses[:, 1] - 5.0)
        constraints = -obses[:, 0]
        return rewards, constraints

    def _action_transformation_for_end2end(self, actions):
        clipped_actions = tf.clip_by_value(actions, -1.05, 1.05)
        acc = 5.0 * clipped_actions
        return acc

    def f_xu(self, x, u, frequency=10.0):
        d, v = tf.cast(x[:, 0], dtype=tf.float32), tf.cast(x[:, 1], dtype=tf.float32)
        a = tf.cast(u[:, 0], dtype=tf.float32)
        frequency = tf.convert_to_tensor(frequency)
        next_state = [d - 1 / frequency * v, v + 1 / frequency * a]
        return tf.stack(next_state, 1)

    def reset(self, obses):  # input are all tensors
        self.obses = obses
        self.actions = None
        self.reward_info = None

class PendulumModel(object):
    def __init__(self):
        self.constraints_num = 1

    def rollout_out(self, actions):
        with tf.name_scope('model_step') as scope:
            self.actions = self._action_transformation_for_end2end(actions)
            rewards, constraints = self.compute_rewards(self.obses, self.actions)
            self.obses = self.f_xu(self.obses, self.actions)
            # self.reward_info.update({'final_rew': rewards.numpy()[0]})

            return self.obses, rewards, constraints

    def compute_rewards(self, obses, actions):
        # rewards = -0.01 * tf.square(actions[:, 0]) # tf.square(obses[:, 0]) + tf.square(obses[:, 1]) +
        rewards = -0.01 * tf.square(obses[:, 1] - 5.0)
        constraints = -obses[:, 0]
        return rewards, constraints

    def _action_transformation_for_end2end(self, actions):
        clipped_actions = tf.clip_by_value(actions, -1.05, 1.05)
        acc = 5.0 * clipped_actions
        return acc

    def f_xu(self, x, u, frequency=10.0):
        d, v = tf.cast(x[:, 0], dtype=tf.float32), tf.cast(x[:, 1], dtype=tf.float32)
        a = tf.cast(u[:, 0], dtype=tf.float32)
        frequency = tf.convert_to_tensor(frequency)
        next_state = [d - 1 / frequency * v, v + 1 / frequency * a]
        return tf.stack(next_state, 1)

    def reset(self, obses):  # input are all tensors
        self.obses = obses
        self.actions = None
        self.reward_info = None

def try_pendulum_env():
    import gym
    env = gym.make('Pendulum-v0')
    env.reset()
    action = np.array([0.])

    for i in range(10):
        obs, reward, done, info = env.step(action)
        env.render()
        print(obs)

if __name__ == '__main__':
    try_pendulum_env()

