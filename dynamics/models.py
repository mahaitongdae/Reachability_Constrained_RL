import numpy as np
import tensorflow as tf

class DynamicsModel(object):
    def __init__(self):
        self.constraints_num = 1
        self.action_range = 1.0

    def rollout_out(self, actions):
        with tf.name_scope('model_step') as scope:
            self.actions = self._action_transformation_for_end2end(actions)
            rewards, constraints = self.compute_rewards(self.obses, self.actions)
            self.obses = self.f_xu(self.obses, self.actions)

            return self.obses, rewards, constraints

    def compute_rewards(self, obses, actions):
        rewards = obses[:, 1]
        constraints = obses[:, 0]
        return rewards, constraints

    def _action_transformation_for_end2end(self, actions):
        clipped_actions = tf.clip_by_value(actions, -1.05, 1.05)
        acc = self.action_range * clipped_actions
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

class EmBrakeModel(object):
    def __init__(self):
        self.constraints_num = 1

    def rollout_out(self, actions):
        with tf.name_scope('model_step') as scope:
            self.actions = self._action_transformation_for_end2end(actions)
            rewards, constraints = self.compute_rewards(self.obses, self.actions)
            self.obses = self.f_xu(self.obses, self.actions)

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


class UpperTriangleModel(object):
    def __init__(self):
        self.constraints_num = 1
        self.action_range = tf.constant(0.5, dtype=tf.float32)
        self.frequency = tf.constant(5.0, dtype=tf.float32)

    def rollout_out(self, actions):
        with tf.name_scope('model_step') as scope:
            self.actions = self._action_transformation_for_end2end(actions)
            rewards, constraints = self.compute_rewards(self.obses, self.actions)
            self.obses = self.f_xu(self.obses, self.actions)

            new_sctr = self.compute_constraints(self.obses)
            done = new_sctr > tf.zeros_like(new_sctr)

            return self.obses, rewards, constraints, done

    def compute_rewards(self, obses, actions):
        obses = tf.cast(obses, dtype=tf.float32)
        actions = tf.cast(actions, dtype=tf.float32)
        rewards = - tf.square(obses[:, 0]) - tf.square(obses[:, 1]) - 10 * tf.square(actions[:, 0])
        constraints = self.compute_constraints(obses)
        return rewards, constraints
    
    def compute_constraints(self, obses):
        constraints = tf.reduce_max(tf.abs(obses), axis=1) - 5.
        return constraints

    def _action_transformation_for_end2end(self, actions):
        clipped_actions = tf.clip_by_value(actions, -1.05, 1.05)
        acc = 0.5 * clipped_actions
        return acc

    def f_xu(self, x, u):
        d, v = tf.cast(x[:, 0], dtype=tf.float32), tf.cast(x[:, 1], dtype=tf.float32)
        a = tf.cast(u[:, 0], dtype=tf.float32)
        next_state = [d + 1 / self.frequency * v, v + 1 / self.frequency * a]
        return tf.stack(next_state, 1)
    
    def g_x(self, x):
        '''
        for a control-affine system, f(x,u) = g(x)*u + f'(x), here we need formulate g(x)
        for simpler model-based feasibility computation
        '''
        return tf.constant([[0.], [1.]], dtype=tf.float32)

    def reset(self, obses):  # input are all tensors
        self.obses = obses
        self.actions = None
        self.reward_info = None


class PendulumModel(object):

    def __init__(self):
        self.max_speed = 8
        self.max_torque = 10.
        self.g = 10.
        self.m = 1.
        self.l = 1.
        self.constraints_num = 1
        self.max_th = np.pi / 3

    def rollout_out(self, actions):
        with tf.name_scope('model_step') as scope:
            self.actions = self._action_transformation_for_end2end(actions)
            rewards, constraints = self.compute_rewards(self.obses, self.actions)
            self.obses = self.f_xu(self.obses, self.actions)

            return self.obses, rewards, constraints

    def compute_rewards(self, obses, actions):
        rewards = -0.1 * tf.square(actions)
        constraints = tf.stack([- obses[:, 0] - self.max_th, - self.max_th + obses[:, 0]], axis=1)
        return rewards, constraints

    def _action_transformation_for_end2end(self, actions):
        clipped_actions = tf.clip_by_value(actions, -1.05, 1.05)
        torque = self.max_torque * clipped_actions
        return torque

    def f_xu(self, x, u, frequency=20.0):
        g = self.g
        m = self.m
        l = self.l
        frequency = tf.convert_to_tensor(frequency)
        dt = 1 / frequency

        th, thdot = tf.cast(x[:, 0], dtype=tf.float32), tf.cast(x[:, 1], dtype=tf.float32)
        u = tf.cast(u[:, 0], dtype=tf.float32)

        newthdot = thdot + (-3 * g / (2 * l) * tf.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = tf.clip_by_value(newthdot, -self.max_speed, self.max_speed)

        next_state = [self.angle_normalize(newth), newthdot]
        return tf.stack(next_state, 1)

    def reset(self, obses):  # input are all tensors
        self.obses = obses
        self.actions = None
        self.reward_info = None

    def angle_normalize(self, x):
        return (((x + np.pi) % (2 * np.pi)) - np.pi)


class Air3dModel(DynamicsModel):
    def __init__(self):
        super(Air3dModel, self).__init__()
        self.evader_spd = 5.
        self.pursuer_spd = 5.
        self.pursuer_turn_rate = 1.

    def f_xu(self, x, u, frequency=10.0):
        px, py, phi = tf.cast(x[:, 0], dtype=tf.float32), tf.cast(x[:, 1], dtype=tf.float32), tf.cast(x[:, 2], dtype=tf.float32)
        u = tf.cast(u[:, 0], dtype=tf.float32)
        dx = -self.evader_spd + self.pursuer_spd * tf.cos(phi) + tf.multiply(py, u)
        dy = self.pursuer_spd * tf.sin(phi) - tf.multiply(px, u)
        dphi = self.pursuer_turn_rate - u
        new_phi = (phi + 1 / frequency * dphi) % (2 * np.pi)
        return tf.stack([px + 1 / frequency * dx, py + 1 / frequency * dy, new_phi], axis=1)

    def compute_rewards(self, obses, actions):
        obses = tf.cast(obses, dtype=tf.float32)
        actions = tf.cast(actions, dtype=tf.float32)
        rewards = - tf.square(actions[:, 0])
        d = tf.square(obses[:, 0]) + tf.square(obses[:, 1])
        constraints = tf.stack([25. - d], axis=1)
        return rewards, constraints

    def check_done(self, obses):
        pass # todo: add done in model based?


def try_pendulum_env():
    import gym
    import time
    env = gym.make('Pendulum-v1')
    env.reset()
    action = np.array([0.])

    for i in range(100):
        obs, reward, done, info = env.step(action)
        env.render()
        time.sleep(1)
        print(obs)

def try_pendulum_model():
    model = PendulumModel()
    model.reset(np.array([[0.,1.],[0.,1.],[0.,1.]]))
    model.rollout_out([[0.,0.,0.]])

def try_up_model():
    model = UpperTriangleModel()
    model.reset(np.array([[0.,1.],[0.,1.],[0.,1.]]))
    obses, rewards, cstrs, dones = model.rollout_out([[0.],[0.5],[1.]])
    print(obses, rewards, cstrs, dones)

def try_air3d_model():
    model = Air3dModel()
    model.reset(np.array([[0., 1., 0.],[0., 1., 0.],[0., 1., 0.]]))
    model.rollout_out([[0.],[0.],[0.]])


if __name__ == '__main__':
    try_up_model()

