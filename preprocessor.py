import numpy as np
import tensorflow as tf


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

    def set_params(self, mean, var, count):
        self.mean = mean
        self.var = var
        self.count = count

    def get_params(self, ):
        return self.mean, self.var, self.count


class Preprocessor(object):
    def __init__(self, ob_space, ob=True, rew_ptype='normalize', factor=None, clipob=10., cliprew=10., gamma=0.99,
                 epsilon=1e-8):
        self.ob_rms = RunningMeanStd(shape=ob_space.shape) if ob else None
        self.rew_ptype = rew_ptype
        self.ret_rms = RunningMeanStd(shape=()) if self.rew_ptype == 'normalize' else None
        self.factor = factor if self.rew_ptype == 'scale' else None

        self.clipob = clipob
        self.cliprew = cliprew

        self.gamma = gamma
        self.epsilon = epsilon
        self.ret = 0.

    def process_rew(self, rew, done):
        if self.rew_ptype == 'normalize':
            self.ret = self.ret * self.gamma + rew
            self.ret_rms.update(np.array([self.ret]))
            rew = np.clip(rew / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
            if done:
                self.ret = 0.

            return rew
        elif self.rew_ptype == 'scale':
            return rew / self.factor
        else:
            return rew

    def _obfilt(self, obs):
        if self.ob_rms:
            self.ob_rms.update(np.array([obs]))
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def process_obs(self, obs):
        return self._obfilt(obs)

    def np_process_obses(self, obses):
        if self.ob_rms:
            obses = np.clip((obses - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob,
                                      self.clipob)
            return obses
        else:
            return obses

    def np_process_rewards(self, rewards):
        if self.rew_ptype == 'normalize':
            rewards = np.clip(rewards / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
            return rewards
        elif self.rew_ptype == 'scale':
            return rewards / self.factor
        else:
            return rewards

    def tf_process_obses(self, obses):
        obses = tf.convert_to_tensor(obses)
        if self.ob_rms:
            obses = tf.clip_by_value((obses - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob,
                                      self.clipob)
            return obses
        else:
            return obses

    def tf_process_rewards(self, rewards):
        rewards = tf.convert_to_tensor(rewards)
        if self.rew_ptype == 'normalize':
            rewards = tf.clip_by_value(rewards / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
            return rewards
        elif self.rew_ptype == 'scale':
            return rewards / self.factor
        else:
            return rewards

    def set_params(self, params):
        if self.ob_rms:
            self.ob_rms.set_params(*params['ob_rms'])
        if self.ret_rms:
            self.ret_rms.set_params(*params['ret_rms'])

    def get_params(self):
        tmp = {}
        if self.ob_rms:
            tmp.update({'ob_rms': self.ob_rms.get_params()})
        if self.ret_rms:
            tmp.update({'ret_rms': self.ret_rms.get_params()})

        return tmp

    def save_params(self, save_dir):
        np.save(save_dir + '/ppc_params.npy', self.get_params())

    def load_params(self, load_dir):
        params = np.load(load_dir + '/ppc_params.npy', allow_pickle=True)
        params = params.item()
        self.set_params(params)
