import numpy as np
import gym
import ray
import logging
from preprocessor import Preprocessor
from mixed_pg_learner import judge_is_nan

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# logger.setLevel(logging.INFO)


class OnPolicyWorker(object):
    """
    Act as both actor and learner
    """
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], 'GPU')

    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_virtual_device_configuration(
    #     gpus[0],
    #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

    def __init__(self, policy_cls, learner_cls, env_id, args):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        self.args = args
        self.env = gym.make(env_id)
        obs_space, act_space = self.env.observation_space, self.env.action_space
        self.learner = learner_cls(policy_cls, self.args)
        self.policy_with_value = policy_cls(obs_space, act_space, self.args)
        self.sample_batch_size = self.args.sample_batch_size
        self.obs = self.env.reset()
        # judge_is_nan([self.obs])
        self.done = False
        self.preprocessor = Preprocessor(obs_space, self.args.obs_normalize, self.args.reward_preprocess_type,
                                         self.args.reward_scale_factor, gamma=self.args.gamma)

        self.stats = {}
        logger.info('Worker initialized')

    def get_stats(self):
        return self.stats

    def save_weights(self, save_dir, iteration):
        self.policy_with_value.save_weights(save_dir, iteration)

    def load_weights(self, load_dir, iteration):
        self.policy_with_value.load_weights(load_dir, iteration)

    def get_weights(self):
        return self.policy_with_value.get_weights()

    def set_weights(self, weights):
        return self.policy_with_value.set_weights(weights)

    def apply_gradients(self, iteration, grads):
        self.policy_with_value.apply_gradients(iteration, grads)

    def get_ppc_params(self):
        return self.preprocessor.get_params()

    def set_ppc_params(self, params):
        self.preprocessor.set_params(params)

    def save_ppc_params(self, save_dir):
        self.preprocessor.save_params(save_dir)

    def load_ppc_params(self, load_dir):
        self.preprocessor.load_params(load_dir)

    def sample(self):
        batch_data = []
        for _ in range(self.sample_batch_size):
            processed_obs = self.preprocessor.process_obs(self.obs)
            # judge_is_nan([processed_obs])

            action, neglogp = self.policy_with_value.compute_action(processed_obs[np.newaxis, :])  # TODO
            # judge_is_nan([action])
            # judge_is_nan([neglogp])
            # print(action[0].numpy())
            obs_tp1, reward, self.done, info = self.env.step(action[0].numpy())
            processed_rew = self.preprocessor.process_rew(reward, self.done)
            # judge_is_nan([obs_tp1])
            l = info.get('l')
            # if l: print(l)

            batch_data.append((self.obs, action[0].numpy(), reward, obs_tp1, self.done, neglogp[0].numpy()))
            self.obs = self.env.reset() if self.done else obs_tp1.copy()
            # judge_is_nan([self.obs])

        return batch_data

    def put_data_into_learner(self, batch_data):
        self.learner.set_ppc_params(self.get_ppc_params())
        self.learner.get_batch_data(batch_data)

    def compute_gradient_over_ith_minibatch(self, i):
        self.learner.set_weights(self.get_weights())
        grad = self.learner.compute_gradient_over_ith_minibatch(i)
        learner_stats = self.learner.get_stats()
        self.stats.update(dict(learner_stats=learner_stats))
        return grad


class OffPolicyWorker(object):
    """just for sample"""

    def __init__(self, policy_cls, env_id, args):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        self.args = args
        self.env = gym.make(env_id)
        obs_space, act_space = self.env.observation_space, self.env.action_space
        self.policy_with_value = policy_cls(obs_space, act_space, self.args)
        self.sample_batch_size = self.args.sample_batch_size
        self.obs = self.env.reset()
        self.done = False
        self.preprocessor = Preprocessor(obs_space, self.args.obs_normalize, self.args.reward_preprocess_type,
                                         self.args.reward_scale_factor, gamma=self.args.gamma)
        self.obs = self.preprocessor.process_obs(self.obs)

        self.iteration = 0
        logger.info('Worker initialized')

    def save_weights(self, save_dir, iteration):
        self.policy_with_value.save_weights(save_dir, iteration)

    def load_weights(self, load_dir, iteration):
        self.policy_with_value.load_weights(load_dir, iteration)

    def get_weights(self):
        return self.policy_with_value.get_weights()

    def set_weights(self, weights):
        return self.policy_with_value.set_weights(weights)

    def apply_gradients(self, iteration, grads):
        self.iteration = iteration
        self.policy_with_value.apply_gradients(iteration, grads)

    def get_ppc_params(self):
        return self.preprocessor.get_params()

    def set_ppc_params(self, params):
        self.preprocessor.set_params(params)

    def save_ppc_params(self, save_dir):
        self.preprocessor.save_params(save_dir)

    def load_ppc_params(self, load_dir):
        self.preprocessor.load_params(load_dir)

    def sample(self):
        batch_data = []
        for _ in range(self.sample_batch_size):
            action, neglogp = self.policy_with_value.compute_action(self.obs[np.newaxis, :])
            obs_tp1, reward, self.done, info = self.env.step(action[0].numpy())

            reward = self.preprocessor.process_rew(reward, self.done)

            batch_data.append((self.obs, action[0].numpy(), reward, obs_tp1, self.done))
            obs = self.env.reset() if self.done else obs_tp1.copy()

            self.obs = self.preprocessor.process_obs(obs)

        return batch_data

    def sample_with_count(self):
        batch_data = self.sample()
        return batch_data, len(batch_data)


def test_worker():
    from policy import PolicyWithValue
    from trainer import built_parser
    ray.init()
    args = built_parser()
    # worker1 = Worker(PolicyWithValue, 'CartPole-v1', args)
    # print(worker1.get_weights())
    worker1 = ray.remote(num_cpus=1)(Worker).remote(PolicyWithValue, 'CartPole-v1', args)
    samples = ray.get(worker1.sample.remote())
    ray.wait([worker1.put_data_into_learner.remote(samples)])
    print(ray.get(worker1.compute_gradient_over_ith_minibatch.remote(0)))


if __name__ == '__main__':
    test_worker()
