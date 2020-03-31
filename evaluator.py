import gym
import numpy as np
import logging
import os
from preprocessor import Preprocessor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# logger.setLevel(logging.INFO)


class Evaluator(object):
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], 'GPU')

    def __init__(self, policy_cls, env_id, args):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        self.args = args
        self.env = gym.make(env_id)
        self.policy_with_value = policy_cls(self.env.observation_space, self.env.action_space, self.args)
        self.iteration = 0
        self.log_dir = self.args.log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.preprocessor = Preprocessor(self.env.observation_space, self.args.obs_normalize,
                                         self.args.reward_preprocess_type, gamma=self.args.gamma)

        self.writer = self.tf.summary.create_file_writer(self.log_dir)

    def run_an_episode(self):
        reward_list = []
        done = 0
        obs = self.env.reset()
        while not done:
            # print('before', obs, type(obs))
            # obs = self.preprocessor.process_obs(obs)
            # print('after', obs, type(obs))
            action, neglogp = self.policy_with_value.compute_action(obs[np.newaxis, :-1])
            # print('action', action)
            obs, reward, done, info = self.env.step(action[0].numpy())
            # print('obs', obs)
            self.env.render()
            reward_list.append(reward)
        self.env.close()
        episode_return = sum(reward_list)
        episode_len = len(reward_list)
        return episode_return, episode_len

    def run_n_episode(self, n):
        list_of_return = []
        list_of_len = []
        for _ in range(n):
            logger.info('logging {}-th episode'.format(_))
            episode_return, episode_len = self.run_an_episode()
            list_of_return.append(episode_return)
            list_of_len.append(episode_len)
        average_return = sum(list_of_return) / len(list_of_return)
        average_len = sum(list_of_len) / len(list_of_len)
        return average_return, average_len

    def set_weights(self, weights):
        self.policy_with_value.set_weights(weights)

    def set_ppc_params(self, params):
        self.preprocessor.set_params(params)

    def run_evaluation(self, iteration):
        self.iteration = iteration
        average_return, average_len = self.run_n_episode(self.args.evaluate_epi_num)
        logger.info('average return is {}, average length is {}'.format(average_return, average_len))
        with self.writer.as_default():
            self.tf.summary.scalar("evaluation/average_return", average_return, step=self.iteration)
            self.tf.summary.scalar("evaluation/average_len", average_len, step=self.iteration)
            self.writer.flush()


def test_evaluator():
    from policy import PolicyWithValue
    from algorithms.ppo.train import built_ppo_parser
    args = built_ppo_parser()
    evaluator = Evaluator(PolicyWithValue, 'CartPole-v1', args)
    evaluator.run_n_episode(100)

def test_gpu():
    from train_script import built_mixedpg_parser
    from policy import PolicyWithValue, PolicyWithQs
    import time
    args = built_mixedpg_parser()
    a = Evaluator(PolicyWithQs, 'Pendulum-v0', args)
    time.sleep(10000)


if __name__ == '__main__':
    test_gpu()

