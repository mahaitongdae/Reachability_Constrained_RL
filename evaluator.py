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
        self.env = gym.make(env_id, num_agent=self.args.num_agent)
        self.policy_with_value = policy_cls(self.env.observation_space, self.env.action_space, self.args)
        self.iteration = 0
        self.log_dir = self.args.log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.preprocessor = Preprocessor(self.env.observation_space, self.args.obs_preprocess_type, self.args.reward_preprocess_type,
                                         self.args.obs_scale_factor, self.args.reward_scale_factor,
                                         gamma=self.args.gamma, num_agent=self.args.num_agent)

        self.writer = self.tf.summary.create_file_writer(self.log_dir + '/evaluator')
        self.stats = {}

    def run_an_episode(self):
        reward_list = []
        done = 0
        obs = self.env.reset()
        for _ in range(200):
            processed_obs = self.preprocessor.tf_process_obses(obs)
            action, neglogp = self.policy_with_value.compute_action(processed_obs)
            obs, reward, done, info = self.env.step(action.numpy())
            self.env.render()
            reward_list.append(reward[0])
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


if __name__ == '__main__':
    pass
