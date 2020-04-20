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
        self.env = gym.make(env_id, num_agent=self.args.num_agent, num_future_data=self.args.num_future_data)
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

    def load_weights(self, load_dir, iteration):
        self.policy_with_value.load_weights(load_dir, iteration)

    def load_ppc_params(self, load_dir):
        self.preprocessor.load_params(load_dir)

    def evaluate_saved_model(self, model_load_dir, ppc_params_load_dir, iteration):
        self.load_weights(model_load_dir, iteration)
        self.load_ppc_params(ppc_params_load_dir)

    def metrics(self, forward_steps, render=True, reset=False):
        # obs: v_xs, v_ys, rs, delta_ys, delta_phis, steers, a_xs, future_delta_ys1,..., future_delta_ysn,
        #      future_delta_phis1,..., future_delta_phisn

        delta_ys_list = []
        delta_phis_list = []
        rewards_list = []
        obs = self.env.reset()
        for _ in range(forward_steps):
            delta_ys_list.append(obs[:, 3])
            delta_phis_list.append(obs[:, 4])
            processed_obs = self.preprocessor.tf_process_obses(obs)
            action, neglogp = self.policy_with_value.compute_action(processed_obs)
            obs, reward, done, info = self.env.step(action.numpy())
            if render:
                self.env.render()
            if reset:
                self.env.reset()
            rewards_list.append(reward)
        self.env.close()
        delta_y_metric = np.sqrt(np.mean(np.square(np.array(delta_ys_list))))
        delta_phis_metric = np.sqrt(np.mean(np.square(np.array(delta_phis_list))))
        rewards_mean = np.mean(np.array(rewards_list))

        return delta_y_metric, delta_phis_metric, rewards_mean

    def set_weights(self, weights):
        self.policy_with_value.set_weights(weights)

    def set_ppc_params(self, params):
        self.preprocessor.set_params(params)

    def run_evaluation(self, iteration):
        self.iteration = iteration
        delta_y_metric, delta_phis_metric, rewards_mean = self.metrics(100, render=True, reset=False)
        logger.info('delta_y_metric is {}, delta_phis_metric is {}, rewards_mean is {}'.format(delta_y_metric,
                                                                                               delta_phis_metric,
                                                                                               rewards_mean))
        with self.writer.as_default():
            self.tf.summary.scalar("evaluation/delta_y_metric", delta_y_metric, step=self.iteration)
            self.tf.summary.scalar("evaluation/delta_phis_metric", delta_phis_metric, step=self.iteration)
            self.writer.flush()


def test_trained_model(model_dir, ppc_params_dir, iteration):
    from train_script import built_mixedpg_parser
    from policy import PolicyWithQs
    args = built_mixedpg_parser()
    evaluator = Evaluator(PolicyWithQs, args.env_id, args)
    evaluator.load_weights(model_dir, iteration)
    evaluator.load_ppc_params(ppc_params_dir)
    return evaluator.metrics(1000, render=False, reset=False)


if __name__ == '__main__':
    model_dir = './results/mixed_pg/experiment-2020-04-20-09-18-30/models'
    print(test_trained_model(model_dir, model_dir, 0))
