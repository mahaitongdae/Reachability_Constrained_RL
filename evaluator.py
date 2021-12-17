#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/9/1
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: evaluator.py
# =====================================

import logging
import os

import gym
import dynamics
import numpy as np

from preprocessor import Preprocessor
from utils.misc import TimerStat, args2envkwargs
from matplotlib.colors import ListedColormap

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Evaluator(object):
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    def __init__(self, policy_cls, env_id, args):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        self.args = args
        self.env = gym.make(env_id, **args2envkwargs(args))
        self.policy_with_value = policy_cls(self.args)
        self.iteration = 0
        if self.args.mode == 'training':
            self.log_dir = self.args.log_dir + '/evaluator'
        else:
            self.log_dir = self.args.test_log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.preprocessor = Preprocessor((self.args.obs_dim, ), self.args.obs_preprocess_type, self.args.reward_preprocess_type,
                                         self.args.obs_scale, self.args.reward_scale, self.args.reward_shift,
                                         gamma=self.args.gamma)

        self.writer = self.tf.summary.create_file_writer(self.log_dir)
        self.stats = {}
        self.eval_timer = TimerStat()
        self.eval_times = 0

    def get_stats(self):
        self.stats.update(dict(eval_time=self.eval_timer.mean))
        return self.stats

    def load_weights(self, load_dir, iteration):
        self.policy_with_value.load_weights(load_dir, iteration)

    def load_ppc_params(self, load_dir):
        self.preprocessor.load_params(load_dir)

    def evaluate_saved_model(self, model_load_dir, ppc_params_load_dir, iteration):
        self.load_weights(model_load_dir, iteration)
        self.load_ppc_params(ppc_params_load_dir)

    def run_an_episode(self, steps=None, render=True):
        reward_list = []
        reward_info_dict_list = []
        action_list = []
        done = 0
        obs = self.env.reset()
        if render: self.env.render()
        if steps is not None:
            for _ in range(steps):
                processed_obs = self.preprocessor.tf_process_obses(obs)
                action = self.policy_with_value.compute_mode(processed_obs[np.newaxis, :])
                obs, reward, done, info = self.env.step(action.numpy()[0])
                reward_info_dict_list.append(info['reward_info'])
                if render: self.env.render()
                reward_list.append(reward)
                action_list.append(action[0])
        else:
            while not done:
                processed_obs = self.preprocessor.tf_process_obses(obs)
                action = self.policy_with_value.compute_mode(processed_obs[np.newaxis, :])
                obs, reward, done, info = self.env.step(action.numpy()[0])
                reward_info_dict_list.append(info['reward_info'])
                if render: self.env.render()
                reward_list.append(reward)
        episode_return = sum(reward_list)
        episode_len = len(reward_list)
        info_dict = dict()
        # import matplotlib.pyplot as plt
        # plt.figure(1)
        # plt.plot(range(len(action_list)), action_list)
        # plt.show()
        for key in reward_info_dict_list[0].keys():
            info_key = list(map(lambda x: x[key], reward_info_dict_list))
            mean_key = sum(info_key) / len(info_key)
            info_dict.update({key: mean_key})
        info_dict.update(dict(episode_return=episode_return,
                              episode_len=episode_len))
        return info_dict

    def run_n_episode(self, n):
        list_of_return = []
        list_of_len = []
        list_of_info_dict = []
        for _ in range(n):
            logger.info('logging {}-th episode'.format(_))
            info_dict = self.run_an_episode(self.args.fixed_steps, self.args.eval_render)
            list_of_info_dict.append(info_dict.copy())
        n_info_dict = dict()
        for key in list_of_info_dict[0].keys():
            info_key = list(map(lambda x: x[key], list_of_info_dict))
            mean_key = sum(info_key) / len(info_key)
            n_info_dict.update({key: mean_key})
        return n_info_dict

    def set_weights(self, weights):
        self.policy_with_value.set_weights(weights)

    def set_ppc_params(self, params):
        self.preprocessor.set_params(params)

    def run_evaluation(self, iteration):
        with self.eval_timer:
            self.iteration = iteration
            n_info_dict = self.run_n_episode(self.args.num_eval_episode)
            with self.writer.as_default():
                for key, val in n_info_dict.items():
                    self.tf.summary.scalar("evaluation/{}".format(key), val, step=self.iteration)
                for key, val in self.get_stats().items():
                    self.tf.summary.scalar("evaluation/{}".format(key), val, step=self.iteration)
                self.writer.flush()
        if self.eval_times % self.args.eval_log_interval == 0:
            logger.info('Evaluator_info: {}, {}'.format(self.get_stats(),n_info_dict))
        self.eval_times += 1

    def compute_action_from_batch_obses(self, path):
        obses = np.load(path)
        preprocess_obs = self.preprocessor.np_process_obses(obses)
        action = self.policy_with_value.compute_mode(preprocess_obs)
        action_np = action.numpy()
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(range(action_np.shape[0]), action_np[:,0])
        plt.show()
        a = 1

    def static_region(self):
        d = np.linspace(0,10,100)
        v = np.linspace(0,10,100)
        cmaplist = ['springgreen'] * 3 + ['crimson'] * 87
        cmap1 = ListedColormap(cmaplist)
        D, V = np.meshgrid(d, v)
        flattenD = np.reshape(D, [-1,])
        flattenV = np.reshape(V, [-1,])
        obses = np.stack([flattenD, flattenV], 1)
        preprocess_obs = self.preprocessor.np_process_obses(obses)
        flattenMU = self.policy_with_value.compute_mu(preprocess_obs).numpy() / 100
        # flattenMU_max = np.max(flattenMU,axis=1)
        for k in range(flattenMU.shape[1]):
            flattenMU_k = flattenMU[:, k]
            mu = flattenMU_k.reshape(D.shape)
            def plot_region(z, name):
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D
                plt.figure()
                z += np.clip((V**2-6*(D-2))/50, 0, 1)
                ct = plt.contourf(D,V,z,50, cmap=cmap1)
                # plt.colorbar(ct)
                plt.grid()
                x = np.linspace(0, 8)
                t = np.sqrt(2 * 3 * x)
                plt.plot(x+2, t, linestyle='--', color='red')
                # plt.plot(d, np.sqrt(2*5*d),lw=2)
                name_2d=name + '_2d.jpg'
                plt.savefig(os.path.join(self.log_dir, name_2d))

                figure = plt.figure()
                ax = Axes3D(figure)
                ax.plot_surface(D, V, z, rstride=1, cstride=1, cmap='rainbow')
                # plt.show()
                name_3d = name + '_3d.jpg'
                plt.savefig(os.path.join(self.log_dir,name_3d))
            plot_region(mu, str(k))

def static_region(test_dir, iteration):
    import json
    import argparse
    import datetime
    from train_script import built_LMAMPC_parser
    from policy import Policy4Reach
    # args = built_LMAMPC_parser()
    params = json.loads(open(test_dir + '/config.json').read())
    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    test_log_dir = params['log_dir'] + '/tester/test-region-{}'.format(time_now)
    params.update(dict(mode='testing',
                       test_dir=test_dir,
                       test_log_dir=test_log_dir,))
    parser = argparse.ArgumentParser()
    for key, val in params.items():
        parser.add_argument("-" + key, default=val)
    args =  parser.parse_args()
    evaluator = Evaluator(Policy4Reach, args.env_id, args)
    evaluator.load_weights(os.path.join(test_dir, 'models'), iteration)
    evaluator.static_region()

def test_evaluator():
    import ray
    ray.init()
    import time
    from train_script import built_parser
    from policy import Policy4Toyota
    args = built_parser('AMPC')

    # evaluator = Evaluator(Policy4Toyota, args.env_id, args)
    # evaluator.run_evaluation(3)
    evaluator = ray.remote(num_cpus=1)(Evaluator).remote(Policy4Toyota, args.env_id, args)
    evaluator.run_evaluation.remote(3)
    time.sleep(10000)


if __name__ == '__main__':
    static_region('./results/toyota3lane/LMAMPC-v2-2021-06-24-14-10-01', 200000)
