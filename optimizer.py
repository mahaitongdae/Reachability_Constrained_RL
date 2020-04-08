import ray
import numpy as np
import tensorflow as tf
import os
import logging
import threading
import queue
import pprint
from utils.task_pool import TaskPool
import random
from utils.misc import safemean
from mixed_pg_learner import TimerStat

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AllReduceOptimizer(object):
    def __init__(self, workers, evaluator, args):
        self.args = args
        self.evaluator = evaluator
        self.workers = workers
        self.local_worker = self.workers['local_worker']
        self.sync_remote_workers()
        self.num_sampled_steps = 0
        self.num_updated_steps = 0
        self.log_dir = self.args.log_dir
        self.model_dir = self.args.model_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.writer = tf.summary.create_file_writer(self.log_dir + '/optimizer')
        self.stats = {}
        self.sampling_timer = TimerStat()
        self.optimizing_timer = TimerStat()

        logger.info('Optimizer initialized')

    def get_stats(self):
        return self.stats

    def step(self):
        self.stats.update({'iteration': self.num_updated_steps,
                           'num_samples': self.num_sampled_steps})
        logger.info('begin the {}-th optimizing step'.format(self.num_updated_steps))
        logger.info('sampling {} in total'.format(self.num_sampled_steps))
        with self.sampling_timer:
            for worker in self.workers['remote_workers']:
                batch_data, epinfos = ray.get(worker.sample.remote())
                worker.put_data_into_learner.remote(batch_data, epinfos)
        with self.optimizing_timer:
            worker_stats = []
            for i in range(self.args.epoch):
                stats_list_per_epoch = []
                for minibatch_index in range(int(self.args.sample_batch_size / self.args.mini_batch_size)):
                    minibatch_grad_futures = [worker.compute_gradient_over_ith_minibatch.remote(minibatch_index)
                                              for worker in self.workers['remote_workers']]
                    minibatch_grads = ray.get(minibatch_grad_futures)
                    stats_list_per_epoch.append(ray.get(self.workers['remote_workers'][0].get_stats.remote()))
                    final_grads = np.array(minibatch_grads).mean(axis=0)
                    self.workers['local_worker'].apply_gradients(self.num_updated_steps, final_grads)
                    self.sync_remote_workers()
                worker_stats.append(stats_list_per_epoch)

        self.stats.update({'worker_stats': worker_stats,
                           'sampling_time': self.sampling_timer.mean,
                           'optimizing_time': self.optimizing_timer.mean})

        if self.num_updated_steps % self.args.log_interval == 0:
            mbvals = []
            mbwlists = []
            vals_name = ['eplenmean', 'eprewmean', 'value_mean', 'q_loss', 'policy_gradient_norm',
                         'q_gradient_norm', 'pg_time', 'q_timer']
            lists_name = ['w_var_list', 'w_heur_bias_list', 'w_list', 'w_q_list']
            logger.info('sampling time: {}, optimizing time: {}'.format(self.stats['sampling_time'],
                                                                        self.stats['optimizing_time']))
            logger.info(pprint.pformat(self.stats['worker_stats'][0][0]['learner_stats']))
            for i in range(self.args.epoch):
                for j in range(int(self.args.sample_batch_size/self.args.mini_batch_size)):
                    learner_stats_ij = self.stats['worker_stats'][i][j]['learner_stats']
                    mbvals.append([learner_stats_ij[val_name] for val_name in vals_name])
                    mbwlists.append([np.array(learner_stats_ij[list_name]) for list_name in lists_name])
            print(mbwlists)
            vals = np.mean(mbvals, axis=0)
            lists = np.mean(mbwlists, axis=0)
            print(lists)
            with self.writer.as_default():
                for val_name, val in zip(vals_name, vals):
                    tf.summary.scalar('optimizer/{}'.format(val_name), val, step=self.num_updated_steps)
                for list_name, l in zip(lists_name, lists):
                    tmps = [[ind] * int(1000 * p) for ind, p in enumerate(l)]
                    hist = []
                    for tmp in tmps:
                        hist.extend(tmp)
                    hist = tf.convert_to_tensor(hist)
                    tf.summary.histogram('optimizer/{}'.format(list_name), hist, step=self.num_updated_steps)
                self.writer.flush()

        if self.num_updated_steps % self.args.eval_interval == 0:
            self.evaluator.set_weights.remote(self.workers['local_worker'].get_weights())
            self.evaluator.set_ppc_params.remote(self.workers['remote_workers'][0].get_ppc_params.remote())
            self.evaluator.run_evaluation.remote(self.num_updated_steps)
        if self.num_updated_steps % self.args.save_interval == 0:
            self.workers['local_worker'].save_weights(self.model_dir, self.num_updated_steps)
            self.workers['remote_workers'][0].save_ppc_params.remote(self.args.model_dir)
        self.num_sampled_steps += self.args.sample_batch_size * len(self.workers['remote_workers'])
        self.num_updated_steps += 1

    def sync_remote_workers(self):
        weights = ray.put(self.workers['local_worker'].get_weights())
        for e in self.workers['remote_workers']:
            e.set_weights.remote(weights)

    def stop(self):
        for r in self.workers['remote_workers']:
            r.__ray_terminate__.remote()
        self.evaluator.__ray_terminate__.remote()
