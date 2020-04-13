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
    def __init__(self, local_worker, evaluator, args):
        self.args = args
        self.evaluator = evaluator
        self.local_worker = local_worker
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
            batch_data = self.local_worker.sample()
            self.local_worker.put_data_into_learner(batch_data, [])
        with self.optimizing_timer:
            worker_stats = []
            for i in range(self.args.epoch):
                stats_list_per_epoch = []
                for minibatch_index in range(int(self.args.sample_n_step * self.args.num_agent / self.args.mini_batch_size)):
                    self.local_worker.compute_gradient_over_ith_minibatch(minibatch_index)
                    stats_list_per_epoch.append(self.local_worker.get_stats())
                    # self.local_worker.apply_gradients(self.num_updated_steps, minibatch_grads)
                worker_stats.append(stats_list_per_epoch)

        self.stats.update({'worker_stats': worker_stats,
                           'sampling_time': self.sampling_timer.mean,
                           'optimizing_time': self.optimizing_timer.mean})

        if self.num_updated_steps % self.args.log_interval == 0:
            # mbvals = []
            # mbwlists = []
            # vals_name = ['eplenmean', 'eprewmean', 'value_mean', 'q_loss', 'policy_gradient_norm',
            #              'q_gradient_norm', 'pg_time', 'q_timer']
            # lists_name = ['w_var_list', 'w_heur_bias_list', 'w_list', 'w_q_list']
            logger.info('sampling time: {}, optimizing time: {}'.format(self.stats['sampling_time'],
                                                                        self.stats['optimizing_time']))
            logger.info(pprint.pformat(self.stats['worker_stats'][0][0]['learner_stats']))
            # for i in range(self.args.epoch):
            #     for j in range(int(self.args.sample_n_step * self.args.num_agent/self.args.mini_batch_size)):
            #         learner_stats_ij = self.stats['worker_stats'][i][j]['learner_stats']
            #         mbvals.append([learner_stats_ij[val_name] for val_name in vals_name])
            #         mbwlists.append([np.array(learner_stats_ij[list_name]) for list_name in lists_name])
            # print(mbwlists)
            # vals = np.mean(mbvals, axis=0)
            # lists = np.mean(mbwlists, axis=0)
            # print(lists)
            # with self.writer.as_default():
            #     for val_name, val in zip(vals_name, vals):
            #         tf.summary.scalar('optimizer/{}'.format(val_name), val, step=self.num_updated_steps)
            #     for list_name, l in zip(lists_name, lists):
            #         tmps = [[ind] * int(1000 * p) for ind, p in enumerate(l)]
            #         hist = []
            #         for tmp in tmps:
            #             hist.extend(tmp)
            #         hist = tf.convert_to_tensor(hist)
            #         tf.summary.histogram('optimizer/{}'.format(list_name), hist, step=self.num_updated_steps)
            #     self.writer.flush()

        if self.num_updated_steps % self.args.eval_interval == 0:
            self.evaluator.set_weights(self.local_worker.get_weights())
            # self.evaluator.set_ppc_params(self.local_worker.get_ppc_params())
            self.evaluator.run_evaluation(self.num_updated_steps)
        self.num_sampled_steps += self.args.sample_n_step * self.args.num_agent
        self.num_updated_steps += 1
