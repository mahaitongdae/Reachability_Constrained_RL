#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/8/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: optimizer.py
# =====================================

import logging
import os
import queue
import random
import threading

import ray
import tensorflow as tf

from utils.misc import random_choice_with_index
from utils.task_pool import TaskPool
from utils.misc import judge_is_nan

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class UpdateThread(threading.Thread):
    """Background thread that updates the local model from gradient list.
    """

    def __init__(self, workers, evaluator, args, optimizer_stats):
        threading.Thread.__init__(self)
        self.args = args
        self.workers = workers
        self.local_worker = workers['local_worker']
        self.evaluator = evaluator
        self.optimizer_stats = optimizer_stats
        self.inqueue = queue.Queue(maxsize=self.args.grads_queue_size)
        self.stopped = False
        self.log_dir = self.args.log_dir
        self.model_dir = self.args.model_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.iteration = 0
        self.writer = tf.summary.create_file_writer(self.log_dir + '/optimizer')

    def run(self):
        while not self.stopped:
            self.step()

    def step(self):
        if not self.inqueue.empty():
            self.optimizer_stats.update({'update_queue_size': self.inqueue.qsize()})

            # updating
            grads, learner_stats = self.inqueue.get()
            try:
                judge_is_nan(grads)
            except ValueError:
                grads = [tf.zeros_like(grad) for grad in grads]
                logger.info('Grad is nan!, zero it')
            self.local_worker.apply_gradients(self.iteration, grads)

            if self.iteration % self.args.log_interval == 0:
                if self.iteration % (100*self.args.log_interval) == 0:
                    logger.info('updating {} in total'.format(self.iteration))
                    logger.info('sampling {} in total'.format(self.optimizer_stats['num_sampled_steps']))
                with self.writer.as_default():
                    for key, val in learner_stats.items():
                        if not isinstance(val, list):
                            tf.summary.scalar('optimizer/{}'.format(key), val, step=self.iteration)
                        else:
                            assert isinstance(val, list)
                            for i, v in enumerate(val):
                                tf.summary.scalar('optimizer/{}/{}'.format(key, i), v, step=self.iteration)
                    for key, val in self.optimizer_stats.items():
                        tf.summary.scalar('optimizer/{}'.format(key), val, step=self.iteration)
                    self.writer.flush()

            # evaluation
            if self.iteration % self.args.eval_interval == 0:
                self.evaluator.set_weights.remote(self.local_worker.get_weights())
                self.evaluator.set_ppc_params.remote(self.workers['remote_workers'][0].get_ppc_params.remote())
                self.evaluator.run_evaluation.remote(self.iteration)

            # save
            if self.iteration % self.args.save_interval == 0:
                self.local_worker.save_weights(self.model_dir, self.iteration)
                self.workers['remote_workers'][0].save_ppc_params.remote(self.model_dir)

            self.iteration += 1


class OffPolicyAsyncOptimizer(object):
    def __init__(self, workers, learners, replay_buffers, evaluator, args):
        """Initialize an off-policy async optimizers.

        Arguments:
            workers (dict): {local worker, remote workers (list)>=0}
            learners (list): list of remote learners, len >= 1
            replay_buffers (list): list of replay buffers, len >= 1
        """
        self.args = args
        self.workers = workers
        self.local_worker = self.workers['local_worker']
        self.learners = learners
        self.replay_buffers = replay_buffers
        self.evaluator = evaluator
        self.num_sampled_steps = 0
        self.num_updated_steps = 0
        self.optimizer_steps = 0
        self.stats = dict(num_sampled_steps=self.num_sampled_steps,
                          num_updated_steps=self.num_updated_steps,
                          optimizer_steps=self.optimizer_steps,
                          update_queue_size=0)
        self.update_thread = UpdateThread(self.workers, self.evaluator, self.args,
                                          self.stats)
        self.update_thread.start()
        self.max_weight_sync_delay = self.args.max_weight_sync_delay
        self.steps_since_update = {}
        self.log_dir = self.args.log_dir
        self.model_dir = self.args.model_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.sample_tasks = TaskPool()
        self._set_workers()

        # fill buffer to replay starts
        logger.info('start filling the replay')
        while not all([l >= self.args.replay_starts for l in
                       ray.get([rb.__len__.remote() for rb in self.replay_buffers])]):
            for worker, objID in list(self.sample_tasks.completed()):
                sample_batch, count = ray.get(objID)
                random.choice(self.replay_buffers).add_batch.remote(sample_batch)
                self.num_sampled_steps += count
                self.sample_tasks.add(worker, worker.sample_with_count.remote())
        logger.info('end filling the replay')

        self.learn_tasks = TaskPool()
        self._set_learners()
        logger.info('Optimizer initialized')

    def get_stats(self):
        self.stats.update(dict(num_sampled_steps=self.num_sampled_steps,
                               num_updated_steps=self.num_updated_steps,
                               optimizer_steps=self.optimizer_steps))
        return self.stats

    def _set_workers(self):
        weights = self.local_worker.get_weights()
        for worker in self.workers['remote_workers']:
            worker.set_weights.remote(weights)
            self.steps_since_update[worker] = 0
            self.sample_tasks.add(worker, worker.sample_with_count.remote())

    def _set_learners(self):
        weights = self.local_worker.get_weights()
        for learner in self.learners:
            learner.set_weights.remote(weights)
            rb, rb_index = random_choice_with_index(self.replay_buffers)
            samples = ray.get(rb.replay.remote())
            learner.get_batch_data.remote(samples[:5], rb_index, samples[-1])
            self.learn_tasks.add(learner, learner.compute_gradient.remote(self.local_worker.iteration))

    def step(self):
        assert self.update_thread.is_alive()
        assert len(self.workers['remote_workers']) > 0

        # sampling
        for worker, objID in self.sample_tasks.completed():
            sample_batch, count = ray.get(objID)
            random.choice(self.replay_buffers).add_batch.remote(sample_batch)
            self.num_sampled_steps += count
            self.steps_since_update[worker] += count
            if self.steps_since_update[worker] >= self.max_weight_sync_delay:
                judge_is_nan(self.local_worker.policy_with_value.policy.trainable_weights)
                worker.set_weights.remote(self.local_worker.get_weights())
                self.steps_since_update[worker] = 0
            self.sample_tasks.add(worker, worker.sample_with_count.remote())

        # learning
        for learner, objID in self.learn_tasks.completed():
            grads = ray.get(objID)
            info_for_buffer = ray.get(learner.get_info_for_buffer.remote())
            learner_stats = ray.get(learner.get_stats.remote())
            if self.args.buffer_type == 'priority':
                self.replay_buffers[info_for_buffer['rb_index']].update_priorities.remote(info_for_buffer['indexes'],
                                                                                          info_for_buffer['td_error'])
            rb, rb_index = random_choice_with_index(self.replay_buffers)
            samples = ray.get(rb.replay.remote())
            assert grads is not None and samples is not None
            learner.set_ppc_params.remote(self.local_worker.get_ppc_params())
            learner.get_batch_data.remote(samples[:5], rb_index, samples[-1])
            learner.set_weights.remote(self.local_worker.get_weights())
            self.learn_tasks.add(learner, learner.compute_gradient.remote(self.local_worker.iteration))
            self.update_thread.inqueue.put([grads, learner_stats])

        self.num_updated_steps = self.update_thread.iteration
        self.optimizer_steps += 1
        self.get_stats()

    def stop(self):
        for r in self.workers['remote_workers']:
            r.__ray_terminate__.remote()
        for r in self.learners:
            r.__ray_terminate__.remote()
        for r in self.replay_buffers:
            r.__ray_terminate__.remote()
        self.evaluator.__ray_terminate__.remote()
        self.update_thread.stopped = True
