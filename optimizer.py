#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/9/1
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

from utils.misc import judge_is_nan, TimerStat
from utils.misc import random_choice_with_index
from utils.task_pool import TaskPool

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

WORKER_DEPTH = 2
BUFFER_DEPTH = 4
LEARNER_QUEUE_MAX_SIZE = 32


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
        self.update_timer = TimerStat()
        self.grad_queue_get_timer = TimerStat()
        self.grad_apply_timer = TimerStat()
        self.writer = tf.summary.create_file_writer(self.log_dir + '/optimizer')

    def run(self):
        while not self.stopped:
            with self.update_timer:
                self.step()
                self.update_timer.push_units_processed(1)

    def step(self):
        self.optimizer_stats.update(dict(update_queue_size=self.inqueue.qsize(),
                                         update_time=self.update_timer.mean,
                                         update_throughput=self.update_timer.mean_throughput,
                                         grad_queue_get_time=self.grad_queue_get_timer.mean,
                                         grad_apply_timer=self.grad_apply_timer.mean
                                    ))
        # fetch grad
        with self.grad_queue_get_timer:
            grads, learner_stats = self.inqueue.get()

        # apply grad
        with self.grad_apply_timer:
            try:
                judge_is_nan(grads)
            except ValueError:
                grads = [tf.zeros_like(grad) for grad in grads]
                logger.info('Grad is nan!, zero it')

            self.local_worker.apply_gradients(self.iteration, grads)

        # log
        if self.iteration % self.args.log_interval == 0:
            logger.info('updating {} in total'.format(self.iteration))
            logger.info('sampling {} in total'.format(self.optimizer_stats['num_sampled_steps']))
            with self.writer.as_default():
                for key, val in learner_stats.items():
                    if not isinstance(val, list):
                        tf.summary.scalar('optimizer/learner_stats/scalar/{}'.format(key), val, step=self.iteration)
                    else:
                        assert isinstance(val, list)
                        for i, v in enumerate(val):
                            tf.summary.scalar('optimizer/learner_stats/list/{}/{}'.format(key, i), v, step=self.iteration)
                for key, val in self.optimizer_stats.items():
                    tf.summary.scalar('optimizer/{}'.format(key), val, step=self.iteration)
                self.writer.flush()

        # evaluate
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
        self.learner_queue = queue.Queue(LEARNER_QUEUE_MAX_SIZE)
        self.replay_buffers = replay_buffers
        self.evaluator = evaluator
        self.num_sampled_steps = 0
        self.num_updated_steps = 0
        self.num_samples_dropped = 0
        self.optimizer_steps = 0
        self.timers = {k: TimerStat() for k in ["sampling_timer", "replay_timer",
                                                "learning_timer"]}
        self.stats = {}
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

        self.replay_tasks = TaskPool()
        self._set_buffers()

        self.learn_tasks = TaskPool()
        self._set_learners()
        logger.info('Optimizer initialized')

    def get_stats(self):
        self.stats.update(dict(num_sampled_steps=self.num_sampled_steps,
                               num_updated_steps=self.num_updated_steps,
                               optimizer_steps=self.optimizer_steps,
                               num_samples_dropped=self.num_samples_dropped,
                               learner_queue_size=self.learner_queue.qsize(),
                               sampling_time=self.timers['sampling_timer'].mean,
                               replay_time=self.timers["replay_timer"].mean,
                               learning_time=self.timers['learning_timer'].mean
                               )
                          )
        return self.stats

    def _set_workers(self):
        weights = self.local_worker.get_weights()
        for worker in self.workers['remote_workers']:
            worker.set_weights.remote(weights)
            self.steps_since_update[worker] = 0
            for _ in range(WORKER_DEPTH):
                self.sample_tasks.add(worker, worker.sample_with_count.remote())

    def _set_buffers(self):
        for rb in self.replay_buffers:
            for _ in range(BUFFER_DEPTH):
                self.replay_tasks.add(rb, rb.replay.remote())

    def _set_learners(self):
        weights = self.local_worker.get_weights()
        ppc_params = self.workers['remote_workers'][0].get_ppc_params.remote()
        for learner in self.learners:
            learner.set_weights.remote(weights)
            if self.args.obs_preprocess_type == 'normalize' or \
                    self.args.reward_preprocess_type == 'normalize':
                learner.set_ppc_params.remote(ppc_params)
            rb, _ = random_choice_with_index(self.replay_buffers)
            samples = ray.get(rb.replay.remote())
            self.learn_tasks.add(learner, learner.compute_gradient.remote(samples[:5], rb, samples[-1],
                                                                          self.local_worker.iteration))

    def step(self):
        assert self.update_thread.is_alive()
        assert len(self.workers['remote_workers']) > 0
        weights = None

        # sampling
        with self.timers['sampling_timer']:
            for worker, objID in self.sample_tasks.completed():
                sample_batch, count = ray.get(objID)
                random.choice(self.replay_buffers).add_batch.remote(sample_batch)
                self.num_sampled_steps += count
                self.steps_since_update[worker] += count
                if self.steps_since_update[worker] >= self.max_weight_sync_delay:
                    judge_is_nan(self.local_worker.policy_with_value.policy.trainable_weights)
                    if weights is None:
                        weights = ray.put(self.local_worker.get_weights())
                    worker.set_weights.remote(weights)
                    self.steps_since_update[worker] = 0
                self.sample_tasks.add(worker, worker.sample_with_count.remote())

        # replay
        with self.timers["replay_timer"]:
            for rb, replay in self.replay_tasks.completed():
                self.replay_tasks.add(rb, rb.replay.remote())
                if self.learner_queue.full():
                    self.num_samples_dropped += 1
                else:
                    samples = ray.get(replay)
                    self.learner_queue.put((rb, samples))

        # learning
        with self.timers['learning_timer']:
            for learner, objID in self.learn_tasks.completed():
                grads = ray.get(objID)
                learner_stats = ray.get(learner.get_stats.remote())
                if self.args.buffer_type == 'priority':
                    info_for_buffer = ray.get(learner.get_info_for_buffer.remote())
                    info_for_buffer['rb'].update_priorities.remote(info_for_buffer['indexes'],
                                                                   info_for_buffer['td_error'])
                rb, samples = self.learner_queue.get(block=False)
                if self.args.obs_preprocess_type == 'normalize' or \
                        self.args.reward_preprocess_type == 'normalize':
                    learner.set_ppc_params.remote(self.workers['remote_workers'][0].get_ppc_params.remote())
                if weights is None:
                    weights = ray.put(self.local_worker.get_weights())
                learner.set_weights.remote(weights)
                self.learn_tasks.add(learner, learner.compute_gradient.remote(samples[:5], rb, samples[-1],
                                                                              self.local_worker.iteration))
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
