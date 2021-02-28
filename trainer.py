#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/8/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: trainer.py
# =====================================

import logging

import ray

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Trainer(object):
    def __init__(self, policy_cls, worker_cls, learner_cls, buffer_cls, optimizer_cls, evaluator_cls, args):
        self.args = args
        if self.args.optimizer_type.startswith('SingleProcess'):
            self.evaluator = evaluator_cls(policy_cls, self.args.env_id, self.args)
            if self.args.off_policy:
                self.local_worker = worker_cls(policy_cls, self.args.env_id, self.args, 0)
                self.buffer = buffer_cls(self.args, 0)
                self.learner = learner_cls(policy_cls, args)
                self.optimizer = optimizer_cls(self.local_worker, self.learner, self.buffer, self.evaluator, self.args)
            else:
                self.local_worker = worker_cls(policy_cls, learner_cls, self.args.env_id, self.args, 0)
                self.optimizer = optimizer_cls(self.local_worker, self.evaluator, self.args)

        else:
            self.evaluator = ray.remote(num_cpus=1)(evaluator_cls).remote(policy_cls, self.args.env_id, self.args)
            if self.args.off_policy:
                self.local_worker = worker_cls(policy_cls, self.args.env_id, self.args, 0)
                self.remote_workers = [
                    ray.remote(num_cpus=1)(worker_cls).remote(policy_cls, self.args.env_id, self.args, i + 1)
                    for i in range(self.args.num_workers)]
                self.workers = dict(local_worker=self.local_worker,
                                    remote_workers=self.remote_workers)
                self.buffers = [ray.remote(num_cpus=1)(buffer_cls).remote(self.args, i+1)
                                for i in range(self.args.num_buffers)]
                self.learners = [ray.remote(num_cpus=1)(learner_cls).remote(policy_cls, args)
                                 for _ in range(self.args.num_learners)]
                self.optimizer = optimizer_cls(self.workers, self.learners, self.buffers, self.evaluator, self.args)
            else:
                self.local_worker = worker_cls(policy_cls, learner_cls, self.args.env_id, self.args, 0)
                self.remote_workers = [
                    ray.remote(num_cpus=1)(worker_cls).remote(policy_cls, learner_cls, self.args.env_id, self.args, i+1)
                    for i in range(self.args.num_workers)]
                self.workers = dict(local_worker=self.local_worker,
                                    remote_workers=self.remote_workers)
                self.optimizer = optimizer_cls(self.workers, self.evaluator, self.args)

    def load_weights(self, load_dir, iteration):
        if self.args.optimizer_type.startswith('SingleProcess'):
            self.local_worker.load_weights(load_dir, iteration)
        else:
            self.local_worker.load_weights(load_dir, iteration)
            self.sync_remote_workers()

    def load_ppc_params(self, load_dir):
        if self.args.optimizer_type.startswith('SingleProcess'):
            self.local_worker.load_ppc_params(load_dir)
        else:
            self.local_worker.load_ppc_params(load_dir)
            for remote_worker in self.remote_workers:
                remote_worker.load_ppc_params.remote(load_dir)

    def sync_remote_workers(self):
        weights = ray.put(self.local_worker.get_weights())
        for e in self.workers['remote_workers']:
            e.set_weights.remote(weights)

    def train(self):
        logger.info('training beginning')
        while self.optimizer.num_sampled_steps < self.args.max_sampled_steps \
                or self.optimizer.iteration < self.args.max_iter:
            self.optimizer.step()
        self.optimizer.stop()
