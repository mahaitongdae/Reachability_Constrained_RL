#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/8/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: trainer.py
# =====================================

import logging

import ray

from evaluator import Evaluator
from worker import OffPolicyWorker

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Trainer(object):
    def __init__(self, policy_cls, learner_cls, buffer_cls, optimizer_cls, args):
        self.args = args
        self.evaluator = ray.remote(num_cpus=1)(Evaluator).remote(policy_cls, self.args.env_id, self.args)
        self.local_worker = OffPolicyWorker(policy_cls, self.args.env_id, self.args, 0)
        self.remote_workers = [
            ray.remote(num_cpus=1)(OffPolicyWorker).remote(policy_cls, self.args.env_id, self.args, i + 1)
            for i in range(self.args.num_workers)]
        self.workers = dict(local_worker=self.local_worker,
                            remote_workers=self.remote_workers)
        self.buffers = [ray.remote(num_cpus=1)(buffer_cls).remote(self.args, i + 1)
                        for i in range(self.args.num_buffers)]
        self.learners = [ray.remote(num_cpus=1)(learner_cls).remote(policy_cls, args)
                         for _ in range(self.args.num_learners)]
        self.optimizer = optimizer_cls(self.workers, self.learners, self.buffers, self.evaluator, self.args)

    def load_weights(self, load_dir, iteration):
        self.local_worker.load_weights(load_dir, iteration)
        self.evaluator.load_weights(load_dir, iteration)

    def load_ppc_params(self, load_dir):
        self.local_worker.load_ppc_params(load_dir)
        self.evaluator.load_ppc_params(load_dir)

    def train(self):
        logger.info('training beginning')
        while self.optimizer.num_sampled_steps < self.args.max_sampled_steps \
                or self.optimizer.num_updated_steps < self.args.max_updated_steps:
            self.optimizer.step()
        self.optimizer.stop()
