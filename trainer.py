import ray
from worker import OnPolicyWorker
from evaluator import Evaluator

import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Trainer(object):
    def __init__(self, policy_cls, learner_cls, buffer_cls, optimizer_cls, args):
        self.args = args

        self.evaluator = ray.remote(num_cpus=1)(Evaluator).remote(policy_cls, self.args.env_id, self.args)
        self.local_worker = OnPolicyWorker(policy_cls, learner_cls, self.args.env_id, self.args, 0)
        self.remote_workers = [ray.remote(num_cpus=3)(OnPolicyWorker).remote(policy_cls,
                                                                             learner_cls,
                                                                             self.args.env_id,
                                                                             self.args,
                                                                             i+1
                                                                             )
                               for i in range(self.args.num_workers)]
        self.workers = dict(local_worker=self.local_worker,
                            remote_workers=self.remote_workers)
        self.optimizer = optimizer_cls(self.workers, self.evaluator, self.args)

    def train(self):
        logger.info('training beginning')
        while self.optimizer.num_sampled_steps < self.args.max_sampled_steps \
                or self.optimizer.num_updated_steps < self.args.max_updated_steps:
            self.optimizer.step()
        self.optimizer.stop()
