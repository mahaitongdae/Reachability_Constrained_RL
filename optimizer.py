import ray
import numpy as np
import tensorflow as tf
import os
import logging
import threading
import queue
from utils.task_pool import TaskPool
import random


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AllReduceOptimizer(object):
    def __init__(self, workers, evaluator, args):
        self.args = args
        self.evaluator = evaluator
        self.workers = workers
        self.sync_remote_workers()
        self.num_sampled_steps = 0
        self.num_updated_steps = 0
        self.log_dir = self.args.log_dir
        self.model_dir = self.args.model_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.writer = tf.summary.create_file_writer(self.log_dir)
        logger.info('Optimizer initialized')

    def step(self):
        logger.info('begin the {}-th optimizing step'.format(self.num_updated_steps))
        logger.info('sampling {} in total'.format(self.num_sampled_steps))
        for worker in self.workers['remote_workers']:
            worker.put_data_into_learner.remote(worker.sample.remote())
        for i in range(self.args.epoch):
            for minibatch_index in range(int(self.args.sample_batch_size / self.args.mini_batch_size)):
                minibatch_grad_futures = [worker.compute_gradient_over_ith_minibatch.remote(minibatch_index)
                                          for worker in self.workers['remote_workers']]
                minibatch_grads = ray.get(minibatch_grad_futures)
                final_grads = np.array(minibatch_grads).mean(axis=0)
                self.workers['local_worker'].apply_gradients(self.num_updated_steps, final_grads)
                self.sync_remote_workers()

        if self.num_updated_steps % self.args.eval_interval == 0:
            self.evaluator.set_weights.remote(self.workers['local_worker'].get_weights())
            self.evaluator.set_ppc_params.remote(self.workers['remote_workers'][0].get_ppc_params.remote())
            self.evaluator.run_evaluation.remote(self.num_updated_steps)
        if self.num_updated_steps % self.args.save_interval == 0:
            self.workers['local_worker'].save_weights(self.model_dir, self.num_updated_steps)
            # self.workers['remote_workers'][0].save_ppc_params.remote(self.args.model_dir)
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


class A3C_Optimizer(object):
    pass


class UpdateThread(threading.Thread):
    """Background thread that updates the local model from gradient list.
    """

    def __init__(self, workers, evaluator, args):
        threading.Thread.__init__(self)
        self.args = args
        self.workers = workers
        self.evaluator = evaluator
        self.inqueue = queue.Queue(maxsize=100)
        self.stopped = False
        self.iteration = 0

    def run(self):
        while not self.stopped:
            self.step()

    def step(self):
        if not self.inqueue.empty():
            # logger.info('grads queue size is {}'.format(self.inqueue.qsize()))
            # updating
            grads = self.inqueue.get()
            self.workers['local_worker'].apply_gradients(self.iteration, grads)

            # evaluation
            if self.iteration % self.args.eval_interval == 0:
                self.evaluator.set_weights.remote(self.workers['local_worker'].get_weights())
                self.evaluator.set_ppc_params.remote(self.workers['remote_workers'][0].get_ppc_params.remote())
                self.evaluator.run_evaluation.remote(self.iteration)

            # save
            if self.iteration % self.args.save_interval == 0:
                self.workers['local_worker'].save_weights(self.args.model_dir, self.iteration)
                self.workers['remote_workers'][0].save_ppc_params.remote(self.args.model_dir)

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
        self.learners = learners
        self.replay_buffers = replay_buffers
        self.evaluator = evaluator
        self.update_thread = UpdateThread(self.workers, self.evaluator, self.args)
        self.update_thread.start()
        self.max_weight_sync_delay = self.args.max_weight_sync_delay
        self.steps_since_update = {}
        self.num_sampled_steps = 0
        self.num_updated_steps = 0
        self.log_dir = self.args.log_dir
        self.model_dir = self.args.model_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.writer = tf.summary.create_file_writer(self.log_dir)

        self.sample_tasks = TaskPool()
        self._set_workers()

        # fill buffer to replay starts
        while not all([l >= self.args.replay_starts for l in
                       ray.get([rb.__len__.remote() for rb in self.replay_buffers])]):
            for worker, objID in list(self.sample_tasks.completed()):
                sample_batch, count = ray.get(objID)
                random.choice(self.replay_buffers).add_batch.remote(sample_batch)
                self.num_sampled_steps += count
                self.sample_tasks.add(worker, worker.sample_with_count.remote())

        self.learn_tasks = TaskPool()
        self._set_learners()
        logger.info('Optimizer initialized')

    def _set_workers(self):
        weights = self.workers['local_worker'].get_weights()
        for worker in self.workers['remote_workers']:
            worker.set_weights.remote(weights)
            self.steps_since_update[worker] = 0
            self.sample_tasks.add(worker, worker.sample_with_count.remote())

    def _set_learners(self):
        weights = self.workers['local_worker'].get_weights()
        for learner in self.learners:
            learner.set_weights.remote(weights)
            rb = random.choice(self.replay_buffers)
            samples = ray.get(rb.replay.remote())
            learner.get_batch_data.remote(samples[:5])
            self.learn_tasks.add(learner, learner.compute_gradient.remote(self.workers['local_worker'].iteration))

    def step(self):
        # logger.info('updating {} in total'.format(self.num_updated_steps))
        # logger.info('sampling {} in total'.format(self.num_sampled_steps))
        assert self.update_thread.is_alive()
        assert len(self.workers['remote_workers']) > 0

        # sampling
        for worker, objID in self.sample_tasks.completed():
            sample_batch, count = ray.get(objID)
            random.choice(self.replay_buffers).add_batch.remote(sample_batch)
            self.num_sampled_steps += count
            self.steps_since_update[worker] += count
            if self.steps_since_update[worker] >= self.max_weight_sync_delay:
                worker.set_weights.remote(self.workers['local_worker'].get_weights())
                self.steps_since_update[worker] = 0
            self.sample_tasks.add(worker, worker.sample_with_count.remote())

        # learning
        for learner, objID in self.learn_tasks.completed():
            grads = ray.get(objID)
            rb = random.choice(self.replay_buffers)
            samples = ray.get(rb.replay.remote())
            assert grads is not None and samples is not None
            if self.args.buffer_type == 'priority':
                td_errors = learner.stats['td_error']
                rb.update_priorities.remote(samples[-1], td_errors)
            learner.get_batch_data.remote(samples[:5])
            del samples
            learner.set_weights.remote(self.workers['local_worker'].get_weights())
            self.learn_tasks.add(learner, learner.compute_gradient.remote(self.workers['local_worker'].iteration))
            self.update_thread.inqueue.put(grads)

        self.num_updated_steps = self.update_thread.iteration

    def stop(self):
        for r in self.workers['remote_workers']:
            r.__ray_terminate__.remote()
        for r in self.learners:
            r.__ray_terminate__.remote()
        for r in self.replay_buffers:
            r.__ray_terminate__.remote()
        self.evaluator.__ray_terminate__.remote()
        self.update_thread.stopped = True
