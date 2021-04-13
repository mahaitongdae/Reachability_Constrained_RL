#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/8/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: train_script.py
# =====================================

import argparse
import datetime
import json
import logging
import os

import gym
import safety_gym
import ray

from buffer import *
from evaluator import Evaluator, EvaluatorWithCost
from learners.ampc import AMPCLearner
from learners.mpg_learner import MPGLearner
from learners.nadp import NADPLearner
from learners.ndpg import NDPGLearner
from learners.sac import SACLearner, SACLearnerWithCost
from learners.td3 import TD3Learner
from optimizer import OffPolicyAsyncOptimizer, SingleProcessOffPolicyOptimizer, OffPolicyAsyncOptimizerWithCost
from policy import PolicyWithQs, PolicyWithMu
from tester import Tester
from trainer import Trainer
from worker import OffPolicyWorker, OffPolicyWorkerWithCost

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '1'
NAME2WORKERCLS = dict([('OffPolicyWorker', OffPolicyWorker),
                       ('OffPolicyWorkerWithCost', OffPolicyWorkerWithCost)])
NAME2LEARNERCLS = dict([('MPG', MPGLearner),
                        ('AMPC', AMPCLearner),
                        ('NADP', NADPLearner),
                        ('NDPG', NDPGLearner),
                        ('TD3', TD3Learner),
                        ('SAC', SACLearner),
                        ('FSAC', SACLearnerWithCost)
                        ])
NAME2BUFFERCLS = dict([('normal', ReplayBuffer),
                       ('priority', PrioritizedReplayBuffer),
                       ('None', None),
                       ('cost', ReplayBufferWithCost),
                       ('priority_cost', PrioritizedReplayBufferWithCost)])
NAME2OPTIMIZERCLS = dict([('OffPolicyAsync', OffPolicyAsyncOptimizer),
                          ('OffPolicyAsyncWithCost', OffPolicyAsyncOptimizerWithCost),
                          ('SingleProcessOffPolicy', SingleProcessOffPolicyOptimizer)])
NAME2POLICYCLS = dict([('PolicyWithQs', PolicyWithQs),('PolicyWithMu',PolicyWithMu)])
NAME2EVALUATORCLS = dict([('Evaluator', Evaluator), ('EvaluatorWithCost', EvaluatorWithCost), ('None', None)])
NUM_WORKER = 14
NUM_LEARNER = 14
NUM_BUFFER = 14

def built_FSAC_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='training') # training testing
    mode = parser.parse_args().mode

    if mode == 'testing':
        test_dir = '../results/FSAC/experiment-2021-04-08-05-03-05_300w'
        params = json.loads(open(test_dir + '/config.json').read())
        time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        test_log_dir = params['log_dir'] + '/tester/test-{}'.format(time_now)
        params.update(dict(test_dir=test_dir,
                           test_iter_list=[2300000],
                           test_log_dir=test_log_dir,
                           num_eval_episode=5,
                           num_eval_agent=1,
                           eval_log_interval=1,
                           fixed_steps=1000,
                           eval_render=True))
        for key, val in params.items():
            parser.add_argument("-" + key, default=val)
        return parser.parse_args()

    parser.add_argument('--motivation', type=str, default='test apply local grad')  # training testing

    # trainer
    parser.add_argument('--policy_type', type=str, default='PolicyWithMu')
    parser.add_argument('--worker_type', type=str, default='OffPolicyWorkerWithCost')
    parser.add_argument('--evaluator_type', type=str, default='EvaluatorWithCost')
    parser.add_argument('--buffer_type', type=str, default='priority_cost')
    parser.add_argument('--optimizer_type', type=str, default='OffPolicyAsyncWithCost') # SingleProcessOffPolicy OffPolicyAsyncWithCost
    parser.add_argument('--off_policy', type=str, default=True)
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--penalty_start', type=int, default=1500000)

    # env
    parser.add_argument('--env_id', default='Safexp-PointButton1-v0')
    parser.add_argument('--num_agent', type=int, default=1)
    parser.add_argument('--num_future_data', type=int, default=0)

    # learner
    parser.add_argument('--alg_name', default='FSAC')
    parser.add_argument('--constrained', default=True)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--cost_gamma', type=float, default=0.99)
    parser.add_argument('--gradient_clip_norm', type=float, default=10.)
    parser.add_argument('--num_batch_reuse', type=int, default=1)
    parser.add_argument('--cost_lim', type=float, default=10.0)
    parser.add_argument('--mlp_lam', default=True) # True: fsac, false: sac-lagrangian todo: add to new algo
    parser.add_argument('--double_QC', type=bool, default=True)

    # worker
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--worker_log_interval', type=int, default=5)
    parser.add_argument('--explore_sigma', type=float, default=None)

    # buffer
    parser.add_argument('--max_buffer_size', type=int, default=500000)
    parser.add_argument('--replay_starts', type=int, default=3000)
    parser.add_argument('--replay_batch_size', type=int, default=2048)
    parser.add_argument('--replay_alpha', type=float, default=0.6)
    parser.add_argument('--replay_beta', type=float, default=0.4)
    parser.add_argument('--buffer_log_interval', type=int, default=40000)

    # tester and evaluator
    parser.add_argument('--num_eval_episode', type=int, default=5)
    parser.add_argument('--eval_log_interval', type=int, default=1)
    parser.add_argument('--fixed_steps', type=int, default=1000)
    parser.add_argument('--eval_render', type=bool, default=False)
    num_eval_episode = parser.parse_args().num_eval_episode
    parser.add_argument('--num_eval_agent', type=int, default=1)

    # policy and model
    parser.add_argument('--obs_dim', type=int, default=None)
    parser.add_argument('--act_dim', type=int, default=None)
    parser.add_argument('--value_model_cls', type=str, default='MLP')
    parser.add_argument('--value_num_hidden_layers', type=int, default=2)
    parser.add_argument('--value_num_hidden_units', type=int, default=256)
    parser.add_argument('--value_hidden_activation', type=str, default='elu')
    parser.add_argument('--value_lr_schedule', type=list, default=[8e-5, 1000000, 8e-6])
    parser.add_argument('--cost_value_lr_schedule', type=list, default=[8e-5, 1000000, 8e-6])
    parser.add_argument('--policy_model_cls', type=str, default='MLP')
    parser.add_argument('--policy_num_hidden_layers', type=int, default=2)
    parser.add_argument('--policy_num_hidden_units', type=int, default=256)
    parser.add_argument('--policy_hidden_activation', type=str, default='elu')
    parser.add_argument('--policy_out_activation', type=str, default='linear')
    parser.add_argument('--policy_lr_schedule', type=list, default=[3e-5, 1000000, 3e-6])
    parser.add_argument('--lam_lr_schedule', type=list, default=[5e-6, 1000000, 5e-6])
    parser.add_argument('--alpha', default='auto')  # 'auto' 0.02
    alpha = parser.parse_args().alpha
    if alpha == 'auto':
        parser.add_argument('--target_entropy', type=float, default=-2)
    parser.add_argument('--alpha_lr_schedule', type=list, default=[8e-5, 1000000, 8e-6])
    parser.add_argument('--policy_only', type=bool, default=False)
    parser.add_argument('--double_Q', type=bool, default=True)
    parser.add_argument('--target', type=bool, default=True)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--delay_update', type=int, default=4)
    parser.add_argument('--dual_ascent_interval', type=int, default=12)
    parser.add_argument('--deterministic_policy', type=bool, default=False)
    parser.add_argument('--action_range', type=float, default=1.0)
    parser.add_argument('--mu_bias', type=float, default=0.0)
    cost_lim = parser.parse_args().cost_lim
    parser.add_argument('--cost_bias', type=float, default=0.0)

    # preprocessor
    parser.add_argument('--obs_ptype', type=str, default='scale')
    num_future_data = parser.parse_args().num_future_data
    parser.add_argument('--obs_scale', type=list, default=None)
    parser.add_argument('--rew_ptype', type=str, default='scale')
    parser.add_argument('--rew_scale', type=float, default=1.)
    parser.add_argument('--rew_shift', type=float, default=0.)

    # Optimizer (PABAL)
    parser.add_argument('--max_sampled_steps', type=int, default=0)
    parser.add_argument('--max_iter', type=int, default=3000000)
    parser.add_argument('--num_workers', type=int, default=NUM_WORKER)
    parser.add_argument('--num_learners', type=int, default=NUM_LEARNER)
    parser.add_argument('--num_buffers', type=int, default=NUM_BUFFER)
    parser.add_argument('--max_weight_sync_delay', type=int, default=30)
    parser.add_argument('--grads_queue_size', type=int, default=25)
    parser.add_argument('--grads_max_reuse', type=int, default=10)
    parser.add_argument('--eval_interval', type=int, default=5000)
    parser.add_argument('--save_interval', type=int, default=200000)
    parser.add_argument('--log_interval', type=int, default=100)

    # IO
    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    results_dir = '../results/FSAC/experiment-{time}'.format(time=time_now)
    parser.add_argument('--result_dir', type=str, default=results_dir)
    parser.add_argument('--log_dir', type=str, default=results_dir + '/logs')
    parser.add_argument('--model_dir', type=str, default=results_dir + '/models')
    parser.add_argument('--model_load_dir', type=str, default=None)
    parser.add_argument('--model_load_ite', type=int, default=None)
    parser.add_argument('--ppc_load_dir', type=str, default=None)

    return parser.parse_args()

def built_parser(alg_name):
    if alg_name == 'TD3':
        args = built_TD3_parser()
    elif alg_name == 'SAC':
        args = built_SAC_parser()
    elif alg_name == 'FSAC':
        args = built_FSAC_parser()
    elif alg_name == 'MPG-v1':
        args = built_MPG_parser('MPG-v1')
    elif alg_name == 'MPG-v2':
        args = built_MPG_parser('MPG-v2')
    elif alg_name == 'NDPG':
        args = built_NDPG_parser()
    elif alg_name == 'NADP':
        args = built_NADP_parser()
    elif alg_name == 'AMPC':
        args = built_AMPC_parser()
    env = gym.make(args.env_id) #  **vars(args)
    args.obs_dim, args.act_dim = int(env.observation_space.shape[0]), int(env.action_space.shape[0])
    args.obs_scale = [1.] * args.obs_dim
    return args

def main(alg_name):
    args = built_parser(alg_name)
    logger.info('begin training agents with parameter {}'.format(str(args)))
    if args.mode == 'training':
        ray.init(object_store_memory=32768*1024*1024)
        os.makedirs(args.result_dir)
        with open(args.result_dir + '/config.json', 'w', encoding='utf-8') as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=4)
        trainer = Trainer(policy_cls=NAME2POLICYCLS[args.policy_type],
                          worker_cls=NAME2WORKERCLS[args.worker_type],
                          learner_cls=NAME2LEARNERCLS[args.alg_name],
                          buffer_cls=NAME2BUFFERCLS[args.buffer_type],
                          optimizer_cls=NAME2OPTIMIZERCLS[args.optimizer_type],
                          evaluator_cls=NAME2EVALUATORCLS[args.evaluator_type],
                          args=args)
        if args.model_load_dir is not None:
            logger.info('loading model')
            trainer.load_weights(args.model_load_dir, args.model_load_ite)
        if args.ppc_load_dir is not None:
            logger.info('loading ppc parameter')
            trainer.load_ppc_params(args.ppc_load_dir)
        trainer.train()

    elif args.mode == 'testing':
        os.makedirs(args.test_log_dir)
        with open(args.test_log_dir + '/test_config.json', 'w', encoding='utf-8') as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=4)
        tester = Tester(policy_cls=NAME2POLICYCLS[args.policy_type],
                        evaluator_cls=NAME2EVALUATORCLS[args.evaluator_type],
                        args=args)
        tester.test()


if __name__ == '__main__':
    main('FSAC')
