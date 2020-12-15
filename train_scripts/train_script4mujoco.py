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
import ray

from buffer import PrioritizedReplayBuffer, ReplayBuffer
from evaluator import Evaluator
from learners.ampc import AMPCLearner
from learners.mpg_learner import MPGLearner
from learners.nadp import NADPLearner
from learners.ndpg import NDPGLearner
from learners.sac import SACLearner
from learners.td3 import TD3Learner
from optimizer import OffPolicyAsyncOptimizer, SingleProcessOffPolicyOptimizer
from policy import PolicyWithQs
from tester import Tester
from trainer import Trainer
from worker import OffPolicyWorker

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
NAME2WORKERCLS = dict([('OffPolicyWorker', OffPolicyWorker)])
NAME2LEARNERCLS = dict([('MPG', MPGLearner),
                        ('AMPC', AMPCLearner),
                        ('NADP', NADPLearner),
                        ('NDPG', NDPGLearner),
                        ('TD3', TD3Learner),
                        ('SAC', SACLearner)
                        ])
NAME2BUFFERCLS = dict([('normal', ReplayBuffer), ('priority', PrioritizedReplayBuffer), ('None', None)])
NAME2OPTIMIZERCLS = dict([('OffPolicyAsync', OffPolicyAsyncOptimizer),
                          ('SingleProcessOffPolicy', SingleProcessOffPolicyOptimizer)])
NAME2POLICYCLS = dict([('PolicyWithQs', PolicyWithQs)])
NAME2EVALUATORCLS = dict([('Evaluator', Evaluator), ('None', None)])


def built_AMPC_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='training') # training testing
    mode = parser.parse_args().mode

    if mode == 'testing':
        test_dir = '../results/AMPC/experiment-2020-09-03-17-04-11'
        params = json.loads(open(test_dir + '/config.json').read())
        time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        test_log_dir = params['log_dir'] + '/tester/test-{}'.format(time_now)
        params.update(dict(test_dir=test_dir,
                           test_iter_list=[99000],
                           test_log_dir=test_log_dir,
                           num_eval_episode=5,
                           num_eval_agent=1,
                           eval_log_interval=1,
                           fixed_steps=500))
        for key, val in params.items():
            parser.add_argument("-" + key, default=val)
        return parser.parse_args()

    # trainer
    parser.add_argument('--policy_type', type=str, default='PolicyWithQs')
    parser.add_argument('--worker_type', type=str, default='OffPolicyWorker')
    parser.add_argument('--evaluator_type', type=str, default='Evaluator')
    parser.add_argument('--buffer_type', type=str, default='normal')
    parser.add_argument('--optimizer_type', type=str, default='OffPolicyAsync')  # OffPolicyAsync, SingleProcessOffPolicy
    parser.add_argument('--off_policy', type=str, default=True)

    # env
    parser.add_argument('--env_id', default='InvertedPendulumConti-v0')
    parser.add_argument('--num_agent', type=int, default=1)

    # learner
    parser.add_argument('--alg_name', default='AMPC')
    parser.add_argument('--M', type=int, default=1)
    parser.add_argument('--num_rollout_list_for_policy_update', type=list, default=[25])
    parser.add_argument('--gamma', type=float, default=1.)
    parser.add_argument('--gradient_clip_norm', type=float, default=3)

    # worker
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--worker_log_interval', type=int, default=5)
    parser.add_argument('--explore_sigma', type=float, default=None)

    # buffer
    parser.add_argument('--max_buffer_size', type=int, default=500000)
    parser.add_argument('--replay_starts', type=int, default=3000)
    parser.add_argument('--replay_batch_size', type=int, default=256)
    parser.add_argument('--replay_alpha', type=float, default=0.6)
    parser.add_argument('--replay_beta', type=float, default=0.4)
    parser.add_argument('--buffer_log_interval', type=int, default=40000)

    # tester and evaluator
    parser.add_argument('--num_eval_episode', type=int, default=5)
    parser.add_argument('--eval_log_interval', type=int, default=1)
    parser.add_argument('--fixed_steps', type=int, default=200)
    parser.add_argument('--eval_render', type=bool, default=True)
    parser.add_argument('--num_eval_agent', type=int, default=1)

    # policy and model
    parser.add_argument('--obs_dim', type=int, default=None)
    parser.add_argument('--act_dim', type=int, default=None)
    parser.add_argument('--value_model_cls', type=str, default='MLP')
    parser.add_argument('--value_num_hidden_layers', type=int, default=2)
    parser.add_argument('--value_num_hidden_units', type=int, default=256)
    parser.add_argument('--value_hidden_activation', type=str, default='elu')
    parser.add_argument('--value_lr_schedule', type=list, default=[8e-5, 100000, 8e-6])
    parser.add_argument('--policy_model_cls', type=str, default='MLP')
    parser.add_argument('--policy_num_hidden_layers', type=int, default=2)
    parser.add_argument('--policy_num_hidden_units', type=int, default=256)
    parser.add_argument('--policy_hidden_activation', type=str, default='elu')
    parser.add_argument('--policy_out_activation', type=str, default='linear')
    parser.add_argument('--policy_lr_schedule', type=list, default=[3e-5, 100000, 3e-6])
    parser.add_argument('--alpha', default=None)
    parser.add_argument('--alpha_lr_schedule', type=list, default=None)
    parser.add_argument('--policy_only', default=True, action='store_true')
    parser.add_argument('--double_Q', type=bool, default=False)
    parser.add_argument('--target', type=bool, default=False)
    parser.add_argument('--tau', type=float, default=None)
    parser.add_argument('--delay_update', type=int, default=None)
    parser.add_argument('--deterministic_policy', type=bool, default=True)
    parser.add_argument('--action_range', type=float, default=3.)

    # preprocessor
    parser.add_argument('--obs_ptype', type=str, default=None)
    parser.add_argument('--obs_scale', type=list, default=None)
    parser.add_argument('--rew_ptype', type=str, default='scale')
    parser.add_argument('--rew_scale', type=float, default=0.01)
    parser.add_argument('--rew_shift', type=float, default=0.0)

    # optimizer (PABAL)
    parser.add_argument('--max_sampled_steps', type=int, default=0)
    parser.add_argument('--max_iter', type=int, default=100000)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--num_learners', type=int, default=8)
    parser.add_argument('--num_buffers', type=int, default=2)
    parser.add_argument('--max_weight_sync_delay', type=int, default=300)
    parser.add_argument('--grads_queue_size', type=int, default=20)
    parser.add_argument('--eval_interval', type=int, default=3000)
    parser.add_argument('--save_interval', type=int, default=3000)
    parser.add_argument('--log_interval', type=int, default=100)

    # IO
    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    results_dir = '../results/ampc/experiment-{time}'.format(time=time_now)
    parser.add_argument('--result_dir', type=str, default=results_dir)
    parser.add_argument('--log_dir', type=str, default=results_dir + '/logs')
    parser.add_argument('--model_dir', type=str, default=results_dir + '/models')
    parser.add_argument('--model_load_dir', type=str, default=None)
    parser.add_argument('--model_load_ite', type=int, default=None)
    parser.add_argument('--ppc_load_dir', type=str, default=None)

    return parser.parse_args()

def built_MPG_parser(version):
    #          Target                Weighting method
    # MPG-v1   n-step                Heuristic
    # MPG-v2   clipped double-Q      Rule-based
    # MPG-v3   n-step                Rule-based
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='training') # training testing
    mode = parser.parse_args().mode

    if mode == 'testing':
        test_dir = '../results/toyota/experiment-2020-09-03-17-04-11'
        params = json.loads(open(test_dir + '/config.json').read())
        time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        test_log_dir = params['log_dir'] + '/tester/test-{}'.format(time_now)
        params.update(dict(test_dir=test_dir,
                           test_iter_list=[99000],
                           test_log_dir=test_log_dir,
                           num_eval_episode=5,
                           num_eval_agent=1,
                           eval_log_interval=1,
                           fixed_steps=500))
        for key, val in params.items():
            parser.add_argument("-" + key, default=val)
        return parser.parse_args()

    # trainer
    parser.add_argument('--policy_type', type=str, default='PolicyWithQs')
    parser.add_argument('--worker_type', type=str, default='OffPolicyWorker')
    parser.add_argument('--buffer_type', type=str, default='normal')
    parser.add_argument('--evaluator_type', type=str, default='Evaluator')
    parser.add_argument('--optimizer_type', type=str, default='OffPolicyAsync')
    parser.add_argument('--off_policy', type=str, default=True)

    # env
    parser.add_argument('--env_id', default='InvertedPendulumConti-v0')
    parser.add_argument('--num_agent', type=int, default=1)

    # learner
    parser.add_argument('--alg_name', default='MPG')
    parser.add_argument('--learner_version', default=version)
    parser.add_argument('--sample_num_in_learner', type=int, default=25)
    parser.add_argument('--M', type=int, default=1)
    parser.add_argument('--deriv_interval_policy', default=False, action='store_true')
    parser.add_argument('--num_rollout_list_for_policy_update', type=list, default=[0, 25])
    parser.add_argument('--num_rollout_list_for_q_estimation', type=list, default=[0, 25] if version == 'MPG-v1' else [])
    if version == 'MPG-v2' or version == 'MPG-v3':
        parser.add_argument('--eta', type=float, default=0.1)
        parser.add_argument('--rule_based_bias_total_ite', type=int, default=9000)
    else:
        assert version == 'MPG-v1'
        parser.add_argument('--thres', type=float, default=0.98)
        parser.add_argument('--w_moving_rate', type=float, default=0.01)

    parser.add_argument('--gamma', type=float, default=0.98)
    parser.add_argument('--gradient_clip_norm', type=float, default=3)
    parser.add_argument('--num_batch_reuse', type=int, default=10 if version == 'MPG-v1' or version == 'MPG-v3' else 1)

    # worker
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--worker_log_interval', type=int, default=5)
    parser.add_argument('--explore_sigma', type=float, default=0.1)

    # buffer
    parser.add_argument('--max_buffer_size', type=int, default=500000)
    parser.add_argument('--replay_starts', type=int, default=3000)
    parser.add_argument('--replay_batch_size', type=int, default=256)
    parser.add_argument('--replay_alpha', type=float, default=0.6)
    parser.add_argument('--replay_beta', type=float, default=0.4)
    parser.add_argument('--buffer_log_interval', type=int, default=40000)

    # tester and evaluator
    parser.add_argument('--num_eval_episode', type=int, default=5)
    parser.add_argument('--eval_log_interval', type=int, default=1)
    parser.add_argument('--fixed_steps', type=int, default=200)
    parser.add_argument('--eval_render', type=bool, default=True)
    parser.add_argument('--num_eval_agent', type=int, default=1)

    # policy and model
    parser.add_argument('--obs_dim', type=int, default=None)
    parser.add_argument('--act_dim', type=int, default=None)
    parser.add_argument('--value_model_cls', type=str, default='MLP')
    parser.add_argument('--value_num_hidden_layers', type=int, default=2)
    parser.add_argument('--value_num_hidden_units', type=int, default=256)
    parser.add_argument('--value_hidden_activation', type=str, default='elu')
    parser.add_argument('--value_lr_schedule', type=list, default=[8e-5, 100000, 8e-6])
    parser.add_argument('--policy_model_cls', type=str, default='MLP')
    parser.add_argument('--policy_num_hidden_layers', type=int, default=2)
    parser.add_argument('--policy_num_hidden_units', type=int, default=256)
    parser.add_argument('--policy_hidden_activation', type=str, default='elu')
    parser.add_argument('--policy_out_activation', type=str, default='linear')
    parser.add_argument('--policy_lr_schedule', type=list, default=[3e-5, 100000, 3e-6])
    parser.add_argument('--alpha', default=None)
    parser.add_argument('--alpha_lr_schedule', type=list, default=None)
    parser.add_argument('--policy_only', type=bool, default=False)
    parser.add_argument('--double_Q', type=bool, default=True if version == 'MPG-v2' else False)
    parser.add_argument('--target', type=bool, default=True)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--delay_update', type=int, default=2)
    parser.add_argument('--deterministic_policy', type=bool, default=True)
    parser.add_argument('--action_range', type=float, default=3.)

    # preprocessor
    parser.add_argument('--obs_ptype', type=str, default=None)
    parser.add_argument('--obs_scale', type=list, default=None)
    parser.add_argument('--rew_ptype', type=str, default='scale')
    parser.add_argument('--rew_scale', type=float, default=1.)
    parser.add_argument('--rew_shift', type=float, default=0.0)

    # Optimizer (PABAL)
    parser.add_argument('--max_sampled_steps', type=int, default=0)
    parser.add_argument('--max_iter', type=int, default=100000)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--num_learners', type=int, default=12)
    parser.add_argument('--num_buffers', type=int, default=2)
    parser.add_argument('--max_weight_sync_delay', type=int, default=300)
    parser.add_argument('--grads_queue_size', type=int, default=20)
    parser.add_argument('--eval_interval', type=int, default=3000)
    parser.add_argument('--save_interval', type=int, default=3000)
    parser.add_argument('--log_interval', type=int, default=100)

    # IO
    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    results_dir = '../results/{}/experiment-{time}'.format(version, time=time_now)
    parser.add_argument('--result_dir', type=str, default=results_dir)
    parser.add_argument('--log_dir', type=str, default=results_dir + '/logs')
    parser.add_argument('--model_dir', type=str, default=results_dir + '/models')
    parser.add_argument('--model_load_dir', type=str, default=None)
    parser.add_argument('--model_load_ite', type=int, default=None)
    parser.add_argument('--ppc_load_dir', type=str, default=None)

    return parser.parse_args()

def built_NADP_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='training') # training testing
    mode = parser.parse_args().mode

    if mode == 'testing':
        test_dir = '../results/NADP/experiment-2020-09-23-20-52-24'
        params = json.loads(open(test_dir + '/config.json').read())
        time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        test_log_dir = params['log_dir'] + '/tester/test-{}'.format(time_now)
        params.update(dict(test_dir=test_dir,
                           test_iter_list=[99000],
                           test_log_dir=test_log_dir,
                           num_eval_episode=5,
                           num_eval_agent=1,
                           eval_log_interval=1,
                           fixed_steps=500))
        for key, val in params.items():
            parser.add_argument("-" + key, default=val)
        return parser.parse_args()

    # trainer
    parser.add_argument('--policy_type', type=str, default='PolicyWithQs')
    parser.add_argument('--worker_type', type=str, default='OffPolicyWorker')
    parser.add_argument('--evaluator_type', type=str, default='Evaluator')
    parser.add_argument('--buffer_type', type=str, default='normal')
    parser.add_argument('--optimizer_type', type=str, default='OffPolicyAsync')
    parser.add_argument('--off_policy', type=str, default=True)

    # env
    parser.add_argument('--env_id', default='InvertedPendulumConti-v0')
    parser.add_argument('--num_agent', type=int, default=1)

    # learner
    parser.add_argument('--alg_name', default='NADP')
    parser.add_argument('--M', type=int, default=1)
    parser.add_argument('--num_rollout_list_for_policy_update', type=list, default=[25])
    parser.add_argument('--num_rollout_list_for_q_estimation', type=list, default=[25])
    parser.add_argument('--gamma', type=float, default=0.98)
    parser.add_argument('--gradient_clip_norm', type=float, default=3)
    parser.add_argument('--num_batch_reuse', type=int, default=1)

    # worker
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--worker_log_interval', type=int, default=5)
    parser.add_argument('--explore_sigma', type=float, default=None)

    # buffer
    parser.add_argument('--max_buffer_size', type=int, default=500000)
    parser.add_argument('--replay_starts', type=int, default=3000)
    parser.add_argument('--replay_batch_size', type=int, default=256)
    parser.add_argument('--replay_alpha', type=float, default=0.6)
    parser.add_argument('--replay_beta', type=float, default=0.4)
    parser.add_argument('--buffer_log_interval', type=int, default=40000)

    # tester and evaluator
    parser.add_argument('--num_eval_episode', type=int, default=5)
    parser.add_argument('--eval_log_interval', type=int, default=1)
    parser.add_argument('--fixed_steps', type=int, default=200)
    parser.add_argument('--eval_render', type=bool, default=True)
    parser.add_argument('--num_eval_agent', type=int, default=1)

    # policy and model
    parser.add_argument('--obs_dim', type=int, default=None)
    parser.add_argument('--act_dim', type=int, default=None)
    parser.add_argument('--value_model_cls', type=str, default='MLP')
    parser.add_argument('--value_num_hidden_layers', type=int, default=2)
    parser.add_argument('--value_num_hidden_units', type=int, default=256)
    parser.add_argument('--value_hidden_activation', type=str, default='elu')
    parser.add_argument('--value_lr_schedule', type=list, default=[8e-5, 100000, 8e-6])
    parser.add_argument('--policy_model_cls', type=str, default='MLP')
    parser.add_argument('--policy_num_hidden_layers', type=int, default=2)
    parser.add_argument('--policy_num_hidden_units', type=int, default=256)
    parser.add_argument('--policy_hidden_activation', type=str, default='elu')
    parser.add_argument('--policy_out_activation', type=str, default='linear')
    parser.add_argument('--policy_lr_schedule', type=list, default=[3e-5, 100000, 3e-6])
    parser.add_argument('--alpha', default=None)
    parser.add_argument('--alpha_lr_schedule', type=list, default=None)
    parser.add_argument('--policy_only', type=bool, default=False)
    parser.add_argument('--double_Q', type=bool, default=False)
    parser.add_argument('--target', type=bool, default=True)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--delay_update', type=int, default=1)
    parser.add_argument('--deterministic_policy', type=bool, default=True)
    parser.add_argument('--action_range', type=float, default=3.)

    # preprocessor
    parser.add_argument('--obs_ptype', type=str, default='scale')
    parser.add_argument('--obs_scale', type=list, default=[0.001, 1/3, 0.1, 0.5])
    parser.add_argument('--rew_ptype', type=str, default='scale')
    parser.add_argument('--rew_scale', type=float, default=1.)
    parser.add_argument('--rew_shift', type=float, default=0.0)

    # optimizer (PABAL)
    parser.add_argument('--max_sampled_steps', type=int, default=0)
    parser.add_argument('--max_iter', type=int, default=100000)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--num_learners', type=int, default=8)
    parser.add_argument('--num_buffers', type=int, default=2)
    parser.add_argument('--max_weight_sync_delay', type=int, default=300)
    parser.add_argument('--grads_queue_size', type=int, default=20)
    parser.add_argument('--eval_interval', type=int, default=3000)
    parser.add_argument('--save_interval', type=int, default=3000)
    parser.add_argument('--log_interval', type=int, default=100)

    # IO
    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    results_dir = '../results/NADP/experiment-{time}'.format(time=time_now)
    parser.add_argument('--result_dir', type=str, default=results_dir)
    parser.add_argument('--log_dir', type=str, default=results_dir + '/logs')
    parser.add_argument('--model_dir', type=str, default=results_dir + '/models')
    parser.add_argument('--model_load_dir', type=str, default=None)
    parser.add_argument('--model_load_ite', type=int, default=None)
    parser.add_argument('--ppc_load_dir', type=str, default=None)

    return parser.parse_args()

def built_NDPG_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='training') # training testing
    mode = parser.parse_args().mode

    if mode == 'testing':
        test_dir = '../results/NDPG/experiment-2020-09-03-17-04-11'
        params = json.loads(open(test_dir + '/config.json').read())
        time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        test_log_dir = params['log_dir'] + '/tester/test-{}'.format(time_now)
        params.update(dict(test_dir=test_dir,
                           test_iter_list=[0],
                           test_log_dir=test_log_dir,
                           num_eval_episode=5,
                           num_eval_agent=5,
                           eval_log_interval=1,
                           fixed_steps=500))
        for key, val in params.items():
            parser.add_argument("-" + key, default=val)
        return parser.parse_args()

    # trainer
    parser.add_argument('--policy_type', type=str, default='PolicyWithQs')
    parser.add_argument('--worker_type', type=str, default='OffPolicyWorker')
    parser.add_argument('--buffer_type', type=str, default='normal')
    parser.add_argument('--optimizer_type', type=str, default='OffPolicyAsync')
    parser.add_argument('--evaluator_type', type=str, default='Evaluator')
    parser.add_argument('--off_policy', type=str, default=True)

    # env
    parser.add_argument('--env_id', default='InvertedPendulumConti-v0')
    parser.add_argument('--num_agent', type=int, default=1)

    # learner
    parser.add_argument('--alg_name', default='NDPG')
    parser.add_argument('--sample_num_in_learner', type=int, default=25)
    parser.add_argument('--gamma', type=float, default=0.98)
    parser.add_argument('--gradient_clip_norm', type=float, default=3)
    parser.add_argument('--num_batch_reuse', type=int, default=10)

    # worker
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--worker_log_interval', type=int, default=5)
    parser.add_argument('--explore_sigma', type=float, default=None)

    # buffer
    parser.add_argument('--max_buffer_size', type=int, default=500000)
    parser.add_argument('--replay_starts', type=int, default=3000)
    parser.add_argument('--replay_batch_size', type=int, default=256)
    parser.add_argument('--replay_alpha', type=float, default=0.6)
    parser.add_argument('--replay_beta', type=float, default=0.4)
    parser.add_argument('--buffer_log_interval', type=int, default=40000)

    # tester and evaluator
    parser.add_argument('--num_eval_episode', type=int, default=5)
    parser.add_argument('--eval_log_interval', type=int, default=1)
    parser.add_argument('--fixed_steps', type=int, default=200)
    parser.add_argument('--eval_render', type=bool, default=True)
    parser.add_argument('--num_eval_agent', type=int, default=1)

    # policy and model
    parser.add_argument('--obs_dim', type=int, default=None)
    parser.add_argument('--act_dim', type=int, default=None)
    parser.add_argument('--value_model_cls', type=str, default='MLP')
    parser.add_argument('--value_num_hidden_layers', type=int, default=2)
    parser.add_argument('--value_num_hidden_units', type=int, default=256)
    parser.add_argument('--value_hidden_activation', type=str, default='elu')
    parser.add_argument('--value_out_activation', type=str, default='tanh')
    parser.add_argument('--value_lr_schedule', type=list, default=[8e-5, 100000, 8e-6])
    parser.add_argument('--policy_model_cls', type=str, default='MLP')
    parser.add_argument('--policy_num_hidden_layers', type=int, default=2)
    parser.add_argument('--policy_num_hidden_units', type=int, default=256)
    parser.add_argument('--policy_hidden_activation', type=str, default='elu')
    parser.add_argument('--policy_out_activation', type=str, default='linear')
    parser.add_argument('--policy_lr_schedule', type=list, default=[3e-5, 100000, 3e-6])
    parser.add_argument('--alpha', default=None)
    parser.add_argument('--alpha_lr_schedule', type=list, default=None)
    parser.add_argument('--policy_only', type=bool, default=False)
    parser.add_argument('--double_Q', type=bool, default=False)
    parser.add_argument('--target', type=bool, default=True)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--delay_update', type=int, default=1)
    parser.add_argument('--deterministic_policy', type=bool, default=True)
    parser.add_argument('--action_range', type=float, default=3.)

    # preprocessor
    parser.add_argument('--obs_ptype', type=str, default='scale')
    parser.add_argument('--obs_scale', type=list, default=[0.001, 1/3, 0.1, 0.5])
    parser.add_argument('--rew_ptype', type=str, default='scale')
    parser.add_argument('--rew_scale', type=float, default=0.01)
    parser.add_argument('--rew_shift', type=float, default=0.0)

    # Optimizer (PABAL)
    parser.add_argument('--max_sampled_steps', type=int, default=0)
    parser.add_argument('--max_iter', type=int, default=100000)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--num_learners', type=int, default=12)
    parser.add_argument('--num_buffers', type=int, default=2)
    parser.add_argument('--max_weight_sync_delay', type=int, default=300)
    parser.add_argument('--grads_queue_size', type=int, default=20)
    parser.add_argument('--eval_interval', type=int, default=3000)
    parser.add_argument('--save_interval', type=int, default=3000)
    parser.add_argument('--log_interval', type=int, default=100)

    # IO
    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    results_dir = '../results/NDPG/experiment-{time}'.format(time=time_now)
    parser.add_argument('--result_dir', type=str, default=results_dir)
    parser.add_argument('--log_dir', type=str, default=results_dir + '/logs')
    parser.add_argument('--model_dir', type=str, default=results_dir + '/models')
    parser.add_argument('--model_load_dir', type=str, default=None)
    parser.add_argument('--model_load_ite', type=int, default=None)
    parser.add_argument('--ppc_load_dir', type=str, default=None)

    return parser.parse_args()

def built_TD3_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='training') # training testing
    mode = parser.parse_args().mode

    if mode == 'testing':
        test_dir = './results/TD3/experiment-2020-09-03-17-04-11'
        params = json.loads(open(test_dir + '/config.json').read())
        time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        test_log_dir = params['log_dir'] + '/tester/test-{}'.format(time_now)
        params.update(dict(test_dir=test_dir,
                           test_iter_list=[0],
                           test_log_dir=test_log_dir,
                           num_eval_episode=5,
                           num_eval_agent=5,
                           eval_log_interval=1,
                           fixed_steps=70))
        for key, val in params.items():
            parser.add_argument("-" + key, default=val)
        return parser.parse_args()

    # trainer
    parser.add_argument('--policy_type', type=str, default='PolicyWithQs')
    parser.add_argument('--worker_type', type=str, default='OffPolicyWorker')
    parser.add_argument('--evaluator_type', type=str, default='Evaluator')
    parser.add_argument('--buffer_type', type=str, default='normal')
    parser.add_argument('--optimizer_type', type=str, default='OffPolicyAsync')
    parser.add_argument('--off_policy', type=str, default=True)

    # env
    parser.add_argument('--env_id', default='PathTracking-v0')
    parser.add_argument('--num_agent', type=int, default=8)
    parser.add_argument('--num_future_data', type=int, default=0)

    # learner
    parser.add_argument('--alg_name', default='TD3')
    parser.add_argument('--gamma', type=float, default=0.98)
    parser.add_argument('--gradient_clip_norm', type=float, default=3)
    parser.add_argument('--policy_smoothing_sigma', type=float, default=0.2)
    parser.add_argument('--policy_smoothing_clip', type=float, default=0.5)
    parser.add_argument('--num_batch_reuse', type=int, default=1)

    # worker
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--worker_log_interval', type=int, default=5)
    parser.add_argument('--explore_sigma', type=float, default=0.1)

    # buffer
    parser.add_argument('--max_buffer_size', type=int, default=500000)
    parser.add_argument('--replay_starts', type=int, default=3000)
    parser.add_argument('--replay_batch_size', type=int, default=256)
    parser.add_argument('--replay_alpha', type=float, default=0.6)
    parser.add_argument('--replay_beta', type=float, default=0.4)
    parser.add_argument('--buffer_log_interval', type=int, default=40000)

    # tester and evaluator
    parser.add_argument('--num_eval_episode', type=int, default=5)
    parser.add_argument('--eval_log_interval', type=int, default=1)
    parser.add_argument('--fixed_steps', type=int, default=200)
    parser.add_argument('--eval_render', type=bool, default=True)
    num_eval_episode = parser.parse_args().num_eval_episode
    parser.add_argument('--num_eval_agent', type=int, default=num_eval_episode)

    # policy and model
    parser.add_argument('--obs_dim', type=int, default=None)
    parser.add_argument('--act_dim', type=int, default=None)
    parser.add_argument('--value_model_cls', type=str, default='MLP')
    parser.add_argument('--value_num_hidden_layers', type=int, default=2)
    parser.add_argument('--value_num_hidden_units', type=int, default=256)
    parser.add_argument('--value_hidden_activation', type=str, default='elu')
    parser.add_argument('--value_lr_schedule', type=list, default=[8e-5, 100000, 8e-6])
    parser.add_argument('--policy_model_cls', type=str, default='MLP')
    parser.add_argument('--policy_num_hidden_layers', type=int, default=2)
    parser.add_argument('--policy_num_hidden_units', type=int, default=256)
    parser.add_argument('--policy_hidden_activation', type=str, default='elu')
    parser.add_argument('--policy_out_activation', type=str, default='tanh')
    parser.add_argument('--policy_lr_schedule', type=list, default=[3e-5, 100000, 3e-6])
    parser.add_argument('--alpha', default=None)
    parser.add_argument('--alpha_lr_schedule', type=list, default=None)
    parser.add_argument('--policy_only', type=bool, default=False)
    parser.add_argument('--double_Q', type=bool, default=True)
    parser.add_argument('--target', type=bool, default=True)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--delay_update', type=int, default=2)
    parser.add_argument('--deterministic_policy', type=bool, default=True)
    parser.add_argument('--action_range', type=float, default=None)

    # preprocessor
    parser.add_argument('--obs_ptype', type=str, default='scale')
    num_future_data = parser.parse_args().num_future_data
    parser.add_argument('--obs_scale', type=list, default=[1., 1., 2., 1., 2.4, 1 / 1200] + [1.] * num_future_data)
    parser.add_argument('--rew_ptype', type=str, default='scale')
    parser.add_argument('--rew_scale', type=float, default=0.01)
    parser.add_argument('--rew_shift', type=float, default=0.)

    # Optimizer (PABAL)
    parser.add_argument('--max_sampled_steps', type=int, default=0)
    parser.add_argument('--max_iter', type=int, default=100000)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--num_learners', type=int, default=2)
    parser.add_argument('--num_buffers', type=int, default=2)
    parser.add_argument('--max_weight_sync_delay', type=int, default=300)
    parser.add_argument('--grads_queue_size', type=int, default=20)
    parser.add_argument('--eval_interval', type=int, default=3000)
    parser.add_argument('--save_interval', type=int, default=3000)
    parser.add_argument('--log_interval', type=int, default=100)

    # IO
    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    results_dir = './results/TD3/experiment-{time}'.format(time=time_now)
    parser.add_argument('--result_dir', type=str, default=results_dir)
    parser.add_argument('--log_dir', type=str, default=results_dir + '/logs')
    parser.add_argument('--model_dir', type=str, default=results_dir + '/models')
    parser.add_argument('--model_load_dir', type=str, default=None)
    parser.add_argument('--model_load_ite', type=int, default=None)
    parser.add_argument('--ppc_load_dir', type=str, default=None)

    return parser.parse_args()

def built_SAC_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='training') # training testing
    mode = parser.parse_args().mode

    if mode == 'testing':
        test_dir = './results/SAC/experiment-2020-09-22-14-46-31'
        params = json.loads(open(test_dir + '/config.json').read())
        time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        test_log_dir = params['log_dir'] + '/tester/test-{}'.format(time_now)
        params.update(dict(test_dir=test_dir,
                           test_iter_list=[100000],
                           test_log_dir=test_log_dir,
                           num_eval_episode=50,
                           num_eval_agent=50,
                           eval_log_interval=1,
                           fixed_steps=200))
        for key, val in params.items():
            parser.add_argument("-" + key, default=val)
        return parser.parse_args()

    # trainer
    parser.add_argument('--policy_type', type=str, default='PolicyWithQs')
    parser.add_argument('--worker_type', type=str, default='OffPolicyWorker')
    parser.add_argument('--evaluator_type', type=str, default='Evaluator')
    parser.add_argument('--buffer_type', type=str, default='normal')
    parser.add_argument('--optimizer_type', type=str, default='OffPolicyAsync')
    parser.add_argument('--off_policy', type=str, default=True)

    # env
    parser.add_argument('--env_id', default='PathTracking-v0')
    parser.add_argument('--num_agent', type=int, default=8)
    parser.add_argument('--num_future_data', type=int, default=0)

    # learner
    parser.add_argument('--alg_name', default='SAC')
    parser.add_argument('--gamma', type=float, default=0.98)
    parser.add_argument('--gradient_clip_norm', type=float, default=3)
    parser.add_argument('--num_batch_reuse', type=int, default=1)

    # worker
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--worker_log_interval', type=int, default=5)
    parser.add_argument('--explore_sigma', type=float, default=None)

    # buffer
    parser.add_argument('--max_buffer_size', type=int, default=500000)
    parser.add_argument('--replay_starts', type=int, default=3000)
    parser.add_argument('--replay_batch_size', type=int, default=256)
    parser.add_argument('--replay_alpha', type=float, default=0.6)
    parser.add_argument('--replay_beta', type=float, default=0.4)
    parser.add_argument('--buffer_log_interval', type=int, default=40000)

    # tester and evaluator
    parser.add_argument('--num_eval_episode', type=int, default=5)
    parser.add_argument('--eval_log_interval', type=int, default=1)
    parser.add_argument('--fixed_steps', type=int, default=200)
    parser.add_argument('--eval_render', type=bool, default=True)
    num_eval_episode = parser.parse_args().num_eval_episode
    parser.add_argument('--num_eval_agent', type=int, default=num_eval_episode)

    # policy and model
    parser.add_argument('--obs_dim', type=int, default=None)
    parser.add_argument('--act_dim', type=int, default=None)
    parser.add_argument('--value_model_cls', type=str, default='MLP')
    parser.add_argument('--value_num_hidden_layers', type=int, default=2)
    parser.add_argument('--value_num_hidden_units', type=int, default=256)
    parser.add_argument('--value_hidden_activation', type=str, default='elu')
    parser.add_argument('--value_lr_schedule', type=list, default=[8e-5, 100000, 8e-6])
    parser.add_argument('--policy_model_cls', type=str, default='MLP')
    parser.add_argument('--policy_num_hidden_layers', type=int, default=2)
    parser.add_argument('--policy_num_hidden_units', type=int, default=256)
    parser.add_argument('--policy_hidden_activation', type=str, default='elu')
    parser.add_argument('--policy_out_activation', type=str, default='linear')
    parser.add_argument('--policy_lr_schedule', type=list, default=[3e-5, 100000, 3e-6])
    parser.add_argument('--alpha', default=0.03)  # 'auto' 0.02
    alpha = parser.parse_args().alpha
    if alpha == 'auto':
        parser.add_argument('--target_entropy', type=float, default=-2)
    parser.add_argument('--alpha_lr_schedule', type=list, default=[8e-5, 100000, 8e-6])
    parser.add_argument('--policy_only', type=bool, default=False)
    parser.add_argument('--double_Q', type=bool, default=True)
    parser.add_argument('--target', type=bool, default=True)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--delay_update', type=int, default=1)
    parser.add_argument('--deterministic_policy', type=bool, default=False)
    parser.add_argument('--action_range', type=float, default=None)

    # preprocessor
    parser.add_argument('--obs_ptype', type=str, default='scale')
    num_future_data = parser.parse_args().num_future_data
    parser.add_argument('--obs_scale', type=list, default=[1., 1., 2., 1., 2.4, 1 / 1200] + [1.] * num_future_data)
    parser.add_argument('--rew_ptype', type=str, default='scale')
    parser.add_argument('--rew_scale', type=float, default=0.01)
    parser.add_argument('--rew_shift', type=float, default=0.)

    # Optimizer (PABAL)
    parser.add_argument('--max_sampled_steps', type=int, default=0)
    parser.add_argument('--max_iter', type=int, default=100000)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--num_learners', type=int, default=2)
    parser.add_argument('--num_buffers', type=int, default=2)
    parser.add_argument('--max_weight_sync_delay', type=int, default=300)
    parser.add_argument('--grads_queue_size', type=int, default=20)
    parser.add_argument('--eval_interval', type=int, default=3000)
    parser.add_argument('--save_interval', type=int, default=3000)
    parser.add_argument('--log_interval', type=int, default=100)

    # IO
    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    results_dir = './results/SAC/experiment-{time}'.format(time=time_now)
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
    elif alg_name == 'MPG-v1':
        args = built_MPG_parser('MPG-v1')
    elif alg_name == 'MPG-v2':
        args = built_MPG_parser('MPG-v2')
    elif alg_name == 'MPG-v3':
        args = built_MPG_parser('MPG-v3')
    elif alg_name == 'NDPG':
        args = built_NDPG_parser()
    elif alg_name == 'NADP':
        args = built_NADP_parser()
    elif alg_name == 'AMPC':
        args = built_AMPC_parser()
    env = gym.make(args.env_id)
    args.obs_dim, args.act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    return args


def main(alg_name):
    args = built_parser(alg_name)
    logger.info('begin training agents with parameter {}'.format(str(args)))
    if args.mode == 'training':
        ray.init(object_store_memory=5120*1024*1024)
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
    main('NDPG')
