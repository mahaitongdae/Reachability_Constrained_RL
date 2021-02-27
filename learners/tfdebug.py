import tensorflow as tf
from learners.ampc import AMPCLearner
from policy import Policy4Lagrange
import gym
from train_script import built_LMAMPC_parser
from learners.ampc_lag import LMAMPCLearner
from buffer import ReplayBuffer
from optimizer import OffPolicyAsyncOptimizer
from tester import Tester
from trainer import Trainer
from worker import OffPolicyWorker
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# a = tf.Variable(3.0)
# with tf.GradientTape() as g:
#     s = a * a
# ds_dx = g.gradient(s, a)
# print(ds_dx)
# if a > 2:
#     print('t')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
NAME2WORKERCLS = dict([('OffPolicyWorker', OffPolicyWorker)])
NAME2LEARNERCLS = dict([('LMAMPC', LMAMPCLearner)])
NAME2BUFFERCLS = dict([('normal', ReplayBuffer), ('None', None)])
NAME2OPTIMIZERCLS = dict([('OffPolicyAsync', OffPolicyAsyncOptimizer)])
NAME2POLICIES = dict([('Policy4Lagrange', Policy4Lagrange)])
# NAME2EVALUATORS = dict([('Evaluator', Evaluator), ('None', None)])

args = built_LMAMPC_parser()
args.replay_starts = 100
args.obs_dim = 37 #left
args.act_dim = 4
args.obs_scale = [0.2, 1., 2., 1 / 50., 1 / 50, 1 / 180.] + \
                         [1., 1 / 15., 0.2] + \
                         [1., 1., 1 / 15.] * args.env_kwargs_num_future_data + \
                         [1 / 50., 1 / 50., 0.2, 1 / 180.] * 7
local_worker = OffPolicyWorker(Policy4Lagrange, args.env_id, args, 0)
# virtual_worker = OffPolicyWorker(PolicyWithQs, args.env_id, args, 1)
local_buffer = ReplayBuffer(args,0)


count = 0
# sampling
while count < 100:
    batch_data, batch_count = local_worker.sample_with_count()
    local_buffer.add_batch(batch_data)
    count += batch_count
# weights = local_worker.get_weights()
# getFlat = U.GetFlat(tf.convert_to_tensor(weights))
# flat_weights = getFlat()
del local_worker
print('learner start')
local_learner = LMAMPCLearner(Policy4Lagrange, args)
# ppc_params = virtual_worker.get_ppc_params()
# weights = np.load('weights.npy',allow_pickle=True).tolist()
# local_learner.set_weights(weights)

samples = local_buffer.replay()
# obs = np.load('obs.npy')
# samples[0] = obs
policy_gradient = local_learner.compute_gradient(samples, local_buffer, samples[-1], 100)
print(policy_gradient)
local_learner.get_batch_data(samples, local_buffer, samples[-1])
mb_obs = local_learner.batch_data['batch_obs']
print(mb_obs.shape)

local_worker = OffPolicyWorker(Policy4Lagrange, args.env_id, args, 0)
local_worker.apply_gradients(0, policy_gradient)
    # local_learner.policy_with_value.apply_gradients(0, policy_gradient)
    # print('success')

# fill in buffer




