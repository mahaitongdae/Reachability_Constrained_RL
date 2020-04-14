from policy import PolicyWithQs
from optimizer import AllReduceOptimizer
from mixed_pg_learner import MixedPGLearner
from trainer import Trainer
import argparse
import logging
import os
import datetime
import json
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

# logging.getLogger().setLevel(logging.INFO)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def built_mixedpg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg_name", default='Mixed_PG')
    parser.add_argument("--env_id", default='PathTracking-v0')
    parser.add_argument('--off_policy', default=False, action='store_true')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--max_sampled_steps', type=int, default=1000000)
    parser.add_argument('--max_updated_steps', type=int, default=1000000)
    parser.add_argument('--sample_n_step', type=int, default=64)
    parser.add_argument('--num_agent', type=int, default=256)

    parser.add_argument("--mini_batch_size", type=int, default=256)
    parser.add_argument("--policy_lr_schedule", type=list,
                        default=[0.0003, int(parser.parse_args().max_updated_steps / 2), 0.0003])
    parser.add_argument("--value_lr_schedule", type=list,
                        default=[0.0008, int(parser.parse_args().max_updated_steps/2), 0.0008])
    parser.add_argument("--gradient_clip_norm", type=float, default=3)
    parser.add_argument("--model_based", default=True, action='store_true')
    parser.add_argument("--deterministic_policy", default=True, action='store_true')

    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=100)

    parser.add_argument('--Q_num', type=int, default=1)
    parser.add_argument('--delay_update', type=int, default=1)
    parser.add_argument('--tau', type=float, default=0.005)

    parser.add_argument('--obs_dim', default=None)
    parser.add_argument('--act_dim', default=None)
    parser.add_argument("--obs_preprocess_type", type=str, default='scale')
    parser.add_argument("--obs_scale_factor", type=list, default=[0.2, 1., 2., 1., 2.4, 2., 0.4])
    parser.add_argument("--reward_preprocess_type", type=str, default=None)
    parser.add_argument("--reward_scale_factor", type=float, default=0.01)

    parser.add_argument('--policy_type', type=str, default='PolicyWithQs')
    parser.add_argument('--buffer_type', type=str, default='None')
    parser.add_argument('--optimizer_type', type=str, default='AllReduce')

    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    results_dir = './results/mixed_pg/experiment-{time}'.format(time=time_now)
    parser.add_argument("--result_dir", type=str, default=results_dir)

    parser.add_argument("--log_dir", type=str, default=results_dir + '/logs')
    parser.add_argument("--model_dir", type=str, default=results_dir + '/models')

    parser.add_argument("--evaluate_epi_num", type=int, default=2)
    return parser.parse_args()


def main():
    args = built_mixedpg_parser()
    logger.info('begin training mixed pg agents with parameter {}'.format(str(args)))
    os.makedirs(args.result_dir)
    with open(args.result_dir + '/config.json', 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)
    trainer = Trainer(policy_cls=PolicyWithQs,
                      learner_cls=MixedPGLearner,
                      buffer_cls=None,
                      optimizer_cls=AllReduceOptimizer,
                      args=args)
    trainer.train()


if __name__ == '__main__':
    main()
