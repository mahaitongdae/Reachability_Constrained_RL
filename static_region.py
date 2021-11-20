from policy import Policy4Lagrange
import os
from evaluator import Evaluator


def static_region(test_dir, iteration):
    import json
    import argparse
    import datetime
    from train_script import built_LMAMPC_parser
    from policy import Policy4Lagrange
    # args = built_LMAMPC_parser()
    params = json.loads(open(test_dir + '/config.json').read())
    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    test_log_dir = params['log_dir'] + '/tester/test-region-{}'.format(time_now)
    params.update(dict(mode='testing',
                       test_dir=test_dir,
                       test_log_dir=test_log_dir,))
    parser = argparse.ArgumentParser()
    for key, val in params.items():
        parser.add_argument("-" + key, default=val)
    args =  parser.parse_args()
    evaluator = Evaluator(Policy4Lagrange, args.env_id, args)
    evaluator.load_weights(os.path.join(test_dir, 'models'), iteration)
    evaluator.static_region()

if __name__ == '__main__':
    static_region('./results/toyota3lane/LMAMPC-v2-2021-06-24-14-10-01', 200000)