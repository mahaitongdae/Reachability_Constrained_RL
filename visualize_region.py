import json
import argparse
import datetime
from train_script import built_LMAMPC_parser
from policy import Policy4Lagrange
from evaluator import Evaluator
import os

import numpy as np
from matplotlib.colors import ListedColormap



def static_region(test_dir, iteration):

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
    d = np.linspace(0,10,100)
    v = np.linspace(0,10,100)
    cmaplist = ['springgreen'] * 5 + ['gold'] * 10 + ['crimson'] * 85
    cmap1 = ListedColormap(cmaplist)
    D, V = np.meshgrid(d, v)
    flattenD = np.reshape(D, [-1,])
    flattenV = np.reshape(V, [-1,])
    obses = np.stack([flattenD, flattenV], 1)
    preprocess_obs = evaluator.preprocessor.np_process_obses(obses)
    flattenMU = evaluator.policy_with_value.compute_mu(preprocess_obs).numpy() / 100
    for k in [flattenMU.shape[1]-1]:
        flattenMU_k = flattenMU[:, k]
        mu = flattenMU_k.reshape(D.shape)
        def plot_region(z, name):
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            plt.figure()
            z += np.clip((V**2-10*D)/100, 0, 1)
            ct = plt.contourf(D,V,z,50, cmap=cmap1)
            plt.colorbar(ct)
            plt.grid()
            x = np.linspace(0, 10)
            t = np.sqrt(2 * 5 * x)
            plt.plot(x, t, linestyle='--', color='red')
            # plt.plot(d, np.sqrt(2*5*d),lw=2)
            name_2d=name + '_2d.jpg'
            plt.savefig(os.path.join(evaluator.log_dir, name_2d))

            figure = plt.figure()
            ax = Axes3D(figure)
            ax.plot_surface(D, V, z, rstride=1, cstride=1, cmap='rainbow')
            # plt.show()
            name_3d = name + '_3d.jpg'
            print(os.path.join(evaluator.log_dir,name_3d))
            plt.savefig(os.path.join(evaluator.log_dir,name_3d))
        plot_region(mu, str(k))

if __name__ == '__main__':
    static_region('./results/toyota3lane/experiment-2021-03-15-08-44-37', 1000000)