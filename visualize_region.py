from policy import Policy4Lagrange
import os
from evaluator import Evaluator
import gym
from utils.em_brake_4test import EmergencyBraking
import numpy as np
from matplotlib.colors import ListedColormap
from dynamics.models import EmBrakeModel, UpperTriangleModel, Air3dModel


def hj_baseline(timet=5.0):
    import jax
    import jax.numpy as jnp
    import hj_reachability as hj
    # from hj_reachability.systems import DetAir3d
    dynamics = hj.systems.DoubleInt()

    grid = hj.Grid.from_grid_definition_and_initial_values(hj.sets.Box(lo=np.array([-5., -5.]),
                                                                       hi=np.array([5., 5.])), (50, 50))
    values = - jnp.linalg.norm(grid.states, axis=-1, ord=jnp.inf) + 5

    solver_settings = hj.SolverSettings.with_accuracy("very_high",
                                                      hamiltonian_postprocessor=hj.solver.backwards_reachable_tube)
    time = 0.
    target_time = -timet
    target_values = hj.step(solver_settings, dynamics, grid, time, values, target_time).block_until_ready()
    return grid, target_values

def static_region(test_dir, iteration,
                  bound=(-5., 5., -5., 5.),
                  sum=True,
                  vector=False,
                  baseline=False):
    import json
    import argparse
    import datetime
    from policy import Policy4Lagrange
    params = json.loads(open(test_dir + '/config.json').read())
    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    test_log_dir = params['log_dir'] + '/tester/test-region-{}'.format(time_now)
    params.update(dict(mode='testing',
                       test_dir=test_dir,
                       test_log_dir=test_log_dir,))
    parser = argparse.ArgumentParser()
    for key, val in params.items():
        parser.add_argument("-" + key, default=val)
    args = parser.parse_args()
    evaluator = Evaluator(Policy4Lagrange, args.env_id, args)
    evaluator.load_weights(os.path.join(test_dir, 'models'), iteration)
    brake_model = EmBrakeModel()
    double_intergrator_model = UpperTriangleModel()
    air3d_model = Air3dModel()
    model_dict = {"UpperTriangle": double_intergrator_model,
                  "Air3d": air3d_model}
    model = model_dict.get(args.env_id.split("-")[0])


    # generate batch obses
    d = np.linspace(bound[0], bound[1], 400)
    v = np.linspace(bound[2], bound[3], 400)
    # cmaplist = ['springgreen'] * 3 + ['crimson'] * 87
    # cmap1 = ListedColormap(cmaplist)
    D, V = np.meshgrid(d, v)
    flatten_d = np.reshape(D, [-1, ])
    flatten_v = np.reshape(V, [-1, ])
    env_name = args.env_id.split("-")[0]
    if env_name == 'Air3d':
        x3 = np.pi * np.ones_like(flatten_d)
        init_obses = np.stack([flatten_d, flatten_v, x3], 1)
    else:
        init_obses = np.stack([flatten_d, flatten_v], 1)

    # define rollout
    def reduced_model_rollout_for_update(obses):
        model.reset(obses)
        constraints_list = []
        for step in range(args.num_rollout_list_for_policy_update[0]):
            processed_obses = evaluator.preprocessor.tf_process_obses(obses)
            actions, _ = evaluator.policy_with_value.compute_action(processed_obses)
            obses, rewards, constraints = model.rollout_out(actions)
            constraints = evaluator.tf.expand_dims(constraints, 1) if len(constraints.shape) == 1 else constraints
            constraints_list.append(constraints)
        flattern_cstr = evaluator.tf.concat(constraints_list, 1).numpy()
        return flattern_cstr
    flatten_cstr = reduced_model_rollout_for_update(init_obses)

    preprocess_obs = evaluator.preprocessor.np_process_obses(init_obses)
    flatten_mu = evaluator.policy_with_value.compute_mu(preprocess_obs).numpy()
    flatten_cstr = np.clip(flatten_cstr, 0, np.inf)

    if vector:
        flatten_cs = np.multiply(flatten_cstr, flatten_mu)
    else:
        con_dim = -args.con_dim
        flatten_cstr = flatten_cstr[:, con_dim:]
        flatten_cs = np.multiply(flatten_cstr, flatten_mu)
    flatten_cs = flatten_cs / np.max(flatten_cs)

    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 16})
    from mpl_toolkits.mplot3d import Axes3D

    plot_items = ['cs']
    data_dict = {'cs': flatten_cs, 'mu':flatten_mu, 'cstr': flatten_cstr}
    if baseline:
        grid, target_values = hj_baseline()
        grid1, target_values1 = hj_baseline(timet=10.0)

    def plot_region(data_reshape, name):
        fig = plt.figure(figsize=[5,6])
        ax = plt.axes([0.1,0.2,0.8,0.75])
        data_reshape += 0.15 * np.where(data_reshape == 0,
                                        np.zeros_like(data_reshape),
                                        np.ones_like(data_reshape))
        ct1 = ax.contourf(D, V, data_reshape, cmap='Accent')  # 50
        # plt.colorbar(ct1)
        ct1.collections[0].set_label('Learned Boundary')
        ax.contour(D, V, data_reshape, levels=0,
                   colors="green",
                   linewidths=3)
        if baseline:
            ct2 = ax.contour(grid.coordinate_vectors[0],
                       grid.coordinate_vectors[1],
                       target_values.T,
                       levels=0,
                       colors="grey",
                       linewidths=3)

            data = np.load('/home/mahaitong/PycharmProjects/toyota_exp_train (copy)/baseline/init_feasible_f1.npy')
            data2 = np.load('/home/mahaitong/PycharmProjects/toyota_exp_train (copy)/baseline/init_feasible_f0.4.npy')
            ds = np.linspace(bound[0], bound[1], 100)
            vs = np.linspace(bound[2], bound[3], 100)
            Ds, Vs = np.meshgrid(ds, vs)
            ct3 = ax.contour(Ds,
                             Vs,
                             data.T,
                             levels=0,
                             colors="cornflowerblue",
                             linewidths=3)
            ct2 = ax.contour(Ds,
                             Vs,
                             data2.T,
                             levels=0,
                             colors="orange",
                             linewidths=3)
            # ct2.collections[0].set_label('HJ-Reachability Boundary')
        name_2d = name + '_' + str(iteration) + '_2d.jpg'
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        rect1 = plt.Rectangle((0, 0), 1, 1, fc=ct1.collections[0].get_facecolor()[0], ec='green', linewidth=3)
        rect2 = plt.Rectangle((0, 0), 1, 1, fill=False, ec='grey', linewidth=3)
        rect3 = plt.Rectangle((0, 0), 1, 1, fill=False, ec='orange', linewidth=3)
        rect4 = plt.Rectangle((0, 0), 1, 1, fill=False, ec='cornflowerblue', linewidth=3)
        ax = plt.axes([0.05, 0.02, 0.9, 0.16])
        plt.axis('off')
        ax.legend((rect1,rect2, rect3, rect4), ('Feasible region', 'HJ avoid set', 'Energy-based','MPC-feasiblity')
                   , loc='lower center',ncol=2, fontsize=15)
        # plt.title('Feasible Region of Double Integrator')
        plt.tight_layout(pad=0.5)
        plt.savefig(os.path.join(evaluator.log_dir, name_2d))
        # figure = plt.figure()
        # ax = Axes3D(figure)
        # ax.plot_surface(D, V, data_reshape, rstride=1, cstride=1, cmap='rainbow')
        # name_3d = name + '_' + str(iteration) + '_3d.jpg'
        # plt.savefig(os.path.join(evaluator.log_dir, name_3d))


    for plot_item in plot_items:
        data = data_dict.get(plot_item)
        # for k in range(data.shape[1]):
        #     data_k = data[:, k]
        #     data_reshape = data_k.reshape(D.shape)
        #     plot_region(data_reshape, plot_item + '_' + str(k))

        if sum:
            data_k = np.sum(data, axis=1)
            data_reshape = data_k.reshape(D.shape)
            plot_region(data_reshape, plot_item + '_sum')



if __name__ == '__main__':
    # static_region('./results/toyota3lane/LMAMPC-v2-2021-11-21-23-04-21', 300000)
    # static_region('./results/Air3d/LMAMPC-vector-2021-12-02-01-41-12', 300000,
    #               bound=(-6., 20., -10., 10.),
    #               baseline=True) #
    # LMAMPC - vector - 2021 - 11 - 29 - 21 - 22 - 40
    static_region('./results/uppep_triangle/LMAMPC-terminal-2021-12-02-23-07-17', 300000,
                  bound=(-5., 5., -5., 5.),
                  vector=False,
                  baseline=True)  #
