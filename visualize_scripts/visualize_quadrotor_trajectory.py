import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ALG2CMAP = dict([('RAC (ours)', 'Blue'),
                 ('SAC-L', 'Green'),
                 ('SAC-Reward Shaping', 'orange'),
                 ('SAC-CBF', 'salmon'),
                 ('SAC-Energy', 'orchid')])

def plt_trajectory(ax, alg, trj_dir, episode=None):
    coordinates_list = np.load(trj_dir + '/coordinates_x_z.npy', allow_pickle=True)
    epi_num = len(coordinates_list)

    for i in range(epi_num):
        coor_dict_i = coordinates_list[i]
        xs = coor_dict_i['x']
        zs = coor_dict_i['z']
        print(xs[0], zs[0])
        trj = ax.plot(xs, zs, c=ALG2CMAP[alg],
                      linewidth=2,
                      label=alg,)
    # plt.colorbar(trj)
    # plt.colorbar(PID_trj_plot)
    # plt.colorbar(PID_ref_plot)

    return trj


def plt_ref_trj(ax):
    theta = np.linspace(0, 2 * np.pi, 100)
    radius = 1.
    offset_x = 0.
    offset_z = 1.
    x = radius * np.cos(theta) + offset_x
    z = radius * np.sin(theta) + offset_z

    ref = ax.plot(x, z, c='Black', label='Reference', linewidth=2.,
                  ls='--', markersize=0.1)

    return ref


def plt_constraint(ax):
    x = np.linspace(-4, 4, 100)
    z_lb = 0.5 * np.ones_like(x)
    z_ub = 1.5 * np.ones_like(x)

    ax.plot(x, z_lb, marker='.', c='r')
    ax.plot(x, z_ub, marker='.', c='r')

def plt_PID_baseline(ax):
    # PID uncstr baseline
    PID_trj = np.load('../baseline/PID_traj.npy', allow_pickle=True)
    PID_ref = np.load('../baseline/PID_ref.npy', allow_pickle=True)
    PID_trj_plot = ax.plot(PID_trj[:, 0], PID_trj[:, 1], linewidth=2, label='PID-baseline',
                           c='violet')
    # PID_ref_plot = ax.plot(PID_ref[:, 0], PID_ref[:, 1], linewidth=2, label='PID',
    #                        c='Blues')


if __name__ == '__main__':
    fig = plt.figure(figsize=[5, 5])
    ax = plt.axes([0.1, 0.2, 0.8, 0.75])
    # # FSAC-Qc
    # plt_trajectory('../results/quadrotor/FSAC-Qc/2021-12-23-22-39-21/logs/tester/test-2021-12-24-13-27-39/')

    # FSAC-Qc change sample rules
    # plt_trajectory('../results/quadrotor/FSAC-Qc/2021-12-24-12-35-56/logs/tester/test-2021-12-24-16-17-45/')  # 1M
    # plt_trajectory('../results/quadrotor/FSAC-Qc/2021-12-24-12-35-56/logs/tester/test-2021-12-24-16-20-46/')  # 0.95M

    # # RAC
    # trj_RAC = plt_trajectory(ax,
    #                          'RAC (ours)',
    #                          '../results/quadrotor/RAC-feasibility/2021-12-30-13-00-03-Zero_violation/logs/tester/test-2021-12-30-21-01-46')
    #
    # # SAC-L
    # trj_SACL = plt_trajectory(ax,
    #                          'SAC-L',
    #                           '../results/quadrotor/SAC-Lagrangian-Qc/2021-12-29-20-54-24_success/logs/tester/test-2021-12-30-21-31-13')
    #
    # # SAC-Reward Shaping
    # trj_SACRS = plt_trajectory(ax,
    #                           'SAC-Reward Shaping',
    #                            '../results/quadrotor/SAC-RewardShaping-Qc/2021-12-30-19-52-12_success/logs/tester/test-2021-12-31-11-02-25')

    # SAC-CBF
    trj_SACCBF = plt_trajectory(ax,
                               'SAC-CBF',
                               '../results/quadrotor/SAC-CBF-CBF/2022-01-05-11-34-37/logs/tester/test-2022-01-07-11-10-26')

    # SAC-Energy
    trj_SACenergy = plt_trajectory(ax,
                                   'SAC-Energy',
                                   '../results/quadrotor/FSAC-A-si/2022-01-04-20-43-11/logs/tester/test-2022-01-07-11-10-26')

    # # SAC-uncstr
    # plt_trajectory('../results/quadrotor/SAC/experiment-2021-12-27-22-26-02/logs/tester/test-2021-12-28-10-53-49')

    # Plot constraint and ref
    ref = plt_ref_trj(ax)
    plt_constraint(ax)
    # plt_PID_baseline(ax)

    # Plot settings
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$z$')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-0, 2)

    # legend1 = ax.legend(*trj_RAC.legend_elements(),
    #                     loc="lower left", title="Algorithms")
    ax.legend(loc='upper left')
    plt.show()