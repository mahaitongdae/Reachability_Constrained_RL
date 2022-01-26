import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

ALG2CMAP = dict([('RAC (ours)', 'Blue'),
                 ('SAC-Lagrangian', 'Green'),
                 ('SAC-Reward Shaping', 'orange'),
                 ('SAC-CBF', 'salmon'),
                 ('SAC-SI', 'orchid')])

params={'font.family': 'Arial',
        # 'font.serif': 'Times New Roman',
        # 'font.style': 'italic',
        # 'font.weight': 'normal', #or 'blod'
        'font.size': 15,  # or large,small
        }
rcParams.update(params)

def plt_trajectory(ax, alg, trj_dir, episode=None):
    coordinates_list = np.load(trj_dir + '/coordinates_x_z.npy', allow_pickle=True)
    epi_num = len(coordinates_list)

    for i in range(epi_num):
        coor_dict_i = coordinates_list[i]
        xs = coor_dict_i['x']
        zs = coor_dict_i['z']
        print(xs[0], zs[0])
        trj = ax.plot(xs, zs, c=ALG2CMAP[alg],
                      linewidth=2 if alg!='RAC (ours)' else 4,
                      label=alg,)
        break
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
    ax.plot(x, z_ub, marker='.', c='r', label='Constraints')

def plt_PID_baseline(ax):
    # PID uncstr baseline
    PID_trj = np.load('../baseline/PID_traj.npy', allow_pickle=True)
    PID_ref = np.load('../baseline/PID_ref.npy', allow_pickle=True)
    PID_trj_plot = ax.plot(PID_trj[:, 0], PID_trj[:, 1], linewidth=2, label='PID-baseline',
                           c='violet')
    # PID_ref_plot = ax.plot(PID_ref[:, 0], PID_ref[:, 1], linewidth=2, label='PID',
    #                        c='Blues')


if __name__ == '__main__':
    fig = plt.figure(figsize=[9, 7])
    ax = plt.axes([0.1, 0.25, 0.8, 0.7])
    # # FSAC-Qc
    # plt_trajectory('../results/quadrotor/FSAC-Qc/2021-12-23-22-39-21/logs/tester/test-2021-12-24-13-27-39/')

    # FSAC-Qc change sample rules
    # plt_trajectory('../results/quadrotor/FSAC-Qc/2021-12-24-12-35-56/logs/tester/test-2021-12-24-16-17-45/')  # 1M
    # plt_trajectory('../results/quadrotor/FSAC-Qc/2021-12-24-12-35-56/logs/tester/test-2021-12-24-16-20-46/')  # 0.95M

    # RAC
    trj_RAC = plt_trajectory(ax,
                             'RAC (ours)',
                             '../results/quadrotor/RAC-feasibility/data2plot/2022-01-21-21-19-42/logs/tester/test-2022-01-22-11-14-25')

    # SAC-L
    trj_SACL = plt_trajectory(ax,
                             'SAC-Lagrangian',
                              '../results/quadrotor/SAC-Lagrangian-Dist/2022-01-18-07-51-09/logs/tester/test-2022-01-22-11-52-41')

    # SAC-Reward Shaping
    trj_SACRS = plt_trajectory(ax,
                              'SAC-Reward Shaping',
                               '../results/quadrotor/SAC-RewardShaping-Qc/2022-01-18-15-59-47/logs/tester/test-2022-01-22-11-51-45')

    # SAC-CBF
    trj_SACCBF = plt_trajectory(ax,
                               'SAC-CBF',
                               '../results/quadrotor/SAC-CBF-CBF/2022-01-19-00-28-04/logs/tester/test-2022-01-22-12-10-45')

    # SAC-Energy
    trj_SACenergy = plt_trajectory(ax,
                                   'SAC-SI',
                                   '../results/quadrotor/FSAC-A-si/2022-01-21-21-53-18/logs/tester/test-2022-01-22-12-11-01')

    # # SAC-uncstr
    # plt_trajectory('../results/quadrotor/SAC/experiment-2021-12-27-22-26-02/logs/tester/test-2021-12-28-10-53-49')

    # Plot constraint and ref
    ref = plt_ref_trj(ax)
    plt_constraint(ax)
    # plt_PID_baseline(ax)

    # Plot settings
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.2, 2.2)

    # legend1 = ax.legend(*trj_RAC.legend_elements(),
    #                     loc="lower left", title="Algorithms")
    ax.legend(frameon=False, fontsize=12,
              bbox_to_anchor=(0.5, -0.25), loc='lower center', ncol=4)
    plt.title('Quadrotor Tracking Trajectories Visualization', fontsize=12)
    plt.tight_layout(pad=0.5)
    plt.show()