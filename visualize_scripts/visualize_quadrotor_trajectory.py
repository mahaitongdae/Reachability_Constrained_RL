import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plt_trajectory(trj_dir, episode=None):
    coordinates_list = np.load(trj_dir + '/coordinates_x_z.npy', allow_pickle=True)
    epi_num = len(coordinates_list)

    fig = plt.figure(figsize=[5, 5])
    ax = plt.axes([0.1, 0.2, 0.8, 0.75])

    for i in range(epi_num):
        coor_dict_i = coordinates_list[i]
        xs = coor_dict_i['x']
        zs = coor_dict_i['z']
        print(xs[0], zs[0])
        trj = ax.scatter(xs, zs, s=5., label='ep'+str(i),
                         c=range(len(xs)),
                         linewidth=1,
                         cmap='Greens')
    plt.colorbar(trj)

    ref = plt_ref_trj()
    ax.add_artist(ref)
    plt_constraint(ax)

    # # PID uncstr baseline
    # PID_trj = np.load('../baseline/PID_traj.npy', allow_pickle=True)
    # PID_ref = np.load('../baseline/PID_ref.npy', allow_pickle=True)
    # PID_trj_plot = ax.scatter(PID_trj[:, 0], PID_trj[:, 1], s=5, label='PID', c=range(PID_trj.shape[0]),
    #                           cmap='Oranges')
    # PID_ref_plot = ax.scatter(PID_ref[:, 0], PID_ref[:, 1], s=5, label='PID', c=range(PID_ref.shape[0]),
    #                           cmap='Blues')
    # plt.colorbar(PID_trj_plot)
    # plt.colorbar(PID_ref_plot)

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$z$')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1, 3)
    plt.show()


def plt_ref_trj():
    ref = plt.Circle((0., 1.), 1., fill=False)
    return ref


def plt_constraint(ax):
    x = np.linspace(-4, 4, 100)
    z_lb = 0.5 * np.ones_like(x)
    z_ub = 1.5 * np.ones_like(x)

    ax.plot(x, z_lb, marker='.', c='r')
    ax.plot(x, z_ub, marker='.', c='r')


if __name__ == '__main__':
    # # FSAC-Qc
    # plt_trajectory('../results/quadrotor/FSAC-Qc/2021-12-23-22-39-21/logs/tester/test-2021-12-24-13-27-39/')

    # FSAC-Qc change sample rules
    # plt_trajectory('../results/quadrotor/FSAC-Qc/2021-12-24-12-35-56/logs/tester/test-2021-12-24-16-17-45/')  # 1M
    # plt_trajectory('../results/quadrotor/FSAC-Qc/2021-12-24-12-35-56/logs/tester/test-2021-12-24-16-20-46/')  # 0.95M

    # RAC
    plt_trajectory('../results/quadrotor/RAC-feasibility/2021-12-26-20-04-19/logs/tester/test-2021-12-27-16-34-00')

    # # SAC-uncstr
    # plt_trajectory('../results/quadrotor/SAC/experiment-2021-12-27-22-26-02/logs/tester/test-2021-12-28-10-53-49')