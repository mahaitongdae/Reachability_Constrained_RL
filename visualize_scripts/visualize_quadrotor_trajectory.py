import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plt_trajectory(trj_dir, episode=None):
    coordinates_list = np.load(trj_dir + 'coordinates_x_z.npy', allow_pickle=True)
    epi_num = len(coordinates_list)

    fig = plt.figure(figsize=[5, 5])
    ax = plt.axes([0.1, 0.2, 0.8, 0.75])

    for i in range(epi_num):
        coor_dict_i = coordinates_list[i]
        xs = coor_dict_i['x']
        zs = coor_dict_i['z']
        print(xs)
        trj = ax.scatter(xs, zs, s=5., label='ep'+str(i),
                         c=range(len(xs)),
                         linewidth=1,
                         cmap='viridis')
        plt.colorbar(trj)

    ref = plt_ref_trj()
    ax.add_artist(ref)
    plt_constraint(ax)

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$z$')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-3, 3)
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

    # # FSAC-uncstr
    plt_trajectory('../results/quadrotor/FSAC-Qc/2021-12-23-22-39-21/logs/tester/test-2021-12-24-16-05-04/')