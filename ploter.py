#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/9/25
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: ploter.py
# =====================================

import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.core.util import event_pb2

sns.set(style="darkgrid")
SMOOTHFACTOR = 0.8


def help_func(env):
    if env == 'path_tracking_env':
        tag2plot = ['episode_return', 'episode_len', 'delta_y_mse', 'delta_phi_mse', 'delta_v_mse',
                    'stationary_rew_mean', 'steer_mse', 'acc_mse']
        alg_list = ['MPG-v3', 'MPG-v2', 'NDPG', 'NADP', 'TD3', 'SAC']
        lbs = ['MPG-v1', 'MPG-v2', r'$n$-step DPG', r'$n$-step ADP', 'TD3', 'SAC']
        palette = "bright"
        goal_perf_list = [-200, -100, -50, -30, -20, -10, -5]
        dir_str = './results/{}/data2plot'
    else:
        tag2plot = ['episode_return', 'episode_len', 'x_mse', 'theta_mse', 'xdot_mse', 'thetadot_mse']
        alg_list = ['MPG-v2', 'NADP', 'TD3', 'SAC']
        lbs = ['MPG-v2', r'$n$-step ADP', 'TD3', 'SAC']
        palette = [(1.0, 0.48627450980392156, 0.0),
                    (0.9098039215686274, 0.0, 0.043137254901960784),
                    (0.5450980392156862, 0.16862745098039217, 0.8862745098039215),
                    (0.6235294117647059, 0.2823529411764706, 0.0),]
        goal_perf_list = [-20, -10, -2, -1, -0.5, -0.1, -0.01]
        dir_str = './results/{}/data2plot_mujoco'
    return tag2plot, alg_list, lbs, palette, goal_perf_list, dir_str


def plot_eval_results_of_all_alg_n_runs(env, dirs_dict_for_plot=None):
    tag2plot, alg_list, lbs, palette, _, dir_str = help_func(env)
    df_list = []
    df_in_one_run_of_one_alg = {}
    for alg in alg_list:
        data2plot_dir = dir_str.format(alg)
        data2plot_dirs_list = dirs_dict_for_plot[alg] if dirs_dict_for_plot is not None else os.listdir(data2plot_dir)
        for num_run, dir in enumerate(data2plot_dirs_list):
            eval_dir = data2plot_dir + '/' + dir + '/logs/evaluator'
            eval_file = os.path.join(eval_dir,
                                     [file_name for file_name in os.listdir(eval_dir) if file_name.startswith('events')][0])
            eval_summarys = tf.data.TFRecordDataset([eval_file])
            data_in_one_run_of_one_alg = {key: [] for key in tag2plot}
            data_in_one_run_of_one_alg.update({'iteration': []})
            for eval_summary in eval_summarys:
                event = event_pb2.Event.FromString(eval_summary.numpy())
                for v in event.summary.value:
                    t = tf.make_ndarray(v.tensor)
                    for tag in tag2plot:
                        if tag == v.tag[11:]:
                            data_in_one_run_of_one_alg[tag].append((1-SMOOTHFACTOR)*data_in_one_run_of_one_alg[tag][-1] + SMOOTHFACTOR*float(t)
                                                                   if data_in_one_run_of_one_alg[tag] else float(t))
                            data_in_one_run_of_one_alg['iteration'].append(int(event.step))
            len1, len2 = len(data_in_one_run_of_one_alg['iteration']), len(data_in_one_run_of_one_alg[tag2plot[0]])
            period = int(len1/len2)
            data_in_one_run_of_one_alg['iteration'] = [data_in_one_run_of_one_alg['iteration'][i*period]/10000. for i in range(len2)]

            data_in_one_run_of_one_alg.update(dict(algorithm=alg, num_run=num_run))
            df_in_one_run_of_one_alg = pd.DataFrame(data_in_one_run_of_one_alg)
            df_list.append(df_in_one_run_of_one_alg)
    total_dataframe = df_list[0].append(df_list[1:], ignore_index=True) if len(df_list) > 1 else df_list[0]
    figsize = (20, 8)
    axes_size = [0.11, 0.11, 0.89, 0.89] if env == 'path_tracking_env' else [0.095, 0.11, 0.905, 0.89]
    fontsize = 25
    f1 = plt.figure(1, figsize=figsize)
    ax1 = f1.add_axes(axes_size)
    sns.lineplot(x="iteration", y="episode_return", hue="algorithm",
                 data=total_dataframe, linewidth=2, palette=palette,
                 )
    base = -30 if env == 'path_tracking_env' else -2
    basescore = sns.lineplot(x=[0., 10.], y=[base, base], linewidth=2, color='black', linestyle='--')
    print(ax1.lines[0].get_data())
    ax1.set_ylabel('Episode Return', fontsize=fontsize)
    ax1.set_xlabel("Iteration [x10000]", fontsize=fontsize)
    handles, labels = ax1.get_legend_handles_labels()
    labels = lbs
    ax1.legend(handles=handles+[basescore.lines[-1]], labels=labels+['Base score'], loc='lower right', frameon=False, fontsize=fontsize)
    lim = (-800, 50) if env == 'path_tracking_env' else (-60, 5)
    plt.xlim(0., 10.2)
    plt.ylim(*lim)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    if env == 'path_tracking_env':
        f2 = plt.figure(2, figsize=figsize)
        ax2 = f2.add_axes(axes_size)
        sns.lineplot(x="iteration", y="delta_y_mse", hue="algorithm",
                     data=total_dataframe, linewidth=2, palette=palette,
                     )
        ax2.set_ylabel('Position Error [m]', fontsize=fontsize)
        ax2.set_xlabel("Iteration [x10000]", fontsize=fontsize)
        handles, labels = ax2.get_legend_handles_labels()
        labels = lbs
        ax2.legend(handles=handles, labels=labels, loc='upper right', frameon=False, fontsize=fontsize)
        plt.xlim(0., 10.2)
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)

        f3 = plt.figure(3, figsize=figsize)
        ax3 = f3.add_axes(axes_size)
        sns.lineplot(x="iteration", y="delta_phi_mse", hue="algorithm",
                     data=total_dataframe, linewidth=2, palette=palette,
                     legend=False)
        ax3.set_ylabel('Heading Angle Error [rad]', fontsize=fontsize)
        ax3.set_xlabel("Iteration [x10000]", fontsize=fontsize)
        plt.xlim(0., 10.2)
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)

        f4 = plt.figure(4, figsize=figsize)
        ax4 = f4.add_axes(axes_size)
        sns.lineplot(x="iteration", y="delta_v_mse", hue="algorithm",
                     data=total_dataframe, linewidth=2, palette=palette,
                     legend=False)
        ax4.set_ylabel('Velocity Error [m/s]', fontsize=fontsize)
        ax4.set_xlabel("Iteration [x10000]", fontsize=fontsize)
        plt.xlim(0., 10.2)
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)

        f5 = plt.figure(5, figsize=figsize)
        ax5 = f5.add_axes(axes_size)
        sns.lineplot(x="iteration", y="steer_mse", hue="algorithm",
                     data=total_dataframe, linewidth=2, palette=palette,
                     )
        ax5.set_ylabel('Front Wheel Angle [rad]', fontsize=fontsize)
        ax5.set_xlabel("Iteration [x10000]", fontsize=fontsize)
        handles, labels = ax5.get_legend_handles_labels()
        labels = lbs
        ax5.legend(handles=handles, labels=labels, loc='upper right', frameon=False, fontsize=fontsize)
        plt.xlim(0., 10.2)
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)

        f6 = plt.figure(6, figsize=figsize)
        ax6 = f6.add_axes(axes_size)
        sns.lineplot(x="iteration", y="acc_mse", hue="algorithm",
                     data=total_dataframe, linewidth=2, palette=palette,
                     legend=False)
        ax6.set_ylabel('Acceleration [$m^2$/s]', fontsize=fontsize)
        ax6.set_xlabel("Iteration [x10000]", fontsize=fontsize)
        plt.xlim(0., 10.2)
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
    else:
        f2 = plt.figure(2, figsize=figsize)
        ax2 = f2.add_axes(axes_size)
        sns.lineplot(x="iteration", y="x_mse", hue="algorithm",
                     data=total_dataframe, linewidth=2, palette=palette,
                     )
        ax2.set_ylabel('Cart Position [m]', fontsize=fontsize)
        ax2.set_xlabel("Iteration [x10000]", fontsize=fontsize)
        handles, labels = ax2.get_legend_handles_labels()
        labels = lbs
        ax2.legend(handles=handles, labels=labels, loc='upper right', frameon=False, fontsize=fontsize)
        plt.xlim(0., 10.2)
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)

        f3 = plt.figure(3, figsize=figsize)
        ax3 = f3.add_axes(axes_size)
        sns.lineplot(x="iteration", y="theta_mse", hue="algorithm",
                     data=total_dataframe, linewidth=2, palette=palette,
                     legend=False)
        ax3.set_ylabel('Pole Angle [rad]', fontsize=fontsize)
        ax3.set_xlabel("Iteration [x10000]", fontsize=fontsize)
        plt.xlim(0., 10.2)
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)

        f4 = plt.figure(4, figsize=figsize)
        ax4 = f4.add_axes(axes_size)
        sns.lineplot(x="iteration", y="xdot_mse", hue="algorithm",
                     data=total_dataframe, linewidth=2, palette=palette,
                     legend=False)
        ax4.set_ylabel('Cart Velocity [m/s]', fontsize=fontsize)
        ax4.set_xlabel("Iteration [x10000]", fontsize=fontsize)
        plt.xlim(0., 10.2)
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)

        f5 = plt.figure(5, figsize=figsize)
        ax5 = f5.add_axes(axes_size)
        sns.lineplot(x="iteration", y="thetadot_mse", hue="algorithm",
                     data=total_dataframe, linewidth=2, palette=palette,
                     legend=False)
        ax5.set_ylabel('Pole Angular Velocity [rad/s]', fontsize=fontsize)
        ax5.set_xlabel("Iteration [x10000]", fontsize=fontsize)
        plt.xlim(0., 10.2)
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
    plt.show()
    allresults = {}
    results2print = {}

    for alg, group in total_dataframe.groupby('algorithm'):
        allresults.update({alg: []})
        for ite, group1 in group.groupby('iteration'):
            mean = group1['episode_return'].mean()
            std = group1['episode_return'].std()
            allresults[alg].append((mean, std))

    for alg, result in allresults.items():
        mean, std = sorted(result, key=lambda x: x[0])[-1]
        results2print.update({alg: [mean, 2 * std]})

    print(results2print)


def compute_convergence_speed(goal_perf, dirs_dict_for_plot=None):
    _, alg_list, _, _, _, dir_str = help_func(env)
    result_dict = {}
    for alg in alg_list:
        result_dict.update({alg: []})
        data2plot_dir = dir_str.format(alg)
        data2plot_dirs_list = dirs_dict_for_plot[alg] if dirs_dict_for_plot is not None else os.listdir(data2plot_dir)
        for num_run, dir in enumerate(data2plot_dirs_list):
            stop_flag = 0
            eval_dir = data2plot_dir + '/' + dir + '/logs/evaluator'
            eval_file = os.path.join(eval_dir,
                                     [file_name for file_name in os.listdir(eval_dir) if
                                      file_name.startswith('events')][0])
            eval_summarys = tf.data.TFRecordDataset([eval_file])

            for eval_summary in eval_summarys:
                if stop_flag != 1:
                    event = event_pb2.Event.FromString(eval_summary.numpy())
                    for v in event.summary.value:
                        if stop_flag != 1:
                            t = tf.make_ndarray(v.tensor)
                            step = float(event.step)
                            if 'episode_return' in v.tag:
                                if t > goal_perf:
                                    result_dict[alg].append(step)
                                    stop_flag = 1
            if stop_flag == 0:
                result_dict[alg].append(np.inf)
    return result_dict


def min_n(inp_list, n):
    return sorted(inp_list)[:n]


def plot_convergence_speed_for_different_goal_perf(env):
    _, _, lbs, palette, goal_perf_list, dir_str = help_func(env)
    result2print = {}
    df_list = []
    for goal_perf in goal_perf_list:
        result2print.update({goal_perf: dict()})
        result_dict_for_this_goal_perf = compute_convergence_speed(goal_perf)
        for alg in result_dict_for_this_goal_perf:
            first_arrive_steps_list = result_dict_for_this_goal_perf[alg]
            df_for_this_alg_this_goal = pd.DataFrame(dict(algorithm=alg,
                                                          goal_perf=str(goal_perf),
                                                          first_arrive_steps=list(map(lambda x: x/10000., min_n(first_arrive_steps_list, 3)))))
            result2print[goal_perf].update({alg: [np.mean(min_n(first_arrive_steps_list, 3)), 2*np.std(min_n(first_arrive_steps_list, 3))]})
            df_list.append(df_for_this_alg_this_goal)
    total_dataframe = df_list[0].append(df_list[1:], ignore_index=True) if len(df_list) > 1 else df_list[0]
    figsize = (20, 8)
    axes_size = [0.06, 0.12, 0.94, 0.88]
    fontsize = 25
    f1 = plt.figure(1, figsize=figsize)
    ax1 = f1.add_axes(axes_size)
    sns.lineplot(x="goal_perf", y="first_arrive_steps", hue="algorithm", data=total_dataframe, linewidth=2,
                 palette=palette, legend=False)
    ax1.set_ylabel('Iterations required [x10000]', fontsize=fontsize)
    ax1.set_xlabel("Goal performance", fontsize=fontsize)
    handles, labels = ax1.get_legend_handles_labels()
    labels = lbs
    ax1.legend(handles=handles, labels=labels, loc='upper left', frameon=False, fontsize=11)
    ax1.set_xticklabels([str(goal) for goal in goal_perf_list])
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    print(result2print)
    plt.show()


def plot_opt_results_of_all_alg_n_runs(env, dirs_dict_for_plot=None):
    _, alg_list, lbs, palette, _, _ = help_func(env)
    dir_str = './results/{}/time'
    tag2plot = ['pg_time']  # 'update_time' 'pg_time']
    df_list = []
    for alg in alg_list:
        data2plot_dir = dir_str.format(alg)
        data2plot_dirs_list = dirs_dict_for_plot[alg] if dirs_dict_for_plot is not None else os.listdir(data2plot_dir)
        for num_run, dir in enumerate(data2plot_dirs_list):
            opt_dir = data2plot_dir + '/' + dir + '/logs/optimizer'
            opt_file = os.path.join(opt_dir,
                                     [file_name for file_name in os.listdir(opt_dir) if
                                      file_name.startswith('events')][0])
            opt_summarys = tf.data.TFRecordDataset([opt_file])
            data_in_one_run_of_one_alg = {key: [] for key in tag2plot}
            data_in_one_run_of_one_alg.update({'iteration': []})
            for opt_summary in opt_summarys:
                event = event_pb2.Event.FromString(opt_summary.numpy())
                for v in event.summary.value:
                    t = tf.make_ndarray(v.tensor)
                    for tag in tag2plot:
                        if tag in v.tag:
                            data_in_one_run_of_one_alg[tag].append(1000*float(t))# if float(t)<0.004 else 1.5)
                            data_in_one_run_of_one_alg['iteration'].append(int(event.step))
            len1, len2 = len(data_in_one_run_of_one_alg['iteration']), len(data_in_one_run_of_one_alg[tag2plot[0]])
            period = int(len1 / len2)
            data_in_one_run_of_one_alg['iteration'] = [data_in_one_run_of_one_alg['iteration'][i * period] / 10000. for
                                                       i in range(len2)]

            data_in_one_run_of_one_alg = {key: val[200:] for key, val in data_in_one_run_of_one_alg.items()}
            data_in_one_run_of_one_alg.update(dict(algorithm=alg, num_run=num_run))
            df_in_one_run_of_one_alg = pd.DataFrame(data_in_one_run_of_one_alg)
            df_list.append(df_in_one_run_of_one_alg)
    total_dataframe = df_list[0].append(df_list[1:], ignore_index=True) if len(df_list) > 1 else df_list[0]
    figsize = (20, 8)
    axes_size = [0.11, 0.12, 0.89, 0.88]
    fontsize = 25
    f1 = plt.figure(1, figsize=figsize)
    ax1 = f1.add_axes(axes_size)
    sns.boxplot(x="algorithm", y=tag2plot[0], data=total_dataframe, palette=palette)
    sns.despine(offset=10, trim=True)
    TAG2LBS = {'pg_time': 'Wall-clock Time per Gradient [ms]',
               'update_time': 'Wall-clock Time per Update [ms]'}
    ax1.set_ylabel(TAG2LBS[tag2plot[0]], fontsize=fontsize)
    labels = lbs
    ax1.set_xticklabels(labels, fontsize=fontsize)
    ax1.set_xlabel("", fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize, rotation=10)
    plt.show()


def calculate_fair_case_path_tracking():
    delta_u, delta_y, delta_phi, r, delta, acc = 2, 1, 10*np.pi/180, 0.2, 0.1, 0.5
    r = -0.01*delta_u**2-0.04*delta_y**2-0.1*delta_phi**2-0.02*r**2-5*delta**2-0.05*acc**2
    print(200*r)


def calculate_fair_case_inverted():
    x, theta, x_dot, theta_dot = 1., 0.1, 0.1, 0.05
    r = -0.01*x**2-theta**2-0.1*x_dot**2-0.1*theta_dot**2
    print(100*r)


if __name__ == "__main__":
    env = 'inverted_pendulum_env'  # inverted_pendulum_env path_tracking_env
    plot_eval_results_of_all_alg_n_runs(env)
    # plot_opt_results_of_all_alg_n_runs(env)
    # print(compute_convergence_speed(-100.))
    # plot_convergence_speed_for_different_goal_perf(env)
    # calculate_fair_case_path_tracking()
    # calculate_fair_case_inverted()