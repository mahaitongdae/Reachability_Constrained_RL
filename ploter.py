#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/9/25
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: ploter.py
# =====================================

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.core.util import event_pb2

sns.set(style="darkgrid")


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
                            data_in_one_run_of_one_alg[tag].append(float(t))
                            data_in_one_run_of_one_alg['iteration'].append(int(event.step))
            len1, len2 = len(data_in_one_run_of_one_alg['iteration']), len(data_in_one_run_of_one_alg[tag2plot[0]])
            period = int(len1/len2)
            data_in_one_run_of_one_alg['iteration'] = [data_in_one_run_of_one_alg['iteration'][i*period]/10000. for i in range(len2)]

            data_in_one_run_of_one_alg.update(dict(algorithm=alg, num_run=num_run))
            df_in_one_run_of_one_alg = pd.DataFrame(data_in_one_run_of_one_alg)
            df_list.append(df_in_one_run_of_one_alg)
    total_dataframe = df_list[0].append(df_list[1:], ignore_index=True) if len(df_list) > 1 else df_list[0]
    f1 = plt.figure(1)
    ax1 = f1.add_axes([0.20, 0.12, 0.78, 0.86])
    sns.lineplot(x="iteration", y="episode_return", hue="algorithm",
                 data=total_dataframe, linewidth=2, palette=palette,
                 )
    print(ax1.lines[0].get_data())
    ax1.set_ylabel('Episode Return', fontsize=15)
    ax1.set_xlabel("Iteration [x10000]", fontsize=15)
    handles, labels = ax1.get_legend_handles_labels()
    labels = lbs
    ax1.legend(handles=handles, labels=labels, loc='lower right', frameon=False, fontsize=11)
    lim = (-800, 50) if env == 'path_tracking_env' else (-100, 5)
    plt.ylim(*lim)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    if env == 'path_tracking_env':
        f2 = plt.figure(2)
        ax2 = f2.add_axes([0.15, 0.12, 0.83, 0.86])
        sns.lineplot(x="iteration", y="delta_y_mse", hue="algorithm",
                     data=total_dataframe, linewidth=2, palette=palette,
                     )
        ax2.set_ylabel('Position Error [m]', fontsize=15)
        ax2.set_xlabel("Iteration [x10000]", fontsize=15)
        handles, labels = ax2.get_legend_handles_labels()
        labels = lbs
        ax2.legend(handles=handles, labels=labels, loc='upper right', frameon=False, fontsize=11)
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)

        f3 = plt.figure(3)
        ax3 = f3.add_axes([0.15, 0.12, 0.83, 0.86])
        sns.lineplot(x="iteration", y="delta_phi_mse", hue="algorithm",
                     data=total_dataframe, linewidth=2, palette=palette,
                     legend=False)
        ax3.set_ylabel('Heading Angle Error [rad]', fontsize=15)
        ax3.set_xlabel("Iteration [x10000]", fontsize=15)
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)

        f4 = plt.figure(4)
        ax4 = f4.add_axes([0.15, 0.12, 0.83, 0.86])
        sns.lineplot(x="iteration", y="delta_v_mse", hue="algorithm",
                     data=total_dataframe, linewidth=2, palette=palette,
                     legend=False)
        ax4.set_ylabel('Velocity Error [m/s]', fontsize=15)
        ax4.set_xlabel("Iteration [x10000]", fontsize=15)
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)

        f5 = plt.figure(5)
        ax5 = f5.add_axes([0.15, 0.12, 0.83, 0.86])
        sns.lineplot(x="iteration", y="steer_mse", hue="algorithm",
                     data=total_dataframe, linewidth=2, palette=palette,
                     )
        ax5.set_ylabel('Front wheel angle [rad]', fontsize=15)
        ax5.set_xlabel("Iteration [x10000]", fontsize=15)
        handles, labels = ax5.get_legend_handles_labels()
        labels = lbs
        ax5.legend(handles=handles, labels=labels, loc='upper right', frameon=False, fontsize=11)
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)

        f6 = plt.figure(6)
        ax6 = f6.add_axes([0.15, 0.12, 0.83, 0.86])
        sns.lineplot(x="iteration", y="acc_mse", hue="algorithm",
                     data=total_dataframe, linewidth=2, palette=palette,
                     legend=False)
        ax6.set_ylabel('Acceleration [$m^2$/s]', fontsize=15)
        ax6.set_xlabel("Iteration [x10000]", fontsize=15)
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
    else:
        f2 = plt.figure(2)
        ax2 = f2.add_axes([0.15, 0.12, 0.83, 0.86])
        sns.lineplot(x="iteration", y="x_mse", hue="algorithm",
                     data=total_dataframe, linewidth=2, palette=palette,
                     )
        ax2.set_ylabel('Position Error [m]', fontsize=15)
        ax2.set_xlabel("Iteration [x10000]", fontsize=15)
        handles, labels = ax2.get_legend_handles_labels()
        labels = lbs
        ax2.legend(handles=handles, labels=labels, loc='upper right', frameon=False, fontsize=11)
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)

        f3 = plt.figure(3)
        ax3 = f3.add_axes([0.15, 0.12, 0.83, 0.86])
        sns.lineplot(x="iteration", y="theta_mse", hue="algorithm",
                     data=total_dataframe, linewidth=2, palette=palette,
                     legend=False)
        ax3.set_ylabel('Angle Error [rad]', fontsize=15)
        ax3.set_xlabel("Iteration [x10000]", fontsize=15)
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)

        f4 = plt.figure(4)
        ax4 = f4.add_axes([0.15, 0.12, 0.83, 0.86])
        sns.lineplot(x="iteration", y="xdot_mse", hue="algorithm",
                     data=total_dataframe, linewidth=2, palette=palette,
                     legend=False)
        ax4.set_ylabel('Velocity [m/s]', fontsize=15)
        ax4.set_xlabel("Iteration [x10000]", fontsize=15)
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)

        f5 = plt.figure(5)
        ax5 = f5.add_axes([0.15, 0.12, 0.83, 0.86])
        sns.lineplot(x="iteration", y="thetadot_mse", hue="algorithm",
                     data=total_dataframe, linewidth=2, palette=palette,
                     legend=False)
        ax5.set_ylabel('Angular velocity [rad/s]', fontsize=15)
        ax5.set_xlabel("Iteration [x10000]", fontsize=15)
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
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
                                                          first_arrive_steps=min_n(first_arrive_steps_list, 3)))
            result2print[goal_perf].update({alg: [np.mean(min_n(first_arrive_steps_list, 3)), 2*np.std(min_n(first_arrive_steps_list, 3))]})
            df_list.append(df_for_this_alg_this_goal)
    total_dataframe = df_list[0].append(df_list[1:], ignore_index=True) if len(df_list) > 1 else df_list[0]
    f1 = plt.figure(1)
    ax1 = f1.add_axes([0.20, 0.12, 0.78, 0.86])
    sns.lineplot(x="goal_perf", y="first_arrive_steps", hue="algorithm", data=total_dataframe, linewidth=2,
                 palette=palette, legend=False)
    ax1.set_ylabel('Convergence speed', fontsize=15)
    ax1.set_xlabel("Goal performance", fontsize=15)
    handles, labels = ax1.get_legend_handles_labels()
    labels = lbs
    ax1.legend(handles=handles, labels=labels, loc='upper left', frameon=False, fontsize=11)
    ax1.set_xticklabels([str(goal) for goal in goal_perf_list])
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    print(result2print)
    plt.show()


def plot_opt_results_of_all_alg_n_runs(env, dirs_dict_for_plot=None):
    _, alg_list, lbs, palette, _, dir_str = help_func(env)
    tag2plot = ['pg_time']  # 'update_time' , 'steer_mse', 'acc_mse']
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
                            data_in_one_run_of_one_alg[tag].append(float(t) if float(t)<0.3 else 0.2)
                            data_in_one_run_of_one_alg['iteration'].append(int(event.step))
            len1, len2 = len(data_in_one_run_of_one_alg['iteration']), len(data_in_one_run_of_one_alg[tag2plot[0]])
            period = int(len1 / len2)
            data_in_one_run_of_one_alg['iteration'] = [data_in_one_run_of_one_alg['iteration'][i * period] / 10000. for
                                                       i in range(len2)]

            data_in_one_run_of_one_alg = {key: val[1:] for key, val in data_in_one_run_of_one_alg.items()}
            data_in_one_run_of_one_alg.update(dict(algorithm=alg, num_run=num_run))
            df_in_one_run_of_one_alg = pd.DataFrame(data_in_one_run_of_one_alg)
            df_list.append(df_in_one_run_of_one_alg)
    total_dataframe = df_list[0].append(df_list[1:], ignore_index=True) if len(df_list) > 1 else df_list[0]
    f1 = plt.figure(1)
    ax1 = f1.add_axes([0.20, 0.15, 0.78, 0.84])
    sns.boxplot(x="algorithm", y=tag2plot[0], data=total_dataframe, palette=palette)
    sns.despine(offset=10, trim=True)
    ax1.set_ylabel('Wall-clock Time per Gradient [s]', fontsize=15)
    labels = lbs
    ax1.set_xticklabels(labels, fontsize=15)
    ax1.set_xlabel("", fontsize=15)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15, rotation=10)
    plt.show()


if __name__ == "__main__":
    env = 'inverted_pendulum_env'  # inverted_pendulum_env path_tracking_env
    # plot_eval_results_of_all_alg_n_runs(env)
    # plot_opt_results_of_all_alg_n_runs(env)
    # print(compute_convergence_speed(-100.))
    plot_convergence_speed_for_different_goal_perf(env)
