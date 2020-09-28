#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/9/25
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: ploter.py
# =====================================

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.core.util import event_pb2

sns.set(style="darkgrid")


def plot_eval_results_of_all_alg_n_runs(dirs_dict_for_plot=None):
    tag2plot = ['episode_return', 'episode_len', 'delta_y_mse', 'delta_phi_mse', 'delta_v_mse',
                'stationary_rew_mean']#, 'steer_mse', 'acc_mse']
    df_list = []
    for alg in ['MPG-v3', 'NDPG', 'NADP']:
        data2plot_dir = './results/{}/data2plot'.format(alg)
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
                        if tag in v.tag:
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
    sns.lineplot(x="iteration", y="episode_return", hue="algorithm", data=total_dataframe, linewidth=2, palette="bright")

    ax1.set_ylabel('Episode Return', fontsize=15)
    ax1.set_xlabel("Ten Thousand Iteration", fontsize=15)
    handles, labels = ax1.get_legend_handles_labels()
    labels = ['Mixed PG', 'DPG', 'ADP']
    ax1.legend(handles=handles, labels=labels, loc='lower right', frameon=False, fontsize=15)
    plt.ylim(-800, 0)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.show()


def plot_opt_results_of_all_alg_n_runs(dirs_dict_for_plot=None):
    tag2plot = ['update_time']  # , 'steer_mse', 'acc_mse']
    df_list = []
    for alg in ['MPG-v3', 'NDPG', 'NADP']:
        data2plot_dir = './results/{}/data2plot'.format(alg)
        data2plot_dirs_list = dirs_dict_for_plot[alg] if dirs_dict_for_plot is not None else os.listdir(data2plot_dir)
        for num_run, dir in enumerate(data2plot_dirs_list):
            opt_dir = dir + '/logs/optimizer'
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
                            data_in_one_run_of_one_alg[tag].append(float(t))
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
    ax1 = f1.add_axes([0.20, 0.12, 0.78, 0.86])
    sns.boxplot(x="algorithm", y="update_time", data=total_dataframe)
    sns.despine(offset=10, trim=True)
    ax1.set_ylabel('Wall-clock Time per Iterations [s]', fontsize=15)
    ax1.set_xlabel("", fontsize=15)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.show()


if __name__ == "__main__":
    plot_eval_results_of_all_alg_n_runs(dict(SAC=['./results/SAC/good'],
                                             ))
    # plot_opt_results_of_all_alg_n_runs(dict(SAC=['./results/SAC/good'],
    #                                         ))

