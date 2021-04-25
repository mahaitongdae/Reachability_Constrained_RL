import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.core.util import event_pb2
import tensorboard as tb

sns.set(style="darkgrid")
SMOOTHFACTOR = 0.8

def load_from_event():
    tag2plot = ['episode_return']
    eval_summarys = tf.data.TFRecordDataset(['/home/mahaitong/PycharmProjects/mpg/results/FSAC/CarButton1-2021-04-20-14-40-50/logs/evaluator/events.out.tfevents.1618900860.mahaitong-virtual-machine.33389.1126.v2'])
    data_in_one_run_of_one_alg = {key: [] for key in tag2plot}
    data_in_one_run_of_one_alg.update({'iteration': []})
    for eval_summary in eval_summarys:
        event = event_pb2.Event.FromString(eval_summary.numpy())
        for v in event.summary.value:
            t = tf.make_ndarray(v.tensor)
            for tag in tag2plot:
                if tag == v.tag[11:]:
                    data_in_one_run_of_one_alg[tag].append(
                        (1 - SMOOTHFACTOR) * data_in_one_run_of_one_alg[tag][-1] + SMOOTHFACTOR * float(t)
                        if data_in_one_run_of_one_alg[tag] else float(t))
                    data_in_one_run_of_one_alg['iteration'].append(int(event.step))
    a = 1
    

def load_from_tf1_event():
    from tensorboard.backend.event_processing import event_accumulator

    tag2plot = []
    ea = event_accumulator.EventAccumulator('/home/mahaitong/PycharmProjects/mpg/results/FSAC/tf1_test/fsac')
    ea.Reload()
    tag2plot += ea.scalars.Keys()
    data_in_one_run_of_one_alg = {key: [] for key in tag2plot}
    data_in_one_run_of_one_alg.update({'iteration': []})
    valid_tag_list = [i for i in tag2plot if i in ea.scalars.Keys()]
    for tag in valid_tag_list:
        events = ea.scalars.Items(tag)
        for idx, event in enumerate(events):
            t = event.value
            data_in_one_run_of_one_alg[tag].append(
                (1 - SMOOTHFACTOR) * data_in_one_run_of_one_alg[tag][-1] + SMOOTHFACTOR * float(t)
                if data_in_one_run_of_one_alg[tag] else float(t))
            if tag == valid_tag_list[0]:
                data_in_one_run_of_one_alg['iteration'].append(int(event.step))


    a = 1

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

if __name__ == '__main__':
    # env = 'inverted_pendulum_env'  # inverted_pendulum_env path_tracking_env
    # plot_eval_results_of_all_alg_n_runs(env)
    load_from_tf1_event()