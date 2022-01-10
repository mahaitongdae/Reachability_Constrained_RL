import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.core.util import event_pb2
import os.path as osp
import tensorboard as tb
from tensorboard.backend.event_processing import event_accumulator
import json

sns.set(style="darkgrid")
SMOOTHFACTOR = 0.9
SMOOTHFACTOR2 = 20
SMOOTHFACTOR3 = 20
DIV_LINE_WIDTH = 50
fontsize = 12
paper = True
env_name_dict = {'quadrotor': 'Quadrotor Circle Tracking'}
tag_name_dict = {'episode_return': 'Total average return',
                 'episode_constraint_violation': 'Total average constraint violation rate (%)'}

txt_store_alg_list = ['CPO', 'PPO-Lagrangian', 'TRPO-Lagrangian']
ylim_dict = {'episode_return': {'quadrotor': [-500, 0],
                                'PointButton': [-5, 33]},
             'episode_constraint_violation': {'CarGoal': [0, 28],
                                              'PointButton': [2, 16],
                                              'quadrotor': [-5, 100]}
             }  # {'CarGoal': [-5, 25]}


def help_func():
    tag2plot = ['episode_return', 'episode_constraint_violation']
    alg_list = ['RAC-feasibility', 'SAC-Lagrangian-Qc', 'SAC-RewardShaping-Qc', 'FSAC-A-si', 'SAC-CBF-CBF']
    lbs = ['RAC (ours)', 'SAC-Lagrangian', 'SAC-Reward Shaping', 'Energy-based SAC', 'CBF-based SAC']  # 'FAC', 'CPO', 'SAC','SAC-Lagrangian',
    task = ['quadrotor']
    palette = "bright"
    goal_perf_list = [-200, -100, -50, -30, -20, -10, -5]
    dir_str = '../results/{}/{}'  # .format(env_id, alg_name) #
    return tag2plot, alg_list, task, lbs, palette, goal_perf_list, dir_str


def plot_eval_results_of_all_alg_n_runs(dirs_dict_for_plot=None):
    tag2plot, alg_list, task_list, lbs, palette, _, dir_str = help_func()
    df_dict = {}
    df_in_one_run_of_one_alg = {}
    for task in task_list:
        df_list = []
        for alg in alg_list:
            dir_str_alg = dir_str + '/data2plot' if alg not in txt_store_alg_list else dir_str
            data2plot_dir = dir_str_alg.format(task, alg)
            data2plot_dirs_list = dirs_dict_for_plot[alg] if dirs_dict_for_plot is not None else os.listdir(
                data2plot_dir)
            for num_run, dir in enumerate(data2plot_dirs_list):
                # if num_run == 1:
                #     break
                if alg in txt_store_alg_list:  # result run by safety-starter-agents
                    eval_dir = data2plot_dir + '/' + dir
                    print(eval_dir)
                    df_in_one_run_of_one_alg = get_datasets(eval_dir, tag2plot, alg=alg, num_run=num_run)
                else:  # result run by PABAL
                    data_in_one_run_of_one_alg = dict()
                    for tag in tag2plot:
                        eval_dir = data2plot_dir + '/' + dir + '/logs/evaluator' if tag.startswith(
                            'ep') else data2plot_dir + '/' + dir + '/logs/optimizer'
                        print(eval_dir)
                        eval_file = os.path.join(eval_dir,
                                                 [file_name for file_name in os.listdir(eval_dir) if
                                                  file_name.startswith('events')][0])
                        eval_summarys = tf.data.TFRecordDataset([eval_file])

                        data_in_one_run_of_one_alg.update({tag: []})
                        data_in_one_run_of_one_alg.update({'iteration': []})
                        for eval_summary in eval_summarys:
                            event = event_pb2.Event.FromString(eval_summary.numpy())
                            if event.step % 10000 != 0: continue  # TODO: step/iteration
                            for v in event.summary.value:
                                t = tf.make_ndarray(v.tensor)
                                tag_in_events = 'evaluation/' + tag if tag.startswith('ep') else 'optimizer/' + tag
                                if tag_in_events == v.tag:
                                    # if tag == 'episode_return':
                                    #     t = np.clip(t, -2.0, 100.0)
                                    if tag == 'episode_constraint_violation':
                                        t *= 100.0
                                    data_in_one_run_of_one_alg[tag].append(
                                        (1 - SMOOTHFACTOR) * data_in_one_run_of_one_alg[tag][-1] + SMOOTHFACTOR * float(
                                            t)
                                        if data_in_one_run_of_one_alg[tag] else float(t))  # TODO: why smooth?
                                    data_in_one_run_of_one_alg['iteration'].append(int(event.step / 10000.))
                    # len1, len2 = len(data_in_one_run_of_one_alg['iteration']), len(data_in_one_run_of_one_alg[tag2plot[0]])
                    # period = int(len1/len2)
                    # data_in_one_run_of_one_alg['iteration'] = [data_in_one_run_of_one_alg['iteration'][i*period] for i in range(len2)]

                    data_in_one_run_of_one_alg.update(dict(algorithm=alg, num_run=num_run))
                    df_in_one_run_of_one_alg = pd.DataFrame(data_in_one_run_of_one_alg)
                df_list.append(df_in_one_run_of_one_alg)
        total_dataframe = df_list[0].append(df_list[1:], ignore_index=True) if len(df_list) > 1 else df_list[0]

        for i, tag in enumerate(tag2plot):
            figsize = (5, 4)
            axes_size = [0.13, 0.14, 0.85, 0.80] if paper else [0.13, 0.11, 0.86, 0.84]
            plt.figure(figsize=figsize)  # figsize=figsize
            ax1 = plt.axes()  # f1.add_axes(axes_size)
            sns.lineplot(x="iteration", y=tag, hue="algorithm",
                         data=total_dataframe, linewidth=2, palette=palette
                         )
            title = env_name_dict[task]
            ax1.set_ylabel(tag_name_dict[tag], fontsize=fontsize)
            ax1.set_xlabel("Iteration [x10000]", fontsize=fontsize)
            handles, labels = ax1.get_legend_handles_labels()
            labels = lbs
            ax1.legend(handles=handles, labels=labels,
                       # bbox_to_anchor=(0.5, -0.1), loc='lower center', ncol=3,
                       loc='best',
                       frameon=False, fontsize=fontsize)
            if i != 0:
                ax1.get_legend().remove()
            plt.yticks(fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.xlim([0, 200])
            plt.ylim(ylim_dict[tag][task])
            plt.title(title, fontsize=fontsize)
            # plt.gcf().set_size_inches(3.85, 2.75)
            plt.tight_layout(pad=0.5)

            save_dir = '../data_process/figure/'
            os.makedirs(save_dir, exist_ok=True)
            fig_name = save_dir + task + '-' + tag + '.pdf'
            plt.savefig(fig_name)

        # plt.show()


def get_datasets(logdir, tag2plot, alg, condition=None, smooth=SMOOTHFACTOR3, num_run=0):
    """
    Recursively look through logdir for output files produced by
    spinup.logx.Logger.

    Assumes that any file "progress.txt" is a valid hit.
    """
    # global exp_idx
    # global units
    datasets = []

    for root, _, files in os.walk(logdir):

        if 'progress.txt' in files:
            try:
                exp_data = pd.read_table(os.path.join(root, 'progress.txt'))
            except:
                print('Could not read from %s' % os.path.join(root, 'progress.txt'))
                continue
            performance = 'AverageTestEpRet' if 'AverageTestEpRet' in exp_data else 'AverageEpRet'
            exp_data.insert(len(exp_data.columns), 'Performance', exp_data[performance])
            exp_data.insert(len(exp_data.columns), 'algorithm', alg)
            exp_data.insert(len(exp_data.columns), 'iteration', exp_data['TotalEnvInteracts'] / 1000000)
            exp_data.insert(len(exp_data.columns), 'episode_constraint_violation', exp_data['AverageEpCost'] / 10)
            exp_data.insert(len(exp_data.columns), 'episode_return', exp_data['AverageEpRet'])
            exp_data.insert(len(exp_data.columns), 'num_run', num_run)
            datasets.append(exp_data)
            data = datasets

            for tag in tag2plot:
                if smooth > 1:
                    """
                    smooth data with moving window average.
                    that is,
                        smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
                    where the "smooth" param is width of that window (2k+1)
                    """
                    y = np.ones(smooth)
                    for datum in data:
                        x = np.asarray(datum[tag])
                        z = np.ones(len(x))
                        smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
                        datum[tag] = smoothed_x

            if isinstance(data, list):
                data = pd.concat(data, ignore_index=True)

            slice_list = tag2plot + ['algorithm', 'iteration', 'num_run']

    return data.loc[:, slice_list]


def dump_results(final_results_dict):
    compare_dict = {}
    for alg in final_results_dict.keys():
        print('alg: {}, mean {}, std {}'.format(alg, np.mean(final_results_dict[alg]), np.std(final_results_dict[alg])))
        compare_dict.update({alg: (np.mean(final_results_dict['FSAC']) - np.mean(final_results_dict[alg])) / np.mean(
            final_results_dict[alg])})
    return compare_dict


if __name__ == '__main__':
    plot_eval_results_of_all_alg_n_runs()
    # load_from_tf1_event()
    # load_from_txt()