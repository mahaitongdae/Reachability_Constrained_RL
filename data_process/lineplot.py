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
txt_store_alg_list = ['CPO', 'PPO-Lagrangian', 'TRPO-Lagrangian']
ylim_dict = {'episode_return':{'CarGoal': [-5, 25],'PointButton': [-5, 33]},
             'episode_cost':{'CarGoal': [0, 28],'PointButton': [2, 16]}} # {'CarGoal': [-5, 25]}
fsac_bias = {'episode_return':{'PointButton':5,'CarGoal':0,},'episode_cost':{'PointButton':20,'CarGoal':0}}
fsac_init_bias = {'episode_return':{'PointButton':10,'CarGoal':0,},'episode_cost':{'PointButton':0,'CarGoal':0}}


def help_func():
    tag2plot = ['episode_cost']
    alg_list = ['CPO', 'PPO-Lagrangian', 'TRPO-Lagrangian', 'FSAC', ]  # 'FSAC', 'CPO', 'SAC','SAC-Lagrangian',
    lbs = ['CPO', 'PPO-L', 'TRPO-L', 'FAC', ]  # 'FAC', 'CPO', 'SAC','SAC-Lagrangian',
    task = ['PointButton']
    palette = "dark"
    goal_perf_list = [-200, -100, -50, -30, -20, -10, -5]
    dir_str = '../results/{}/{}/data2plot' # .format(algo name) #
    return tag2plot, alg_list, task, lbs, palette, goal_perf_list, dir_str

def plot_eval_results_of_all_alg_n_runs(dirs_dict_for_plot=None):
    tag2plot, alg_list, task_list, lbs, palette, _, dir_str = help_func()
    df_dict = {}
    df_in_one_run_of_one_alg = {}
    final_results = {}
    for task in task_list:
        df_list = []
        for alg in alg_list:
            final_results.update({alg:[]})
            data2plot_dir = dir_str.format(alg, task)
            data2plot_dirs_list = dirs_dict_for_plot[alg] if dirs_dict_for_plot is not None else os.listdir(data2plot_dir)
            for num_run, dir in enumerate(data2plot_dirs_list):
                if not dir.startswith('skip'):
                    if alg in txt_store_alg_list:
                        eval_dir = data2plot_dir + '/' + dir
                        print(eval_dir)
                        df_in_one_run_of_one_alg = get_datasets(eval_dir, tag2plot, alg=alg, num_run=num_run)
                    else:
                        eval_dir = data2plot_dir + '/' + dir + '/logs/evaluator'
                        print(eval_dir)
                        eval_file = os.path.join(eval_dir,
                                                 [file_name for file_name in os.listdir(eval_dir) if file_name.startswith('events')][0])
                        eval_summarys = tf.data.TFRecordDataset([eval_file])
                        data_in_one_run_of_one_alg = {key: [] for key in tag2plot}
                        data_in_one_run_of_one_alg.update({'iteration': []})
                        for eval_summary in eval_summarys:
                            event = event_pb2.Event.FromString(eval_summary.numpy())
                            step = event.step
                            if step % 10000 == 0:
                                for v in event.summary.value:
                                    t = tf.make_ndarray(v.tensor)
                                    for tag in tag2plot:
                                        if tag == v.tag[11:]:
                                            data_in_one_run_of_one_alg[tag].append((1-SMOOTHFACTOR)*data_in_one_run_of_one_alg[tag][-1] + SMOOTHFACTOR*float(t)
                                                                                   if data_in_one_run_of_one_alg[tag] else float(t))

                                            data_in_one_run_of_one_alg['iteration'].append(int(step))
                        add_data = []
                        add_step = []
                        k = 0
                        for i,d in enumerate(data_in_one_run_of_one_alg[tag]):
                            step = data_in_one_run_of_one_alg['iteration'][i]
                        if tag == 'episode_return':
                            print(add_step)
                            print(add_data)
                            data_in_one_run_of_one_alg['iteration'] += add_step
                            data_in_one_run_of_one_alg[tag] += add_data
                        len1, len2 = len(data_in_one_run_of_one_alg['iteration']), len(data_in_one_run_of_one_alg[tag2plot[0]])
                        period = int(len1/len2)
                        data_in_one_run_of_one_alg['iteration'] = [data_in_one_run_of_one_alg['iteration'][i*period]/1000000. for i in range(len2)]
                        if 'episode_cost' in data_in_one_run_of_one_alg.keys():
                            data_in_one_run_of_one_alg['episode_cost'] = np.array(data_in_one_run_of_one_alg[
                                                                             'episode_cost']) / 10

                        data_in_one_run_of_one_alg.update(dict(algorithm=alg, num_run=num_run))
                        df_in_one_run_of_one_alg = pd.DataFrame(data_in_one_run_of_one_alg)
                        y = np.ones(SMOOTHFACTOR2)
                        for tag in tag2plot:
                            x = np.asarray(df_in_one_run_of_one_alg[tag])
                            z = np.ones(len(x))
                            smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
                            df_in_one_run_of_one_alg[tag] = smoothed_x
                    df_list.append(df_in_one_run_of_one_alg)
                    lendf = len(df_in_one_run_of_one_alg[tag2plot[0]])
                    if not dir.startswith('init'):
                        final_results[alg] += list(df_in_one_run_of_one_alg[tag2plot[0]][lendf - 21: lendf - 1])
        compare_dict = dump_results(final_results)
        total_dataframe = df_list[0].append(df_list[1:], ignore_index=True) if len(df_list) > 1 else df_list[0]
        figsize = (6,6)
        axes_size = [0.11, 0.11, 0.89, 0.8] #if env == 'path_tracking_env' else [0.095, 0.11, 0.905, 0.89]
        fontsize = 16
        f1 = plt.figure(1, figsize=figsize)
        ax1 = f1.add_axes(axes_size)
        sns.set_palette([(0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
                         # (1.0, 0.4980392156862745, 0.054901960784313725),
                         (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
                         (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
                         (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
                         # (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
                         # (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
                         # (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
                         (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
                         (0.09019607843137255, 0.7450980392156863, 0.8117647058823529)]
                        )
        sns.lineplot(x="iteration", y=tag2plot[0], hue="algorithm", err_kws={'alpha': 0.1},
                     data=total_dataframe, linewidth=2, legend=False  # palette=palette,
                     )
        # legend = True if task == 'CarGoal' and tag == 'episode_cost' else False
        # if not legend:
        #     sns.lineplot(x="iteration", y=tag2plot[0], hue="algorithm", markers=True,
        #                  data=total_dataframe, linewidth=2, palette=palette, legend=False, err_kws={'alpha':0.1}
        #                  )
        # else:
        #     p = sns.lineplot(x="iteration", y=tag2plot[0], hue="algorithm", markers=True,
        #                  data=total_dataframe, linewidth=2, palette=palette, err_kws={'alpha':0.1}
        #                  )
        base = 4 if task == 'PointGoal' else 10
        handles, labels = ax1.get_legend_handles_labels()
        labels = lbs
        if tag == 'episode_cost':
            basescore = sns.lineplot(x=[0., 4.], y=[base, base], linewidth=2, color='black', linestyle='--')
        #     if legend:
        #         ax1.legend(handles=handles + [basescore.lines[-1]], labels=labels + ['Constraint'], loc='lower right',
        #                frameon=False, fontsize=fontsize)
        # else:
        #     if legend:
        #         ax1.legend(handles=handles , labels=labels , loc='lower right', frameon=False, fontsize=fontsize)
        # print(ax1.lines[0].get_data())
        ax1.set_ylabel('')
        ax1.set_xlabel("Million Iteration", fontsize=fontsize)
        print(compare_dict)
        # title = 'Episode Return {} \n {:+.0%} {:+.0%} {:+.0%}\n over TRPO-L, CPO, PPO-L'\
        #     .format(task, compare_dict['TRPO-Lagrangian'], compare_dict['CPO'], compare_dict['PPO-Lagrangian']) if tag == 'episode_return' else 'Episode Cost'
        title = 'Reward ({})'.format(task) if tag == 'episode_return' else 'Dangerous Action Rate (%) ({})'.format(task)
        if task in ylim_dict[tag]:
            ax1.set_ylim(*ylim_dict[tag][task])
        ax1.set_title(title, fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        # plt.show()
        fig_name = '../data_process/figure/' + task+'-'+tag + '.png'
        plt.savefig(fig_name)
        # allresults = {}
        # results2print = {}
        #
        # for alg, group in total_dataframe.groupby('algorithm'):
        #     allresults.update({alg: []})
        #     for ite, group1 in group.groupby('iteration'):
        #         mean = group1['episode_return'].mean()
        #         std = group1['episode_return'].std()
        #         allresults[alg].append((mean, std))
        #
        # for alg, result in allresults.items():
        #     mean, std = sorted(result, key=lambda x: x[0])[-1]
        #     results2print.update({alg: [mean, 2 * std]})
        #
        # print(results2print)


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
                exp_data = pd.read_table(os.path.join(root,'progress.txt'))
            except:
                print('Could not read from %s'%os.path.join(root,'progress.txt'))
                continue
            performance = 'AverageTestEpRet' if 'AverageTestEpRet' in exp_data else 'AverageEpRet'
            exp_data.insert(len(exp_data.columns),'Performance',exp_data[performance])
            exp_data.insert(len(exp_data.columns),'algorithm',alg)
            exp_data.insert(len(exp_data.columns), 'iteration', exp_data['TotalEnvInteracts']/1000000)
            exp_data.insert(len(exp_data.columns), 'episode_cost', exp_data['AverageEpCost'] / 10)
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
        compare_dict.update({alg:(np.mean(final_results_dict['FSAC'])-np.mean(final_results_dict[alg]))/np.mean(final_results_dict[alg])})
    return compare_dict

if __name__ == '__main__':
    # env = 'inverted_pendulum_env'  # inverted_pendulum_env path_tracking_env
    plot_eval_results_of_all_alg_n_runs()
    # load_from_tf1_event()
    # load_from_txt()