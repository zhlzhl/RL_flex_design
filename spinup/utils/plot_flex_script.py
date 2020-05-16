import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import os.path as osp
import numpy as np
from pathlib import Path

DIV_LINE_WIDTH = 50

# Global vars for tracking and labeling data at load time.
exp_idx = 0
units = dict()


def plot_data(data, xaxis=None, value=None, condition="Condition1", smooth=1, legend_name=None, eval=False, **kwargs):
    if smooth > 1:
        """
        smooth data with moving window average.
        that is,
            smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
        where the "smooth" param is width of that window (2k+1)
        """
        y = np.ones(smooth)
        for datum in data:
            x = np.asarray(datum[value])
            z = np.ones(len(x))
            smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
            datum[value] = smoothed_x

    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True, sort=True)

    # change 'Condition1' to legend_name
    if legend_name is not None:
        data.rename(columns={'Condition1': legend_name}, inplace=True)
        condition = legend_name

    if not eval:
        # change TotalEnvInteracts to training steps
        if xaxis is not None:
            if xaxis == 'training steps':
                data.rename(columns={'TotalEnvInteracts': xaxis}, inplace=True)
        # change AverageEpRet to average episode return
        if value is not None:
            if value == 'average return':
                data.rename(columns={'AverageEpRet': value}, inplace=True)

        sns.set(style="darkgrid", font_scale=1.)
        # sns.tsplot(data=data, time=xaxis, value=value, unit="Unit", condition=condition, ci='sd', **kwargs)

        sns.lineplot(data=data, x=xaxis, y=value, hue=condition, ci='sd', **kwargs)
    else:
        xaxis = 'Epoch'
        value = 'EpRet'
        sns.lineplot(data=data, x=xaxis, y=value, hue=condition, ci='sd', **kwargs)

    """
    If you upgrade to any version of Seaborn greater than 0.8.1, switch from 
    tsplot to lineplot replacing L29 with:


    Changes the colorscheme and the default legend style, though.
    """
    plt.legend(loc='best').set_draggable(True)

    """
    For the version of the legend used in the Spinning Up benchmarking page, 
    swap L38 with:

    plt.legend(loc='upper center', ncol=6, handlelength=1,
               mode="expand", borderaxespad=0., prop={'size': 13})
    """

    xscale = np.max(np.asarray(data[xaxis])) > 5e3
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    plt.tight_layout(pad=0.5)


def get_Tarc(exp_name):
    # exaple exp_name 2020-02-04_F20x20T40_SP50_PPO_CH2048-512_SPE24000_ITR80_EP700, the target arc # is after the first "T"

    if 'T' in exp_name:
        splits = exp_name.split('T')
        # use the second split to split again
        target_arc = splits[1].split('_')[0]
        return "T{}".format(target_arc)
    elif 'tar' in exp_name:
        splits = exp_name.split('tar')
        target_arc = splits[-1]
        return "K = {}".format(target_arc)
    else:
        print("exp_name {} doesn't not contain 'T'".format(exp_name))
        return exp_name


def extract_seed(root):
    return int(root.split('_s')[-1])



def get_datasets(logdir, eval=False, condition=None):
    """
    Recursively look through logdir for output files produced by
    spinup.logx.Logger. 

    Assumes that any file "progress.txt" is a valid hit. 
    """
    global exp_idx
    global units
    datasets = []

    progress_file = 'progress_eval.txt' if eval else 'progress.txt'

    for root, _, files in os.walk(logdir):
        if progress_file in files:
            exp_name = None
            try:
                config_path = open(os.path.join(root, 'config.json'))
                config = json.load(config_path)
                if 'exp_name' in config:
                    exp_name = config['exp_name']
            except:
                print('No file named config.json')
            condition1 = condition or get_Tarc(exp_name) or 'exp'
            condition2 = condition1 + '-' + str(exp_idx)
            exp_idx += 1
            if condition1 not in units:
                units[condition1] = 0
            unit = units[condition1]
            units[condition1] += 1

            try:
                exp_data = pd.read_table(os.path.join(root, progress_file))
            except:
                print('Could not read from %s' % os.path.join(root, progress_file))
                continue
            if not eval:
                performance = 'AverageTestEpRet' if 'AverageTestEpRet' in exp_data else 'AverageEpRet'
            else:
                performance = 'EpRet'
            exp_data.insert(len(exp_data.columns), 'Unit', unit)
            exp_data.insert(len(exp_data.columns), 'Condition1', condition1)
            exp_data.insert(len(exp_data.columns), 'Condition2', condition2)
            exp_data.insert(len(exp_data.columns), 'Performance', exp_data[performance])

            if not eval:
            # extract seed from root and add it to data frame of progress.txt
                seed = extract_seed(root)
                exp_data.insert(len(exp_data.columns), 'Seed', seed)

            # convert Time from seconds to hours
            if 'Time' in exp_data.columns:
                exp_data['Time'] = exp_data['Time'] / 3600
                exp_data.rename(columns={'Time': 'Time (hours)'}, inplace=True)

            datasets.append(exp_data)
            break
    return datasets


def match_all_idenfiers(dir, identifiers):
    assert identifiers is not None, "identifiers: {}".format(identifiers)

    for identifier in identifiers:
        if identifier not in dir:
            return False

    return True


def get_datasets_by_identifier(logdir_identifiers, data_dir=None):
    assert logdir_identifiers is not None, 'please specify logdir_identifier {}'.format(logdir_identifiers)

    target_logdirs = []

    if data_dir is None:  # use default data dir
        # /home/user/git/spinningup/spinup/utils/plot_flex.py
        basedir = os.getcwd()
        path = Path(basedir)
        # get parent_path /home/user/git/spinningup/
        parent_path = path.parent.parent
        # get data dir /home/user/git/spinningup/data
        data_dir = osp.join(parent_path, 'data')

    for root, dirs, files in os.walk(data_dir):
        if match_all_idenfiers(root, logdir_identifiers):
            for dir in dirs:
                if match_all_idenfiers(dir, logdir_identifiers):
                    target_logdirs.append(osp.join(root, dir))

    return target_logdirs


def get_all_datasets(all_logdirs, legend=None, select=None, exclude=None, eval=False):
    """
    For every entry in all_logdirs,
        1) check if the entry is a real directory and if it is, 
           pull data from it; 

        2) if not, check to see if the entry is a prefix for a 
           real directory, and pull data from that.
    """
    logdirs = []
    for logdir in all_logdirs:
        if osp.isdir(logdir) and logdir[-1] == os.sep:
            logdirs += [logdir]
        else:
            basedir = osp.dirname(logdir)
            fulldir = lambda x: osp.join(basedir, x)
            prefix = logdir.split(os.sep)[-1]
            listdir = os.listdir(basedir)
            logdirs += sorted([fulldir(x) for x in listdir if prefix in x])

    """
    Enforce selection rules, which check logdirs for certain substrings.
    Makes it easier to look at graphs from particular ablations, if you
    launch many jobs at once with similar names.
    """
    if select is not None:
        logdirs = [log for log in logdirs if any(x in log for x in select)]
    if exclude is not None:
        logdirs = [log for log in logdirs if all(not (x in log) for x in exclude)]

    # Verify logdirs
    print('Plotting from...\n' + '=' * DIV_LINE_WIDTH + '\n')
    for logdir in logdirs:
        print(logdir)
    print('\n' + '=' * DIV_LINE_WIDTH)

    # Make sure the legend is compatible with the logdirs
    assert not (legend) or (len(legend) == len(logdirs)), \
        "Must give a legend title for each set of experiments."

    # Load data from logdirs
    data = []
    if legend:
        for log, leg in zip(logdirs, legend):
            data += get_datasets(log, eval, leg)
    else:
        for log in logdirs:
            data += get_datasets(log, eval)
    return data


# def make_plots(all_logdirs, legend=None, xaxis=None, values=None, count=False,
#                font_scale=1.5, smooth=1, select=None, exclude=None, estimator='mean', legend_name=None, eval=False):
#     data = get_all_datasets(all_logdirs, legend, select, exclude, eval)
#     values = values if isinstance(values, list) else [values]
#     condition = 'Condition2' if count else 'Condition1'
#     estimator = getattr(np, estimator)  # choose what to show on main curve: mean? max? min?
#     for value in values:
#         plt.figure()
#         plot_data(data, xaxis=xaxis, value=value, condition=condition, smooth=smooth, estimator=estimator,
#                   legend_name=legend_name, eval=eval)
#     plt.show()


if __name__ == "__main__":
    logdir_identifiers = ['10x10a']
    xaxis = 'TotalEnvInteracts'
    values = ['Performance']
    count = False  # if True, then do not aggregate seeds
    smooth = 1  # window size for running average
    select = None
    exclude = None
    estimator = 'mean'  # estimator for sns.lineplot
    data_dir = "/home/user/git/RL_flex_design/data_backup/plotting_training_curves"  # explicite specify data directory
    legend = None  # a list of legends with size equal to number of experiments
    legend_name = 'K'

    logdirs = get_datasets_by_identifier(logdir_identifiers, data_dir)

    # sort the dirs by 1) getting the last directory from the path using split('/')[-1], and 2) removing the dates in
    # the dir name only getting characters after indexed by 20 and above
    all_logdirs = sorted(logdirs, key=lambda x: x.split('/')[-1][20:])
    print(logdirs)

    data = get_all_datasets(all_logdirs, legend, select, exclude, eval=False)
    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True, sort=True)
    data_eval = get_all_datasets(all_logdirs, legend, select, exclude, eval=True)
    if isinstance(data_eval, list):
        data_eval = pd.concat(data_eval, ignore_index=True, sort=True)

    condition = 'Condition2' if count else 'Condition1'
    estimator = getattr(np, estimator)  # choose what to show on main curve: mean? max? min?
    # change 'Condition1' to legend_name
    if legend_name is not None:
        data.rename(columns={'Condition1': legend_name}, inplace=True)
        data_eval.rename(columns={'Condition1': legend_name}, inplace=True)
        condition = legend_name
    values = values if isinstance(values, list) else [values]


    for value in values:
        ##### plot training curve
        plt.figure()

        sns.set(style="darkgrid", font_scale=1.)
        sns.lineplot(data=data, x=xaxis, y=value, hue=condition, ci='sd', estimator=estimator)
        plt.xlabel('training steps')
        plt.ylabel('average return')
        plt.legend(loc='best').set_draggable(True)
        # set x-axis to scale in scientific notation
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.tight_layout(pad=0.5)

        ##### plot eval curve
        time = data[['Time (hours)', 'K', 'Seed', 'Epoch', 'TotalEnvInteracts']]
        data_eval = pd.merge(data_eval, time, how='left', on=['K', 'Seed', 'Epoch'])
        # drop un-used columns
        data_eval.drop(columns=['Condition2', 'EpLen', 'Performance', 'Unit', 'Episode'], inplace=True)
        plt.figure()
        sns.lineplot(data=data_eval, x=xaxis, y='EpRet', hue=condition, ci='sd', estimator=estimator)
        plt.xlabel('training steps')
        plt.ylabel('average return')
        plt.legend(loc='best').set_draggable(True)
        # set x-axis to scale in scientific notation
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.tight_layout(pad=0.5)


        ##### plot eval with AverageEpRet curve
        data_eval_AverageEpRet = data_eval.groupby(['K', 'Seed','Epoch'], as_index=False).mean()
        plt.figure()
        sns.lineplot(data=data_eval_AverageEpRet, x=xaxis, y='EpRet', hue=condition, ci='sd', estimator=estimator)
        plt.xlabel('training steps')
        plt.ylabel('average return')
        plt.legend(loc='best').set_draggable(True)
        # set x-axis to scale in scientific notation
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.tight_layout(pad=0.5)

        # ##### plot eval with MaxEpRet curve
        # # data_eval_MaxEpRet = data_eval.groupby(['K', 'Seed','Epoch'], as_index=False).max()
        # data_eval_MaxEpRet = data_eval.groupby(['K', 'Seed', 'Epoch'],
        #                                        as_index=False).agg({'EpRet': lambda grp: grp.nlargest(3).mean(),
        #                                                             'Time (hours)': np.mean,
        #                                                             'TotalEnvInteracts': np.mean})
        # plt.figure()
        # sns.lineplot(data=data_eval_MaxEpRet, x=xaxis, y='EpRet', hue=condition, ci='sd', estimator=estimator)
        # plt.xlabel('training steps')
        # plt.ylabel('average return')
        # plt.legend(loc='best').set_draggable(True)
        # # set x-axis to scale in scientific notation
        # plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        # plt.tight_layout(pad=0.5)

    plt.show()
