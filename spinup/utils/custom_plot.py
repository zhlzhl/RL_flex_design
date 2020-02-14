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


def plot_data(data, xaxis='Epoch', value="AverageEpRet", condition="Condition1", smooth=1, legend_name=None, **kwargs):
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

    sns.set(style="darkgrid", font_scale=1.5)
    # sns.tsplot(data=data, time=xaxis, value=value, unit="Unit", condition=condition, ci='sd', **kwargs)

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
    else:
        print("exp_name {} doesn't not contain 'T'".format(exp_name))
        return exp_name




def get_datasets(logdir, condition=None):
    """
    Recursively look through logdir for output files produced by
    spinup.logx.Logger. 

    Assumes that any file "progress.txt" is a valid hit. 
    """
    global exp_idx
    global units
    datasets = []
    for root, _, files in os.walk(logdir):
        if 'progress.txt' in files:
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
                exp_data = pd.read_table(os.path.join(root, 'progress.txt'))
            except:
                print('Could not read from %s' % os.path.join(root, 'progress.txt'))
                continue
            performance = 'AverageTestEpRet' if 'AverageTestEpRet' in exp_data else 'AverageEpRet'
            exp_data.insert(len(exp_data.columns), 'Unit', unit)
            exp_data.insert(len(exp_data.columns), 'Condition1', condition1)
            exp_data.insert(len(exp_data.columns), 'Condition2', condition2)
            exp_data.insert(len(exp_data.columns), 'Performance', exp_data[performance])

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


def get_datasets_by_identifier(logdir_identifiers):
    assert logdir_identifiers is not None, 'please specify logdir_identifier {}'.format(logdir_identifiers)

    target_logdirs = []

    # /home/user/git/spinningup/spinup/utils/custom_plot.py
    basedir = os.getcwd()
    path = Path(basedir)
    # get parent_path /home/user/git/spinningup/
    parent_path = path.parent.parent
    # get data dir /home/user/git/spinningup/data
    data_dir = osp.join(parent_path, 'data')
    # walk through data dir and look for log_dir that match with logdir_identifier

    for root, dir_list, file_list in os.walk(data_dir):
        if dir_list is not None:
            for dir in dir_list:
                if match_all_idenfiers(dir, logdir_identifiers):
                    for s_root, s_dir_list, s_file_list in os.walk(osp.join(root, dir)):
                        if s_dir_list is not None:
                            for s_dir in s_dir_list:
                                if match_all_idenfiers(s_dir, logdir_identifiers):
                                    target_logdirs.append(osp.join(root, s_root, s_dir))

    return target_logdirs


def get_all_datasets(all_logdirs, legend=None, select=None, exclude=None):
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
            data += get_datasets(log, leg)
    else:
        for log in logdirs:
            data += get_datasets(log)
    return data


def make_plots(all_logdirs, legend=None, xaxis=None, values=None, count=False,
               font_scale=1.5, smooth=1, select=None, exclude=None, estimator='mean', legend_name=None):
    data = get_all_datasets(all_logdirs, legend, select, exclude)
    values = values if isinstance(values, list) else [values]
    condition = 'Condition2' if count else 'Condition1'
    estimator = getattr(np, estimator)  # choose what to show on main curve: mean? max? min?
    for value in values:
        plt.figure()
        plot_data(data, xaxis=xaxis, value=value, condition=condition, smooth=smooth, estimator=estimator,
                  legend_name=legend_name)
    plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('logdir', nargs='*')
    parser.add_argument('logdir_identifiers', nargs='*')
    parser.add_argument('--legend', '-l', nargs='*')
    parser.add_argument('--legend_name', '-ln', type=str, default=None)
    parser.add_argument('--xaxis', '-x', default='TotalEnvInteracts')
    parser.add_argument('--value', '-y', default='Performance', nargs='*')
    parser.add_argument('--count', action='store_true')
    parser.add_argument('--smooth', '-s', type=int, default=1)
    parser.add_argument('--select', nargs='*')
    parser.add_argument('--exclude', nargs='*')
    parser.add_argument('--est', default='mean')
    args = parser.parse_args()
    """

    Args: 
        logdir_identifiers (list of strings): As many identifiers for experiments names to obtain the directories under 
        data directory you'd like to plot from.

        legend (strings): Optional way to specify legend for the plot. The 
            plotter legend will automatically use the ``exp_name`` from the
            config.json file, unless you tell it otherwise through this flag.
            This only works if you provide a name for each directory that
            will get plotted. (Note: this may not be the same as the number
            of logdir args you provide! Recall that the plotter looks for
            autocompletes of the logdir args: there may be more than one 
            match for a given logdir prefix, and you will need to provide a 
            legend string for each one of those matches---unless you have 
            removed some of them as candidates via selection or exclusion 
            rules (below).)

        xaxis (string): Pick what column from data is used for the x-axis.
             Defaults to ``TotalEnvInteracts``.

        value (strings): Pick what columns from data to graph on the y-axis. 
            Submitting multiple values will produce multiple graphs. Defaults
            to ``Performance``, which is not an actual output of any algorithm.
            Instead, ``Performance`` refers to either ``AverageEpRet``, the 
            correct performance measure for the on-policy algorithms, or
            ``AverageTestEpRet``, the correct performance measure for the 
            off-policy algorithms. The plotter will automatically figure out 
            which of ``AverageEpRet`` or ``AverageTestEpRet`` to report for 
            each separate logdir.

        count: Optional flag. By default, the plotter shows y-values which
            are averaged across all results that share an ``exp_name``, 
            which is typically a set of identical experiments that only vary
            in random seed. But if you'd like to see all of those curves 
            separately, use the ``--count`` flag.

        smooth (int): Smooth data by averaging it over a fixed window. This 
            parameter says how wide the averaging window will be.

        select (strings): Optional selection rule: the plotter will only show
            curves from logdirs that contain all of these substrings.

        exclude (strings): Optional exclusion rule: plotter will only show 
            curves from logdirs that do not contain these substrings.

    """

    logdirs = get_datasets_by_identifier(args.logdir_identifiers)

    # sort the dirs by 1) getting the last directory from the path using split('/')[-1], and 2) removing the dates in
    # the dir name only getting characters after indexed by 20 and above
    logdirs = sorted(logdirs, key= lambda x : x.split('/')[-1][20:])

    make_plots(logdirs, args.legend, args.xaxis, args.value, args.count,
               smooth=args.smooth, select=args.select, exclude=args.exclude,
               estimator=args.est, legend_name=args.legend_name)


if __name__ == "__main__":
    main()
