import numpy as np
import os
import pickle
from pathlib import Path
import shutil
import pickle
from spinup.FlexibilityEnv_input.inputload import load_FlexibilityEnv_input
from spinup.FlexibilityEnv.FlexibilityEnv import expected_sales_for_structure
from spinup.FlexibilityEnv_input.FlexibilityEnv_INPUTS import INPUTS




# looking for filenames starting with "best_eval_performance_n_structure", in sub-directory "simple_save999999" of
# sub-directory of specified directory
def collect_best_structures(experiment, env, exclude):
    dir = os.getcwd()
    output_dir = dir
    path = Path(dir)
    data_dir = os.path.join(path.parent, 'data')

    dst_files = []
    src_parent_dirs = []
    count = 0
    for root, _, files in os.walk(data_dir):
        if 'simple_save999999' in root:
            if all([identifier in root for identifier in [experiment, env]]):
                if all([ex not in root for ex in exclude]):
                    # find the right directory of simple_save999999. Now retrieve the right file.
                    for file in files:
                        if 'best_eval_performance_n_structure' in file:
                            src_file = os.path.join(root, file)
                            # append a count to the end of the file name to differentiate files with the same names
                            dst_file = append_suffix(file, count)
                            dst_file = os.path.join(output_dir, dst_file)

                            dst = shutil.copy2(src_file, dst_file)
                            count += 1

                            dst_files.append(dst)
                            parent_dir = root.split('data/')[1].split('/')[1].split('/')[0]
                            # store the name of the dir which is parent of the simple_save directoreis
                            src_parent_dirs.append(parent_dir)
                            print('copied {} from {} to {}'.format(file, parent_dir, dst))

    print('Copied {} files in total'.format(len(dst_files)))
    return dst_files, src_parent_dirs


def get_input_path(input):
    dir = os.getcwd()
    path = Path(dir)
    input_dir = os.path.join(path.parent, 'spinup/FlexibilityEnv_input', input)
    return input_dir


def get_input_key(experiment):
    if '-' in experiment:
        exp = experiment.split('-')[0]
    else:
        exp = experiment
    return exp


def append_suffix(dst_file, count):

    dst_file = "{}_SF{}.pkl".format(dst_file.split('.pk')[0], count)

    return dst_file


def remove_suffix_and_get_base_file(file):
    file = file.split('_SF')[0] + ".pkl"

    file = file.split('/')[-1]

    return file


def remove_previously_copied_files(path):
    count = 0
    for dir_or_file in os.listdir(path):
        if "best_eval_performance_n_structure_" in dir_or_file:
            os.remove(os.path.join(path, dir_or_file))
            count += 1
    print("removed {} previously copied best_eval_performance_n_structure files".format(count))


if __name__ == "__main__":

    experiment = '10x10a-lspe'
    envs = ['ENV5']
    print()
    input_path = get_input_path(INPUTS[get_input_key(experiment)])

    exclude = ['abcdef']

    m, n, mean_c, mean_d, sd_d, profit_mat, target_arcs, fixed_costs, flex_0 = load_FlexibilityEnv_input(input_path)

    # first remove all previously copied best_eval_performance_n_structure pickle files
    remove_previously_copied_files(os.getcwd())



    perf_dicts = []
    files_dicts = []
    parent_dir_dicts = []
    for env in envs:
        print("==== processing files for {}".format(env))
        files, parent_dirs = collect_best_structures(experiment, env, exclude)

        dict_tar_perf = {}  # this stores for a target arc the best performance value for target
        dict_tar_file = {}  # this stores for a target arc the structure pickle file which has the best performance
        dict_src_dir = {}  # this stores for a target arc the source dir (with name of experiment and seed) where the best strcture is copied from

        # files = sorted(files)

        for file, parent_dir in zip(files, parent_dirs):
            tar = file.split('T')[1].split('_SP')[0]
            if int(tar) not in target_arcs:
                continue

            # processing files with target_arcs that are specified in the input file of FlexibilityEnv
            with open(file, 'rb') as f:
                best_performance, best_structure, epoch = pickle.load(f)
                structure_performance, _ = expected_sales_for_structure(best_structure,
                                                                        10000,
                                                                        mean_c,
                                                                        demand_mean=mean_d,
                                                                        demand_std=sd_d,
                                                                        flow_profits=profit_mat,
                                                                        fixed_costs=fixed_costs,
                                                                        test=True,
                                                                        seed=7)

                ended_fixed_costs = np.sum(np.multiply(fixed_costs, best_structure))
                starting_fixed_costs = np.sum(np.multiply(fixed_costs, flex_0))
                induced_fixed_costs = - (ended_fixed_costs - starting_fixed_costs)
                test_best_perf = structure_performance + induced_fixed_costs

                print('tar {} | run {} | eval_best_perf {} | test_best_perf {}'.format(tar,
                                                                                       file[-16:].split('_s')[1].split(
                                                                                           '.pkl')[0],
                                                                                       best_performance,
                                                                                       test_best_perf))

                if tar not in dict_tar_perf:
                    dict_tar_perf[tar] = test_best_perf
                    dict_tar_file[tar] = file
                    dict_src_dir[tar] = parent_dir

                else:
                    if test_best_perf > dict_tar_perf[tar]:
                        print('updated dict_tar_file[{}] with {} from {}, '
                              'replacing {} from {}'.format(tar,
                                                            test_best_perf,
                                                            parent_dir,
                                                            dict_tar_perf[tar],
                                                            dict_src_dir[tar]))
                        dict_tar_perf[tar] = test_best_perf
                        dict_tar_file[tar] = file
                        dict_src_dir[tar] = parent_dir

        # store dict_tar_perf to perf_dicts
        perf_dicts.append(dict_tar_perf.copy())
        files_dicts.append(dict_tar_file.copy())
        parent_dir_dicts.append(dict_src_dir)

        # copy files in dict_tar_file to an env folder
        output_dir = os.path.join(os.getcwd(), experiment)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_dir = os.path.join(output_dir, env)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # remove previously copied files if existing
        remove_previously_copied_files(output_dir)

        for file in dict_tar_file.values():
            output_file = os.path.join(output_dir, remove_suffix_and_get_base_file(file))
            shutil.copy2(file, output_file)

        print('selected best structures are from experiments below: ')

        [print(key, " :: ", value) for (key, value) in sorted(dict_src_dir.items())]

        [print(key, " :: ", value) for (key, value) in sorted(dict_tar_perf.items())]


        # todo add the print of the original files of best structure selected

        print("---- done for {}\n\n".format(env))
