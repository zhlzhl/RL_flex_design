import numpy as np
import os
import pickle
from pathlib import Path
import shutil
import pickle
from spinup.FlexibilityEnv_input.inputload import load_FlexibilityEnv_input
from spinup.FlexibilityEnv.FlexibilityEnv import expected_sales_for_structure


# looking for filenames starting with "best_eval_performance_n_structure", in sub-directory "simple_save999999" of
# sub-directory of specified directory
def collect_best_structures(experiment, env, exclude):
    dir = os.getcwd()
    output_dir = dir
    path = Path(dir)
    data_dir = os.path.join(path.parent, 'data')

    dst_files = []
    for root, _, files in os.walk(data_dir):
        if 'simple_save999999' in root:
            if all([identifier in root for identifier in [experiment, env]]):
                if all([ex not in root for ex in exclude]):
                    # find the right directory of simple_save999999. Now retrieve the right file.
                    for file in files:
                        if 'best_eval_performance_n_structure' in file:
                            src_file = os.path.join(root, file)
                            dst_file = os.path.join(output_dir, file)

                            if os.path.exists(dst_file):
                                # file already exists, no need to copy
                                dst_files.append(dst_file)
                            else:
                                # copy file
                                dst = shutil.copy2(src_file, dst_file)
                                dst_files.append(dst)
                                print('copied {} to {}'.format(src_file, dst_file))

    print('Located {} files in total'.format(len(dst_files)))
    return dst_files


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


from spinup.FlexibilityEnv_input.FlexibilityEnv_input_files import INPUTS

if __name__ == "__main__":
    experiment = '10x10a-lspe'
    envs = ['ENV5']
    print()
    input_path = get_input_path(INPUTS[get_input_key(experiment)])

    exclude = ['abcdef']

    m, n, mean_c, mean_d, sd_d, profit_mat, target_arcs, fixed_costs, flex_0 = load_FlexibilityEnv_input(input_path)

    perf_dicts = []
    files_dicts = []
    for env in envs:
        print("==== processing files for {}".format(env))
        files = collect_best_structures(experiment, env, exclude)

        dict_tar_perf = {}
        dict_tar_file = {}

        files = sorted(files)

        for file in files:
            tar = file.split('T')[1].split('_SP')[0]
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
                else:
                    if test_best_perf > dict_tar_perf[tar]:
                        print('updated dict_tar_file[{}] with {}, replacing {}'.format(tar, test_best_perf,
                                                                                       dict_tar_perf[tar]))
                        dict_tar_perf[tar] = test_best_perf
                        dict_tar_file[tar] = file

        # store dict_tar_perf to perf_dicts
        perf_dicts.append(dict_tar_perf.copy())
        files_dicts.append(dict_tar_file.copy())

        # copy files in dict_tar_file to an env folder
        output_dir = os.path.join(os.getcwd(), experiment)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_dir = os.path.join(output_dir, env)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # output_dir = os.path.join(output_dir, '{}_{}'.format(experiment, env))
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)

        for file in dict_tar_file.values():
            shutil.copy2(file, output_dir)

        for v in dict_tar_perf.values():
            print(v)


        print(files)
        print("---- done for {}\n\n".format(env))
