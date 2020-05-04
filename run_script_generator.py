import argparse
from spinup.FlexibilityEnv_input.inputload import load_FlexibilityEnv_input
from spinup.run_flexibility import run_experiment
from multiprocessing import Process
import os
import numpy as np
import statistics


def _get_full_path(env_input):
    prefix = os.getcwd().split('RL_flex_design')[0]
    return prefix + "RL_flex_design/spinup/FlexibilityEnv_input/{}".format(env_input)


def get_tars_list(num_tars_per_script, target_arcs):
    list_tars = []
    count = 0
    while (count + 1) * num_tars_per_script < len(target_arcs):
        list_tars.append(target_arcs[count * num_tars_per_script: (count + 1) * num_tars_per_script])
        count += 1
    list_tars.append(target_arcs[count * num_tars_per_script:])
    return list_tars


def _get_string(target_arcs):
    string = ''
    for tar in target_arcs:
        string += '{} '.format(tar)

    return string


def make_executable(path):
    mode = os.stat(path).st_mode
    mode |= (mode & 0o444) >> 2  # copy R bits to X
    os.chmod(path, mode)


def get_seed_str(starting_seed, num_runs):
    seed_str = ""

    for i in range(num_runs):
        seed_str += "{} ".format(starting_seed + i * 10)

    return seed_str.strip()


def get_input(experiment):
    if '-' in experiment:
        exp = experiment.split('-')[0]
    else:
        exp = experiment
    return INPUTS[exp]


# INPUTS = {'10x10b': 'input_ran10x10b_cv0.8.pkl'}

from spinup.FlexibilityEnv_input.FlexibilityEnv_INPUTS import INPUTS


def generate_scripts_for_multiple_target_arcs(experiment, env_input, env_version_list, epoch_episodes,
                                              num_tars_per_script, num_batches, num_runs, gamma=None, lam=None):
    m, n, mean_c, mean_d, sd_d, profit_mat, target_arcs, fixed_costs, flex_0 = load_FlexibilityEnv_input(
        _get_full_path(env_input))
    print("number of existing arcs {}".format(flex_0.sum()))
    tar_list = get_tars_list(num_tars_per_script, target_arcs)
    print(tar_list)

    for batch in range(num_batches):
        starting_seed = 100 * batch
        for env_version in env_version_list:
            # create entrypoint script
            # !/bin/bash
            path = 'run_{}_ENV{}_batch{}_entrypoint.sh'.format(experiment, env_version, batch)
            python_string = 'for((i=0;i < {};i++)); do bash run_{}_ENV{}_batch{}_'.format(len(tar_list),
                                                                                          experiment,
                                                                                          env_version,
                                                                                          batch) \
                            + '$' + '{' + 'i' + '}' + '.sh & done'
            with open(path, 'w') as f:
                f.write('#!/bin/bash\n\n')
                f.write(python_string)
            make_executable(path)

            print(python_string)

            # create scripts to be called in parallel
            for idx, target_arcs in enumerate(tar_list):

                assert len(target_arcs) >= 1
                if len(target_arcs) == 1:
                    # append an additional target arc to target_arcs to maintain the right directory name
                    new_t = target_arcs[0] + 3
                    target_arcs.append(new_t)

                target_arcs_string = _get_string(target_arcs)
                python_string = "python -m spinup.run_flexibility   \\\n   \
                                --algo ppo    \\\n   \
                                --env_name F{}-v{}   \\\n   \
                                --exp_name F{}_CH1024-128_ENV{}    \\\n   \
                                --cpu 2   \\\n   \
                                --epochs 800    \\\n   \
                                --custom_h 1024-128   \\\n   \
                                --env_version {}   \\\n   \
                                --env_input {}   \\\n   \
                                --target_arcs  {}   \\\n   \
                                --seed {}   \\\n   \
                                --save_freq 10    \\\n   \
                                --steps_per_epoch {}   \\\n   \
                                --do_checkpoint_eval ".format(
                    experiment,
                    env_version,
                    experiment,
                    env_version,
                    env_version,
                    env_input,
                    target_arcs_string,
                    get_seed_str(starting_seed, num_runs),
                    int(np.ceil((int(statistics.mean(target_arcs)) - flex_0.sum()) * epoch_episodes))
                )

                if gamma is not None:
                    python_string += ' --gamma {}   '.format(gamma)

                if lam is None:
                    python_string += ';'
                else:
                    python_string += ' --lam {};'.format(lam)

                path = 'run_{}_ENV{}_batch{}_{}.sh'.format(experiment, env_version, batch, idx)
                with open(path, 'w') as f:
                    f.write('#!/bin/bash\n\n')
                    f.write(python_string)

                make_executable(path)

                print(python_string)


def generate_scripts_for_one_target_arcs(experiment, env_input, env_version_list, epoch_episodes, target_arcs,
                                         num_batches, num_runs, starting_seed, gamma=None, lam=None):
    m, n, mean_c, mean_d, sd_d, profit_mat, _, fixed_costs, flex_0 = load_FlexibilityEnv_input(
        _get_full_path(env_input))
    print("number of existing arcs {}".format(flex_0.sum()))

    for env_version in env_version_list:
        # create entrypoint script
        # !/bin/bash
        path = 'run_{}_ENV{}_tar{}_entrypoint.sh'.format(experiment, env_version, target_arcs)
        python_string = 'for((i=0;i < {};i++)); do bash run_{}_ENV{}_tar{}_'.format(num_runs,
                                                                                    experiment,
                                                                                    env_version,
                                                                                    target_arcs) \
                        + '$' + '{' + 'i' + '}' + '.sh & done'
        with open(path, 'w') as f:
            f.write('#!/bin/bash\n\n')
            f.write(python_string)
        make_executable(path)

        print(python_string)

        # create scripts to be called in parallel
        for idx in range(num_runs):

            # python_string = "python -m spinup.run_flexibility \
            #                 --algo ppo  \
            #                 --env_name F{}-v{} \
            #                 --exp_name F{}_CH1024-128_ENV{}_tar{}  \
            #                 --cpu 2 \
            #                 --epochs 800  \
            #                 --custom_h 1024-128 \
            #                 --env_version {} \
            #                 --env_input {} \
            #                 --target_arcs  {} \
            #                 --seed {} \
            #                 --save_freq 10  \
            #                 --steps_per_epoch {} \
            #                 --do_checkpoint_eval ".format(
            #     experiment,
            #     env_version,
            #     experiment,
            #     env_version,
            #     target_arcs,
            #     env_version,
            #     env_input,
            #     target_arcs,
            #     starting_seed + 10 * idx,
            #     int(np.ceil((target_arcs - flex_0.sum()) * epoch_episodes))
            # )

            python_string = "python -m spinup.run_flexibility  \\\n \
                            --algo ppo  \\\n \
                            --env_name F{}-v{}  \\\n \
                            --exp_name F{}_CH1024-128_ENV{}_tar{}  \\\n \
                            --cpu 2 \\\n \
                            --epochs 800  \\\n \
                            --custom_h 1024-128  \\\n \
                            --env_version {}  \\\n \
                            --env_input {}  \\\n \
                            --target_arcs  {}  \\\n \
                            --seed {}  \\\n \
                            --save_freq 10   \\\n \
                            --steps_per_epoch {}  \\\n \
                            --do_checkpoint_eval ".format(
                experiment,
                env_version,
                experiment,
                env_version,
                target_arcs,
                env_version,
                env_input,
                target_arcs,
                starting_seed + 10 * idx,
                int(np.ceil((target_arcs - flex_0.sum()) * epoch_episodes))
            )

            if gamma is not None:
                python_string += ' --gamma {}   '.format(gamma)

            if lam is None:
                python_string += ';'
            else:
                python_string += ' --lam {};'.format(lam)

            path = 'run_{}_ENV{}_tar{}_{}.sh'.format(experiment, env_version, target_arcs, idx)
            with open(path, 'w') as f:
                f.write('#!/bin/bash\n\n')
                f.write(python_string)

            make_executable(path)

            print(python_string)


if __name__ == "__main__":
    # specify parameters
    experiment = '8x16JG'
    env_input = get_input(experiment)
    env_version_list = [5]
    epoch_episodes = 800
    num_tars_per_script = 2
    # the number of entrypoints to be created with different seeds and everything else the same, the purpose is to do more parallelization
    num_batches = 2
    # the number of runs with different seed for each target arc
    num_runs = 4
    gamma = 0.99
    lam = 0.999

    experiment += "-gamma{}-lam{}".format(gamma, lam)

    # to generate scripts for a list of target_arcs. make sure the sub_groups of target_arcs in each script is at least two.
    # this allows log directories to be created with tar_arc specified in the directory name
    generate_scripts_for_multiple_target_arcs(experiment, env_input, env_version_list, epoch_episodes,
                                              num_tars_per_script, num_batches, num_runs, gamma, lam)

    # # to generate scripts for one particular target_arcs but with different seeds, which will then be called in parallel
    # target_arcs = 31
    # num_runs = 4
    # starting_seed = 500
    # generate_scripts_for_one_target_arcs(experiment, env_input, env_version_list, epoch_episodes,
    #                                      target_arcs, num_batches, num_runs, starting_seed, gamma, lam)
