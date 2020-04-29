import argparse
from spinup.FlexibilityEnv_input.inputload import load_FlexibilityEnv_input
from spinup.run_flexibility import run_experiment
from multiprocessing import Process
import os


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def create_args(env_version, target_arcs):
    return Namespace(
        env_name='F{}-v{}'.format(EXPERIMENT, env_version),
        exp_name='F{}_CH1024-128_ENV{}'.format(EXPERIMENT, env_version),
        algo='ppo',
        cpu=2,
        epochs=400,
        custom_h='1024-128',
        env_version=env_version,
        env_input=ENV_INPUT,
        target_arcs=target_arcs,
        steps_per_epoch=24000,
        num_runs=8,
        seed=None,
        save_freq=8,
        do_checkpoint_eval=True,
        act="tf.nn.relu",
        eval_episodes=50,
        train_v_iters=80,
        train_pi_iters=80,
        eval_temp=1.0,
        train_starting_temp=1.0,
        gamma=0.99,
        env_n_sample=50
    )


def run_in_parallel(*fns):
    proc = []
    for fn in fns:
        p = Process(target=fn)
        p.start()
        proc.append(p)
    for p in proc:
        p.join()


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


# INPUTS = {'10x10b': 'input_ran10x10b_cv0.8.pkl'}

from spinup.FlexibilityEnv_input.FlexibilityEnv_input_files import INPUTS

EXPERIMENT = '10x10b'
ENV_INPUT = INPUTS[EXPERIMENT]

if __name__ == "__main__":
    m, n, mean_c, mean_d, sd_d, profit_mat, target_arcs, fixed_costs, flex_0 = load_FlexibilityEnv_input(
        _get_full_path(ENV_INPUT))
    num_tars_per_script = 2
    tar_list = get_tars_list(num_tars_per_script, target_arcs)

    print(tar_list)

    env_version_list = [3, 4, 5]

    for env_version in env_version_list:
        # create entrypoint script
        # !/bin/bash
        path = 'run_{}_ENV{}.sh'.format(EXPERIMENT, env_version)
        python_string = 'for((i=0;i < {};i++)); do bash run_{}_ENV{}_'.format(len(tar_list),
                                                                               EXPERIMENT,
                                                                               env_version) \
                        + '$' + '{' + 'i' + '}' + '.sh & done'
        with open(path, 'w') as f:
            f.write('#!/bin/bash\n\n')
            f.write(python_string)
        make_executable(path)

        print(python_string)

        # create scripts to be called in parallel
        for idx, target_arcs in enumerate(tar_list):
            target_arcs_string = _get_string(target_arcs)
            python_string = "python -m spinup.run_flexibility \
                            --algo ppo  \
                            --env_name F{}-v{} \
                            --exp_name F{}_CH1024-128_ENV{}  \
                            --cpu 2 \
                            --epochs 400  \
                            --custom_h 1024-128 \
                            --env_version {} \
                            --env_input {} \
                            --target_arcs  {} \
                            --num_runs 4 \
                            --save_freq 8  \
                            --steps_per_epoch 24000 \
                            --do_checkpoint_eval;".format(
                EXPERIMENT,
                env_version,
                EXPERIMENT,
                env_version,
                env_version,
                ENV_INPUT,
                target_arcs_string
            )

            path = 'run_{}_ENV{}_{}.sh'.format(EXPERIMENT, env_version, idx)
            with open(path, 'w') as f:
                f.write('#!/bin/bash\n\n')
                f.write(python_string)

            make_executable(path)

            print(python_string)

    # p1 = Process(target=run_experiment(args1))
    # p1.start()
    # p2 = Process(target=run_experiment(args2))
    # p2.start()
    # p1.join()
    # p2.join()
