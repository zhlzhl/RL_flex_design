import numpy as np
from spinup.utils.mpi_tools import mpi_statistics_scalar
from spinup.utils.FlexibilityEnv import FlexibilityEnv
import time
import os
import pickle
import gym
import flexibility

import json


# to be used for logging to tensorboard
def log_key_to_tb(tb_logger, logger, epoch, key, with_min_and_max=False, eval=False):
    mean, std, min, max = get_stats(logger, key=key, with_min_and_max=with_min_and_max)
    if not eval:  # normal logging during training
        if with_min_and_max:
            tb_logger.log_scalar(tag="{}-Average".format(key), value=mean, step=epoch)
            tb_logger.log_scalar(tag="{}-Std".format(key), value=std, step=epoch)
            tb_logger.log_scalar(tag="{}-Max".format(key), value=max, step=epoch)
            tb_logger.log_scalar(tag="{}-Min".format(key), value=min, step=epoch)
        else:
            tb_logger.log_scalar(tag=key, value=mean, step=epoch)
    else:  # logging for evaluating(testing) policy during training
        if with_min_and_max:
            tb_logger.log_scalar(tag="Eval_{}-Average".format(key), value=mean, step=epoch)
            tb_logger.log_scalar(tag="Eval_{}-Std".format(key), value=std, step=epoch)
            tb_logger.log_scalar(tag="Eval_{}-Max".format(key), value=max, step=epoch)
            tb_logger.log_scalar(tag="Eval_{}-Min".format(key), value=min, step=epoch)
        else:
            tb_logger.log_scalar(tag="Eval_{}".format(key), value=mean, step=epoch)

    return mean, std, min, max


# to be used for logging to tensorboard
def get_stats(logger, key, with_min_and_max=True):
    v = logger.epoch_dict[key]
    vals = np.concatenate(v) if isinstance(v[0], np.ndarray) and len(v[0].shape) > 0 else v
    stats = mpi_statistics_scalar(vals, with_min_and_max=with_min_and_max)
    if with_min_and_max:
        return stats  # mean, std, min, max
    else:
        return stats[0], None, None, None


def save_best_eval(best_performance, best_structure, epoch, env_name, log_dir):
    pickle_data = (best_performance, best_structure, epoch)
    with open(os.path.join(log_dir, "best_eval_performance_n_structure_{}.pkl".format(env_name)), 'wb') as f:
        pickle.dump(pickle_data, f)


# to be used for testing policy during training
def eval_and_save_best_model(
        best_eval_AverageEpRet, best_eval_StdEpRet, eval_logger, train_logger, tb_logger, epoch,
        env_name, get_action, render=True, n_sample=5000, num_episodes=50, save=False
):
    # for envs with different versions of different n_sample, choose a corresponding env with indicated n_sample for
    # testing
    eval_env_name = get_new_env_name(env_name, n_sample)
    mean, std, _, _, best_performance, best_structure = run_policy_with_custom_logging(eval_env_name, get_action,
                                                                                       logger=eval_logger,
                                                                                       tb_logger=tb_logger,
                                                                                       epoch=epoch,
                                                                                       max_ep_len=100,
                                                                                       render=True,
                                                                                       n_sample=n_sample,
                                                                                       n_episodes=num_episodes)

    if best_eval_AverageEpRet <= mean:
        best_eval_AverageEpRet = mean
        if (std <= best_eval_StdEpRet * 1.5) and save:
            # save the best model so far to simple_save999999. This is a hack to leverage the available codes to save
            # the best model identified by episode 999999
            # env = (lambda: gym.make(env_name))()
            # train_logger.save_state({'env': env}, itr=999999)
            # eval_env = (lambda: gym.make(eval_env_name))()
            # eval_env = get_custom_env_fn(eval_env_name)()
            # train_logger.save_state({'env': eval_env_name}, itr=999999)
            train_logger.custom_save_state({'env': eval_env_name}, itr=999999)

            # save the best_performance and best_structure across the different runs during evaluation
            save_best_eval(best_performance, best_structure, epoch, env_name, log_dir=eval_logger.output_dir)

            # del eval_env

    if best_eval_StdEpRet > std:
        best_eval_StdEpRet = std

    return best_eval_AverageEpRet, best_eval_StdEpRet


def get_new_env_name(env_name, n_sample):
    new_env_name = env_name

    if "_SP" in env_name and "-v0" in env_name:
        n_sample_training = env_name.split("_SP")[1].split("-v0")[0]
        n_sample_training = int(n_sample_training)
        if n_sample_training != n_sample:
            if n_sample == 5000:
                new_env_name = env_name.split("_SP")[0] + "-v0"
                print("using new env {} to evaluate performance".format(new_env_name))
            elif (n_sample in {1, 10, 50, 100}) and ('20x20' in env_name):
                new_env_name = env_name.split("_SP")[0] + "_SP{}-v0".format(n_sample)
                print("using new env {} to evaluate performance".format(new_env_name))
            else:
                raise NotImplementedError("The eval env {} with n_sample = {} is not implemented"
                                          .format(env_name, n_sample))

    return new_env_name


def _parse_attributes(env_name):

    assert 'T' in env_name, 'wrong format of env_name: {}. Please make sure input an env in the form of ' \
                            'Flexibility10x10T30_SP50-v0, or F20x20T20_SP10, or F3x3T4-v0, etc.'.format(env_name)
    # example env_name Flexibility20x20T40_SP50-v0, Flexibility10x10T15-v0
    if '-v0' in env_name:
        env_name = env_name.split('-')[0]  # remove '-v0'

    if '_SP' in env_name:
        splits = env_name.split('_SP')
        n_sample = int(splits[1])
        env_name = splits[0]  # got Flexibility20x20T40
    else:
        n_sample = 5000  # default value

    env_name = env_name[11:]  # removed 'Flexibility' to get 20x20T40

    splits = env_name.split('x')
    n_plant = int(splits[0])

    splits = splits[1].split('T')  # got 20T40
    n_product = int(splits[0])
    target_arcs = int(splits[1])

    return n_plant, n_product, target_arcs, n_sample


def get_custom_env_fn(env_name):
    n_plant, n_product, target_arcs, n_sample = _parse_attributes(env_name)

    class CustomFlexibilityEnv(FlexibilityEnv):
        def __init__(self):
            super().__init__(n_plant=n_plant, n_product=n_product,
                             target_arcs=target_arcs, n_sample=n_sample, name=env_name)
            print('using custom env: {} | n_plant: {} | n_product: {} | target_arcs: {} | n_sample: {}'
                  .format(env_name, n_plant, n_product, target_arcs, n_sample))

    return CustomFlexibilityEnv


def insert_target_arcs_to_env_name(env_name, target_arcs):
    assert not ('T' in env_name), "When --target_arcs is used, T should not appear in env_name: {}".format(env_name)
    # insert target arcs to env_name
    if '_SP' in env_name:
        # insert target arcs in front of _SP
        index = env_name.find('_SP')
        env_name = env_name[:index] + 'T{}'.format(target_arcs) + env_name[index:]
    elif '-v0' in env_name:
        # insert target arcs in front '-v0'
        index = env_name.find('-v0')
        env_name = env_name[:index] + 'T{}'.format(target_arcs) + env_name[index:]
    else:
        # append to the end
        env_name = env_name + 'T{}'.format(target_arcs)

    return env_name


def run_policy_with_custom_logging(env_name, get_action, logger, tb_logger, epoch,
                                   max_ep_len=None, n_episodes=50, render=True, n_sample=5000):
    # env = (lambda: gym.make(env_name))()
    env = get_custom_env_fn(env_name)()

    n_plant = env.n_plant
    n_product = env.n_product

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    best_performance = 0.0
    best_structure = None

    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    while n < n_episodes:
        if render:
            env.render()
            # time.sleep(1e-3)

        a = get_action(o)
        o, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpLen %d' % (n, ep_ret, ep_len))

            if best_performance <= ep_ret:
                best_performance = ep_ret
                best_structure = np.squeeze(o).reshape(n_plant, n_product)
                # save_best_eval(best_performance, best_structure, epoch, env_name, log_dir=logger.output_dir)

            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1

    mean, std, min, max = log_key_to_tb(tb_logger, logger, epoch, key="EpRet", with_min_and_max=True, eval=True)

    log_key_to_tb(tb_logger, logger, epoch, key="EpLen", with_min_and_max=False, eval=True)

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()

    env.close()
    del env

    return mean, std, min, max, best_performance, best_structure


def main():
    env_name = "F20x20"
    print(insert_target_arcs_to_env_name(env_name, target_arcs=10))

    print(insert_target_arcs_to_env_name('Flexibility10x10', target_arcs=3))

    print(insert_target_arcs_to_env_name('Flexibility2x2_SP10-v0', target_arcs=2))

    print(insert_target_arcs_to_env_name('Flexibility20x20_SP10', target_arcs=2))




if __name__ == '__main__':
    main()