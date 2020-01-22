import numpy as np
from spinup.utils.mpi_tools import mpi_statistics_scalar
import time
import os
import pickle
import gym
import flexibility

import json


# # to be used for logging to tensorboard
# def log_to_tb(tb_logger, logger, epoch, eval=False):
#     log_key_to_tb(tb_logger, logger, epoch, key="EpRet", with_min_and_max=True)
#     log_key_to_tb(tb_logger, logger, epoch, key="EpLen", with_min_and_max=False)
#     log_key_to_tb(tb_logger, logger, epoch, key="VVals", with_min_and_max=True)
#     log_key_to_tb(tb_logger, logger, epoch, key="LossPi", with_min_and_max=False)
#     log_key_to_tb(tb_logger, logger, epoch, key="LossV", with_min_and_max=False)
#     log_key_to_tb(tb_logger, logger, epoch, key="DeltaLossPi", with_min_and_max=False)
#     log_key_to_tb(tb_logger, logger, epoch, key="DeltaLossV", with_min_and_max=False)
#     log_key_to_tb(tb_logger, logger, epoch, key="Entropy", with_min_and_max=False)
#     log_key_to_tb(tb_logger, logger, epoch, key="KL", with_min_and_max=False)
#     log_key_to_tb(tb_logger, logger, epoch, key="ClipFrac", with_min_and_max=False)
#     log_key_to_tb(tb_logger, logger, epoch, key="StopIter", with_min_and_max=False)


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
        env_name, get_action, render=True
):
    mean, std, _, _ = run_policy_with_custom_logging(env_name, get_action, logger=eval_logger, tb_logger=tb_logger,
                                                     epoch=epoch, max_ep_len=None, render=True)

    if best_eval_AverageEpRet < mean:
        best_eval_AverageEpRet = mean
        if std < best_eval_StdEpRet * 1.05:
            # save the best model so far to simple_save999999. This is a hack to leverage the available codes to save
            # the best model identified by episode 999999
            env = (lambda: gym.make(env_name))()
            train_logger.save_state({'env': env}, itr=999999)

            # close the pyglet window and delete env
            env.close()
            del env

    if best_eval_StdEpRet > std:
        best_eval_StdEpRet = std

    return best_eval_AverageEpRet, best_eval_StdEpRet


def run_policy_with_custom_logging(env_name, get_action, logger, tb_logger, epoch,
                                   max_ep_len=None, num_episodes=50, render=True):
    env = (lambda: gym.make(env_name))()
    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    best_performance = 0.0
    best_structure = None

    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    while n < num_episodes:
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

            if best_performance < ep_ret:
                best_performance = ep_ret
                best_structure = np.squeeze(o).reshape(10, 10)
                save_best_eval(best_performance, best_structure, epoch, env_name, log_dir=logger.output_dir)

            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1

    mean, std, min, max = log_key_to_tb(tb_logger, logger, epoch, key="EpRet", with_min_and_max=True, eval=True)

    log_key_to_tb(tb_logger, logger, epoch, key="EpLen", with_min_and_max=False, eval=True)

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()

    env.close()
    del env

    return mean, std, min, max
