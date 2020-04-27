import numpy as np
from spinup.utils.mpi_tools import mpi_statistics_scalar
from spinup.FlexibilityEnv.FlexibilityEnv import FlexibilityEnv
import os
import pickle
from spinup.FlexibilityEnv_input.inputload import load_FlexibilityEnv_input
from spinup.FlexibilityEnv.FlexibilityEnv import expected_sales_for_structure


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


def save_best_eval(best_performance, best_structure, epoch, env_name, log_dir, seed):
    pickle_data = (best_performance, best_structure, epoch)
    with open(os.path.join(log_dir, "best_eval_performance_n_structure_{}_s{}.pkl".format(env_name, seed)), 'wb') as f:
        pickle.dump(pickle_data, f)


# to be used for testing policy during training
def eval_and_save_best_model(
        best_eval_AverageEpRet, best_eval_StdEpRet, eval_logger, train_logger, tb_logger, epoch,
        env_name, env_version, env_input, target_arcs, get_action, render=True, n_sample=5000,
        num_episodes=50, save=False, seed=0
):
    # for envs with different versions of different n_sample, choose a corresponding env with indicated n_sample for
    # testing. Only useful for env_version == 1 or 2. This is not effective for env_version == 3
    eval_env_name = get_new_env_name(env_name, n_sample, env_version)
    mean, std, _, _, best_performance, best_structure = run_policy_with_custom_logging(eval_env_name,
                                                                                       env_version,
                                                                                       env_input,
                                                                                       target_arcs,
                                                                                       get_action,
                                                                                       logger=eval_logger,
                                                                                       tb_logger=tb_logger,
                                                                                       epoch=epoch,
                                                                                       max_ep_len=70,
                                                                                       render=True,
                                                                                       n_sample=n_sample,
                                                                                       n_episodes=num_episodes)

    # # logs for debugging purpose
    # eval_logger.log("epoch {}, to save? {}".format(epoch, save), color='white')
    # if save:
    #     eval_logger.log('Before eval | epoch {} | best_eval_AverageEpRet {} | best_eval_StdEpRet {}'
    #                     .format(epoch, best_eval_AverageEpRet, best_eval_StdEpRet), color='white')
    #     eval_logger.log('After eval | epoch {} | mean {} | std {}'
    #                     .format(epoch, mean, std), color='white')
    #     eval_logger.log('best_eval_AverageEpRet < mean {} | std <= best_eval_StdEpRet * 1.5 {} | save {}'.
    #                     format(best_eval_AverageEpRet < mean, std <= best_eval_StdEpRet * 1.5, save))
    saved = False
    if best_eval_AverageEpRet < mean:
        best_eval_AverageEpRet = mean
        if (std <= best_eval_StdEpRet * 1.5) and save:
            # save the best model so far to simple_save999999. This is a hack to leverage the available codes to save
            # the best model identified by episode 999999
            train_logger.save_state({'env_name': eval_env_name}, itr=999999)

            # save the best_performance and best_structure across the different runs during evaluation
            save_best_eval(best_performance, best_structure, epoch, env_name, log_dir=eval_logger.output_dir, seed=seed)
            saved = True

    if best_eval_StdEpRet > std:
        best_eval_StdEpRet = std

    return best_eval_AverageEpRet, best_eval_StdEpRet, saved


def get_new_env_name(env_name, n_sample, env_version):
    new_env_name = env_name
    if env_version == 1 or env_version == 2:
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

    if env_version in (3, 4, 41, 5):
        new_env_name = env_name.split("_SP")[0] + "_SP5000_v{}".format(env_version)
        print("using new env {} to evaluate performance".format(new_env_name))
    return new_env_name


def _parse_attributes(env_name):
    # example env_name Flexibility20x20T40_SP50-v0, Flexibility10x10T15-v0
    if '-' in env_name:
        env_name = env_name.split('-')[0]

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


def get_custom_env_fn(env_name, env_version=None, target_arcs=None, env_input=None, env_n_sample=None):
    if env_version in (1, 2):
        # parse FlexibilityEnv settings from env_name
        n_plant, n_product, target_arcs, n_sample = _parse_attributes(env_name)

        class CustomFlexibilityEnv(FlexibilityEnv):
            def __init__(self):
                super().__init__(n_plant=n_plant, n_product=n_product,
                                 target_arcs=target_arcs, n_sample=n_sample, env_version=env_version)
                print(
                    'using custom env: {} | n_plant: {} | n_product: {} | target_arcs: {} | n_sample: {} | env_version: {}'
                        .format(env_name, n_plant, n_product, target_arcs, n_sample, env_version))

    if env_version in (3, 4, 41, 5):
        # load FlexibilityEnv settings from env_input
        n_plant, n_product, mean_c, mean_d, sd_d, profit_mat, _, fixed_costs, flex_0 = load_FlexibilityEnv_input(env_input)

        def to_numpy_array(obj):
            if isinstance(obj, list):
                obj_array = np.asarray(obj, dtype=np.float32)
                return obj_array
            else:
                return obj

        mean_c = to_numpy_array(mean_c)
        mean_d = to_numpy_array(mean_d)
        sd_d = to_numpy_array(sd_d)

        class CustomFlexibilityEnv(FlexibilityEnv):
            def __init__(self):
                super().__init__(n_plant=n_plant,
                                 n_product=n_product,
                                 target_arcs=target_arcs,  # for env_version=3, target_arcs is passed into the function call
                                 n_sample=env_n_sample,
                                 capacity_mean=mean_c,
                                 env_version=env_version,
                                 demand_mean=mean_d,
                                 demand_std=sd_d,
                                 profit_matrix=profit_mat,
                                 fixed_costs=fixed_costs,
                                 starting_structure=flex_0)
                print('using env: {} | n_plant: {} | n_product: {} | target_arcs: {} | n_sample: {} | env_version: {}'
                      .format(env_name, n_plant, n_product, target_arcs, env_n_sample, env_version))

    return CustomFlexibilityEnv


def run_policy_with_custom_logging(env_name, env_version, env_input, target_arcs,
                                   get_action, logger, tb_logger, epoch,
                                   max_ep_len=None, n_episodes=150, render=True, n_sample=5000):
    if env_version in (1, 2):
        env = get_custom_env_fn(env_name, env_version)()
    else:  # env_version in (3, 4, 41, 5):
        env = get_custom_env_fn(env_name,
                                env_version,
                                target_arcs,
                                env_input,
                                env_n_sample=n_sample)()

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
            logger.store(EpRet=ep_ret, EpLen=ep_len, EpTotalArcs=env.adjacency_matrix.sum())
            print('Episode %d \t EpRet %.3f \t EpLen %d' % (n, ep_ret, ep_len))

            if best_performance < ep_ret:
                best_performance = ep_ret
                best_structure = np.squeeze(o).reshape(n_plant, n_product)
                # save_best_eval(best_performance, best_structure, epoch, env_name, log_dir=logger.output_dir)

            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1

    mean, std, min, max = log_key_to_tb(tb_logger, logger, epoch, key="EpRet", with_min_and_max=True, eval=True)

    log_key_to_tb(tb_logger, logger, epoch, key="EpLen", with_min_and_max=False, eval=True)
    log_key_to_tb(tb_logger, logger, epoch, key="EpTotalArcs", with_min_and_max=False, eval=True)


    ### below is for debugging
    if env_version in (3, 4, 41, 5):
        # evaluate structure performance
        structure_performance, _ = expected_sales_for_structure(best_structure,
                                                                env.n_sample,
                                                                env.capacity_mean,
                                                                demand_mean=env.demand_mean,
                                                                demand_std=env.demand_std,
                                                                flow_profits=env.profit_matrix,
                                                                fixed_costs=env.fixed_costs)
        ended_fixed_costs = np.sum(np.multiply(env.fixed_costs, best_structure))
        starting_fixed_costs = np.sum(np.multiply(env.fixed_costs, env.starting_structure))
        fixed_costs = - (ended_fixed_costs - starting_fixed_costs)
        best_performance_directly_computed = structure_performance + fixed_costs

        # print("best_struct_perf_w_fc_dc {} | struct_perf {} | fixed_cost {} | starting_fc {} | ending_fc {} | n_sample {}".format(
        #     best_performance_directly_computed,
        #     structure_performance,
        #     fixed_costs,
        #     starting_fixed_costs,
        #     ended_fixed_costs,
        #     env.n_sample
        # ))
        #
        # print('final structure')
        # print(best_structure)
        #
        # print('fixed cost')
        # print(env.fixed_costs)

        logger.store(BestStructPerf=best_performance_directly_computed)
        log_key_to_tb(tb_logger, logger, epoch, key="BestStructPerf", with_min_and_max=False, eval=True)
        logger.log_tabular('BestStructPerf', with_min_and_max=True)

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.log_tabular('EpTotalArcs', with_min_and_max=True)
    logger.dump_tabular()

    env.close()
    del env

    return mean, std, min, max, best_performance, best_structure


class DummyActionStats:
    def __init__(self, epoch, env_name, n_plant, n_product, allowed_steps):
        self.env_name = env_name
        self.dummy_action_count = 0
        self.dummy_action_steps = []
        self.total_action_count = 0
        self.game_done = False
        self.env_n_plant = n_plant
        self.env_n_product = n_product
        self.env_allowed_steps = allowed_steps

        self.epoch = None
        self.episode_dummy_counts = {}
        self.episode_dummy_steps_ratio = {}

    def update(self, episode_step, action, done):
        self.total_action_count += 1
        if action == self.env_n_plant * self.env_n_product:
            self.dummy_action_count += 1
            self.dummy_action_steps.append(episode_step)

        self.game_done = done

        if done:
            # compute stats
            pass


    def reset(self):
        self.dummy_action_count = 0
        self.game_done = False