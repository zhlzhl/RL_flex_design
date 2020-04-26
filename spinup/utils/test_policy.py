import time
import joblib
import os
import os.path as osp
import tensorflow as tf
from spinup import EpochLogger
from spinup.utils.logx import restore_tf_graph
import gym
import flexibility
from spinup.utils.custom_utils import get_custom_env_fn


def load_policy(fpath, itr='last', deterministic=False, eval_temp=1.0, use_temp=True, env_name=None, env_version=1):
    # handle which epoch to load from
    if itr == 'last':
        saves = [int(x[11:]) for x in os.listdir(fpath) if 'simple_save' in x and len(x) > 11]
        itr = '%d' % max(saves) if len(saves) > 0 else ''
    else:
        itr = '%d' % itr

    # load the things!
    sess = tf.Session()
    model = restore_tf_graph(sess, osp.join(fpath, 'simple_save' + itr))

    # get the correct op for executing actions
    if deterministic and 'mu' in model.keys():
        # 'deterministic' is only a valid option for SAC policies
        print('Using deterministic action op.')
        action_op = model['mu']
    else:
        print('Using default action op.')
        action_op = model['pi']

    # make function for producing an action given a single state
    if not use_temp:
        get_action = lambda x: sess.run(action_op, feed_dict={model['x']: x[None, :]})[0]
    else:
        get_action = lambda x: sess.run(action_op, feed_dict={model['x']: x[None, :],
                                                              model['temperature']: eval_temp})[0]

    if env_name is None:
        # try to load environment from save
        # (sometimes this will fail because the environment could not be pickled)
        try:
            state = joblib.load(osp.join(fpath, 'vars' + itr + '.pkl'))
            env = state['env']
        except:
            env = None
    else:
        # env = (lambda: gym.make(env_name))()
        if args.env_version in (1, 2):
            env = get_custom_env_fn(env_name, env_version)()
        if args.env_version in (3, 4):
            env = get_custom_env_fn(env_name, env_version, target_arcs=args.target_arcs, env_input=args.env_input,
                                    env_n_sample=5000)()

    return env, get_action


def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True):
    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    logger = EpochLogger()
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-3)

        a = get_action(o)
        o, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpLen %d' % (n, ep_ret, ep_len))
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=100)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    parser.add_argument('--eval_temp', type=float, default=1.0)
    parser.add_argument('--use_temp', action='store_true')
    parser.add_argument('--env_name', type=str, default=None,
                        help='manually specify an env for testing policy. this is optional. ')
    parser.add_argument('--env_version', type=int, default=1,
                        help="version of env. env1 is both add/drop until reached target_arcs. env2 is both add/drop "
                             "until taken target_arcs steps.")
    parser.add_argument("--env_input", type=str, default=None,
                        help="input file specifying settings for FlexibilityEnv")

    parser.add_argument('--target_arcs', type=int, default=None,
                        help="to specify the number of target_arcs for FlexibilityEnv")

    args = parser.parse_args()
    env, get_action = load_policy(args.fpath,
                                  args.itr if args.itr >= 0 else 'last',
                                  args.deterministic,
                                  args.eval_temp,
                                  use_temp=args.use_temp,
                                  env_name=args.env_name,
                                  env_version=args.env_version)
    run_policy(env, get_action, args.len, args.episodes, not (args.norender))
