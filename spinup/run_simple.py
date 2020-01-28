
from spinup.utils.run_utils import ExperimentGrid
from spinup import ddpg, ppo, vpg

import gym
import tensorflow as tf

def run_experiment(args):
    # def env_fn():
    #     import flexibility  # register flexibility to gym env registry
    #     return gym.make(args.env_name)

    eg = ExperimentGrid(name=args.exp_name)
    eg.add('seed', [10*i for i in range(args.num_runs)])
    eg.add('epochs', args.epochs)
    eg.add('steps_per_epoch', args.steps_per_epoch)
    eg.add('save_freq', args.save_freq)
    eg.add('max_ep_len', 200)
    eg.add('ac_kwargs:activation', eval(args.act), '')
    eg.add('custom_h', args.custom_h)
    eg.add('do_checkpoint_eval', args.do_checkpoint_eval)
    eg.add('env_name', args.env_name)
    eg.add('train_v_iters', args.train_v_iters)

    if args.algo == "ppo":
        eg.add('train_pi_iters', args.train_pi_iters)
        eg.run(ppo)
    elif args.algo == "vpg":
        eg.run(vpg)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="ppo")
    parser.add_argument("--cpu", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument('--steps_per_epoch', type=int, default=5000)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--env_name', type=str, default="Flexibility-v0")
    parser.add_argument('--exp_name', type=str, default='Flexibility-PPO')
    # added custom_h to specify hidden layers with different sizes, e.g., "1024-128".
    parser.add_argument('--custom_h', nargs='+', default=None)
    parser.add_argument('--act', type=str, default="tf.nn.relu")
    parser.add_argument('--do_checkpoint_eval', action='store_true', help="Whether to do evaluation per save frequency")
    parser.add_argument('--train_pi_iters', type=int, default=80, help="# of iterations per each training step for pi")
    parser.add_argument('--train_v_iters', type=int, default=80, help="# of iterations per each training step for v")

    args = parser.parse_args()

    run_experiment(args)
