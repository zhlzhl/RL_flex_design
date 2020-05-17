
from spinup.utils.run_utils import ExperimentGrid
from spinup import ppo, vpg
from spinup.FlexibilityEnv_input.inputload import load_FlexibilityEnv_input

import os
import tensorflow as tf

def run_experiment(args):
    # def env_fn():
    #     import flexibility  # register flexibility to gym env registry
    #     return gym.make(args.env_name)

    eg = ExperimentGrid(name=args.exp_name)
    eg.add('seed', [10*i for i in range(args.num_runs)] if args.seed is None else args.seed)
    eg.add('epochs', args.epochs)
    eg.add('steps_per_epoch', args.steps_per_epoch)
    eg.add('save_freq', args.save_freq)
    eg.add('max_ep_len', 200)
    eg.add('ac_kwargs:activation', eval(args.act), '')
    eg.add('custom_h', args.custom_h)
    eg.add('do_checkpoint_eval', args.do_checkpoint_eval)
    eg.add('eval_episodes', args.eval_episodes)
    eg.add('train_v_iters', args.train_v_iters)
    eg.add('eval_temp', args.eval_temp)
    eg.add('train_starting_temp', args.train_starting_temp)
    eg.add('gamma', args.gamma)
    eg.add('env_version', args.env_version)
    eg.add('env_name', args.env_name)
    eg.add('env_subtract_full_flex', args.env_subtract_full_flex)
    eg.add('meta_learning', args.meta_learning)
    eg.add('lam', args.lam)
    eg.add('early_stop_epochs', args.early_stop_epochs)
    eg.add('save_all_eval', args.save_all_eval)
    if args.episodes_per_epoch is not None:
        eg.add('episodes_per_epoch', args.episodes_per_epoch)

    if args.env_version >= 3:
        # args.file_path = "/home/user/git/spinningup/spinup/FlexibilityEnv/input_m8n12_cv0.8.pkl"
        prefix = os.getcwd().split('RL_flex_design')[0]
        args.file_path = prefix + "RL_flex_design/spinup/FlexibilityEnv_input/{}".format(args.env_input)

        m, n, mean_c, mean_d, sd_d, profit_mat, target_arcs, fixed_costs, flex_0 = load_FlexibilityEnv_input(args.file_path)

        eg.add('env_input', args.file_path)
        eg.add('env_n_sample', args.env_n_sample)

        if args.target_arcs is None:
            eg.add('target_arcs', target_arcs)
        else:  # target_arcs is explicitly specified by the scripts, which overrides the target_arc from the input file
            eg.add('target_arcs', args.target_arcs)

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
    parser.add_argument('--seed', nargs='+', type=int, default=None,
                        help="to directly specify seed instead of using --num_runs.")
    parser.add_argument('--steps_per_epoch', type=int, default=6000)
    parser.add_argument('--episodes_per_epoch', type=int, default=None)
    parser.add_argument('--early_stop_epochs', type=int, default=60,
                        help="The number of consecutive epochs with no eval performance improvement for early stop."
                             "If negative, then do not early stop.")
    parser.add_argument('--save_all_eval', action='store_true',
                        help="Save performance and structure of each episode of evaluation of each checkpoints for "
                             "plotting purpose")


    parser.add_argument('--save_freq', type=int, default=5000)
    parser.add_argument('--do_checkpoint_eval', action='store_true', help="Whether to do evaluation per save frequency")
    parser.add_argument('--eval_episodes', type=int, default=50,
                        help="number of episodes to run during evaluation.")

    parser.add_argument('--custom_h', nargs='+', default=None,
                        help="to specify hidden layers with different sizes, e.g., 1024-128.")
    parser.add_argument('--act', type=str, default="tf.nn.relu")
    parser.add_argument('--train_pi_iters', type=int, default=80, help="# of iterations per each training step for pi")
    parser.add_argument('--train_v_iters', type=int, default=80, help="# of iterations per each training step for v")
    parser.add_argument("--eval_temp", type=float, default=1.0,
                        help="temperature used during evaluation. ")
    parser.add_argument("--train_starting_temp", type=float, default=1.0,
                        help="starting temperature used during training. If larger than 1.0, training temperature "
                             "decreases to 1.0 in the first 1/3 of epochs. ")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor of PPO. ")
    parser.add_argument("--lam", type=float, default=0.97,
                        help="Lambda for GAE-Lambda. (Always between 0 and 1, close to 1.). ")

    parser.add_argument('--exp_name', type=str, default='Flexibility-PPO')

    parser.add_argument('--env_name', type=str, default="Flexibility")
    parser.add_argument("--env_version", type=int, default=1,
                        help="version of env. env1 is both add/drop until reached target_arcs. env2 is both add/drop "
                             "until taken target_arcs steps. env3 is a FlexibilityEnv created from an input file, "
                             "e.g., input_m8n12_cv0.8.pkl")
    parser.add_argument("--env_n_sample", type=int, default=50,
                        help="number of samples to draw during structure performance evaluation")
    parser.add_argument("--env_input", type=str, default=None,
                        help="input file specifying settings for FlexibilityEnv")
    parser.add_argument("--env_subtract_full_flex", action='store_true',
                        help="Whether to substract full flexibility performance from structure performance in reward "
                             "to reduce variance")
    parser.add_argument("--meta_learning", action='store_true',
                        help="Whether to do meta learning")

    parser.add_argument('--target_arcs', type=int, nargs='+', default=None,
                        help="to specify target arcs with different values, e.g., 27 29 31 33."
                             "This would override the target_arc from input file in env_version=3")



    args = parser.parse_args()

    run_experiment(args)
