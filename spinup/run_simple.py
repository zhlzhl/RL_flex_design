
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
    # eg.add('use_custom_env', args.use_custom_env)
    eg.add('env_name', args.env_name)
    # eg.add('n_sample', args.n_sample)
    eg.add('eval_episodes', args.eval_episodes)
    eg.add('train_v_iters', args.train_v_iters)
    eg.add('eval_temp', args.eval_temp)
    eg.add('train_starting_temp', args.train_starting_temp)
    eg.add('train_continuously', args.train_continuously)
    eg.add('transfer_learning_base_model_path', args.transfer_learning_base_model_path)
    if args.target_arcs is not None:
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
    parser.add_argument('--steps_per_epoch', type=int, default=6000)
    parser.add_argument('--save_freq', type=int, default=5000)
    parser.add_argument('--env_name', type=str, default="Flexibility-v0")
    parser.add_argument('--exp_name', type=str, default='Flexibility-PPO')
    parser.add_argument('--eval_episodes', type=int, default=50,
                        help="number of episodes to run during evaluation.")

    # added custom_h to specify hidden layers with different sizes, e.g., "1024-128".
    parser.add_argument('--custom_h', nargs='+', default=None)
    parser.add_argument('--act', type=str, default="tf.nn.relu")
    parser.add_argument('--do_checkpoint_eval', action='store_true', help="Whether to do evaluation per save frequency")
    parser.add_argument('--train_continuously', action='store_true',
                        help="to use the best saved model in the previous experiment to start training")

    # parser.add_argument('--use_custom_env', action='store_true',
    #                     help="whether to use custom env instead of env defined in gym format")
    parser.add_argument('--train_pi_iters', type=int, default=80, help="# of iterations per each training step for pi")
    parser.add_argument('--train_v_iters', type=int, default=80, help="# of iterations per each training step for v")
    parser.add_argument("--eval_temp", type=float, default=1.0,
                        help="temperature used during evaluation. ")
    parser.add_argument("--train_starting_temp", type=float, default=1.0,
                        help="starting temperature used during training. If larger than 1.0, training temperature "
                             "decreases to 1.0 in the first half of epochs. ")
    parser.add_argument('--target_arcs', type=int, nargs='+', default=None, help="a list of target arcs")
    parser.add_argument('--transfer_learning_base_model_path', type=str, default=None,
                        help="path to a trained model to be used as the base model of transfer learning. saved model "
                             "in simple_save999999 in the directory of path is used. ")

    args = parser.parse_args()

    run_experiment(args)
