import numpy as np
import tensorflow as tf
import gym
import time
import spinup.algos.ppo.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.tensorboard_logging import Logger
from datetime import datetime
from spinup.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from spinup.utils.custom_utils import *
from spinup.utils.logx import restore_tf_graph
import copy


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.temperature_buf = np.zeros(size, dtype=np.float32)

        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp, temperature):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.temperature_buf[self.ptr] = temperature
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]  # TD residual -- equation (11) in GAE paper
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)  # eq (14) of GAE paper

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.obs_buf, self.act_buf, self.adv_buf,
                self.ret_buf, self.logp_buf, self.temperature_buf[0]]


"""

Proximal Policy Optimization (by clipping), 

with early stopping based on approximate KL

"""


def ppo(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, episodes_per_epoch=None, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, logger_kwargs=dict(), save_freq=10, custom_h=None, eval_episodes=50,
        do_checkpoint_eval=False, env_name=None, eval_temp=1.0, train_starting_temp=1.0,
        env_version=None, env_input=None, target_arcs=None, early_stop_epochs=None,
        save_all_eval=False, meta_learning=False, finetune=False, finetune_model_path=None):
    """

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in placeholder symbols 
            for state, ``x_ph``, and action, ``a_ph``, and returns the main 
            outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       (batch, act_dim)  | Samples actions from policy given 
                                           | states.
            ``logp``     (batch,)          | Gives log probability, according to
                                           | the policy, of taking actions ``a_ph``
                                           | in states ``x_ph``.
            ``logp_pi``  (batch,)          | Gives log probability, according to
                                           | the policy, of the action sampled by
                                           | ``pi``.
            ``v``        (batch,)          | Gives the value estimate for states
                                           | in ``x_ph``. (Critical: make sure 
                                           | to flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
            function you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while 
            still profiting (improving the objective function)? The new policy 
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.)

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take 
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """
    # create logger for training
    logger = EpochLogger(meta_learning_or_finetune=(finetune or meta_learning), **logger_kwargs)
    logger.save_config(locals())

    # create logger for evaluation to keep track of evaluation values at each checkpoint (or save frequency)
    # using eval_progress.txt. It is different from the logger_eval used inside one evaluation epoch.
    logger_eval_progress = EpochLogger(output_fname='progress_eval.txt', **logger_kwargs)

    # create logger for evaluation and save best performance, best structure, and best model in simple_save999999
    logger_eval = EpochLogger(**dict(exp_name=logger_kwargs['exp_name'],
                                     output_dir=os.path.join(logger.output_dir, "simple_save999999")))

    # create logger for tensorboard
    tb_logdir = "{}/tb_logs/".format(logger.output_dir)
    tb_logger = Logger(log_dir=tb_logdir)

    seed += 10000 * proc_id()
    tf.set_random_seed(seed)
    np.random.seed(seed)
    logger.log('set tf and np random seed = {}'.format(seed))


    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    if custom_h is not None:
        hidden_layers_str_list = custom_h.split('-')
        hidden_layers_int_list = [int(h) for h in hidden_layers_str_list]
        ac_kwargs['hidden_sizes'] = hidden_layers_int_list

    # create a tf session with GPU memory usage option to be allow_growth so that one program will not use up the
    # whole GPU memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    # log tf graph
    tf.summary.FileWriter(tb_logdir, sess.graph)

    if not finetune:
        # Inputs to computation graph
        x_ph, a_ph = core.placeholders_from_spaces(env.observation_space, env.action_space)
        adv_ph, ret_ph, logp_old_ph = core.placeholders(None, None, None)

        temperature_ph = tf.placeholder(tf.float32, shape=(), name="init")

        # Main outputs from computation graph
        pi, logp, logp_pi, v = actor_critic(x_ph, a_ph, temperature_ph, **ac_kwargs)

        # PPO objectives
        ratio = tf.exp(logp - logp_old_ph)  # pi(a|s) / pi_old(a|s)
        min_adv = tf.where(adv_ph > 0, (1 + clip_ratio) * adv_ph, (1 - clip_ratio) * adv_ph)
        pi_loss = -tf.reduce_mean(tf.minimum(ratio * adv_ph, min_adv))
        v_loss = tf.reduce_mean((ret_ph - v) ** 2)

        # Info (useful to watch during learning)
        approx_kl = tf.reduce_mean(logp_old_ph - logp)  # a sample estimate for KL-divergence, easy to compute
        approx_ent = tf.reduce_mean(-logp)  # a sample estimate for entropy, also easy to compute
        clipped = tf.logical_or(ratio > (1 + clip_ratio), ratio < (1 - clip_ratio))
        clipfrac = tf.reduce_mean(tf.cast(clipped, tf.float32))

        # Optimizers
        train_pi = tf.compat.v1.train.AdamOptimizer(learning_rate=pi_lr).minimize(pi_loss, name='train_pi')
        train_v = tf.compat.v1.train.AdamOptimizer(learning_rate=vf_lr).minimize(v_loss, name='train_v')

        sess.run(tf.global_variables_initializer())

    else:  # do finetuning -- load model from meta_model_path
        assert finetune_model_path is not None, "Please specify the path to the meta learnt model using --finetune_model_path"
        if 'simple_save' in finetune_model_path:
            model = restore_tf_graph(sess, fpath=finetune_model_path,
                                     meta_learning_or_finetune=finetune)
        else:
            model = restore_tf_graph(sess, fpath=finetune_model_path + '/simple_save999999',
                                     meta_learning_or_finetune=finetune)

        # get placeholders
        x_ph, a_ph, adv_ph = model['x'], model['a'], model['adv']
        ret_ph, logp_old_ph, temperature_ph = model['ret'], model['logp_old'], model['temperature']

        # get model output
        pi, logp, logp_pi, v = model['pi'], model['logp'], model['logp_pi'], model['v']
        pi_loss, v_loss = model['pi_loss'], model['v_loss']
        approx_kl, approx_ent, clipfrac = model['approx_kl'], model['approx_ent'], model['clipfrac']

        # get Optimizers
        train_pi = model['train_pi']
        train_v = model['train_v']

    # Need all placeholders in *this* order later (to zip with data from buffer)
    all_phs = [x_ph, a_ph, adv_ph, ret_ph, logp_old_ph, temperature_ph]

    # Every step, get: action, value, and logprob
    get_action_ops = [pi, v, logp_pi]

    # Experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in ['pi', 'v'])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

    # # log tf graph
    # tf.summary.FileWriter(tb_logdir, sess.graph)

    # Sync params across processes
    sess.run(sync_all_params())

    # Setup model saving
    logger.setup_tf_saver(sess,
                          inputs={'x': x_ph,
                                  'a': a_ph,
                                  'adv': adv_ph,
                                  'ret': ret_ph,
                                  'logp_old': logp_old_ph,
                                  'temperature': temperature_ph},
                          outputs={'pi': pi,
                                   'v': v,
                                   'logp': logp,
                                   'logp_pi': logp_pi,
                                   'pi_loss': pi_loss,
                                   'v_loss': v_loss,
                                   'approx_kl': approx_kl,
                                   'approx_ent': approx_ent,
                                   'clipfrac': clipfrac
                                   })

    def update():
        inputs = {k: v for k, v in zip(all_phs, buf.get())}
        pi_l_old, v_l_old, ent = sess.run([pi_loss, v_loss, approx_ent], feed_dict=inputs)

        # Training
        for i in range(train_pi_iters):
            _, kl = sess.run([train_pi, approx_kl], feed_dict=inputs)
            kl = mpi_avg(kl)
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.' % i)
                break
        logger.store(StopIter=i)
        for _ in range(train_v_iters):
            sess.run(train_v, feed_dict=inputs)

        # Log changes from update
        pi_l_new, v_l_new, kl, cf = sess.run([pi_loss, v_loss, approx_kl, clipfrac], feed_dict=inputs)
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(pi_l_new - pi_l_old),
                     DeltaLossV=(v_l_new - v_l_old))

    start_time = time.time()
    o, r, d, ep_ret, ep_len, ep_dummy_action_count, ep_len_normalized = env.reset(), 0, False, 0, 0, 0, []

    # initialize variables for keeping track of BEST eval performance
    best_eval_AverageEpRet = -0.05  # a negative value so that best model is saved at least once.
    best_eval_StdEpRet = 1.0e30

    # below are used for early-stop. We early stop if
    # 1) a best model has been saved, and,
    # 2) 50 epochs have passed without a new save
    saved = False
    early_stop_count_started = False
    episode_count_after_saved = 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        current_temp = _get_current_temperature(epoch, epochs, train_starting_temp)
        for t in range(local_steps_per_epoch):
            a, v_t, logp_t = sess.run(get_action_ops, feed_dict={x_ph: o.reshape(1, -1),
                                                                 temperature_ph: current_temp})

            # save and log
            buf.store(o, a, r, v_t, logp_t, current_temp)
            logger.store(VVals=v_t)

            o, r, d, _ = env.step(a[0])
            ep_ret += r
            ep_len += 1

            if env_version >= 4:
                ep_len_normalized.append(ep_len / env.allowed_steps)
                if env.action_is_dummy:  # a is dummy action
                    ep_dummy_action_count += 1


            terminal = d or (ep_len == max_ep_len)

            if terminal or (t == local_steps_per_epoch - 1):
                if not terminal:
                    print('Warning: trajectory cut off by epoch at %d steps.' % ep_len)
                # if trajectory didn't reach terminal state, bootstrap value target
                last_val = r if d else sess.run(v, feed_dict={x_ph: o.reshape(1, -1),
                                                              temperature_ph: current_temp})
                buf.finish_path(last_val)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                    if env_version >= 4:
                        logger.store(EpDummyCount=ep_dummy_action_count)
                        logger.store(EpTotalArcs=env.adjacency_matrix.sum())

                        assert len(ep_len_normalized) > 0
                        ep_len_normalized = np.asarray(ep_len_normalized, dtype=np.float32).mean()
                        logger.store(EpDummyStepsNormalized=ep_len_normalized)

                o, r, d, ep_ret, ep_len, ep_dummy_action_count, ep_len_normalized = env.reset(), 0, False, 0, 0, 0, []

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs - 1):

            if meta_learning:
                # Save a new model every save_freq and at the last epoch. Do not overwrite the previous save.
                logger.save_state({'env_name': env_name}, epoch)
            else:
                # Save a new model every save_freq and at the last epoch. Only keep one copy - the current model
                logger.save_state({'env_name': env_name})

            # Evaluate and save best model
            if do_checkpoint_eval and epoch > 0:
                # below is a hack. best model related stuff is saved at itr 999999, therefore, simple_save999999.
                # Doing this way, I can use test_policy and plot directly to test the best models.
                # saved best models includes:
                # 1) a copy of the env_name
                # 2) the best rl model with parameters
                # 3) a pickle file "best_eval_performance_n_structure" storing best_performance, best_structure and epoch
                # note that 1) and 2) are spinningup defaults, and 3) is a custom save
                best_eval_AverageEpRet, best_eval_StdEpRet, saved = eval_and_save_best_model(
                    best_eval_AverageEpRet,
                    best_eval_StdEpRet,
                    # a new best logger is created and passed in so that the new logger can leverage the directory
                    # structure without messing up the logger in the training loop
                    # eval_logger=EpochLogger(**dict(
                    #     exp_name=logger_kwargs['exp_name'],
                    #     output_dir=os.path.join(logger.output_dir, "simple_save999999"))),
                    eval_logger=logger_eval,
                    train_logger=logger,
                    eval_progress_logger=logger_eval_progress,
                    tb_logger=tb_logger,
                    epoch=epoch,
                    # the env_name is passed in so that to create an env when and where it is needed. This is to
                    # logx.save_state() error where an env pointer cannot be pickled
                    env_name="F{}x{}T{}_SP{}_v{}".format(env.n_plant, env.n_product, env.target_arcs, env.n_sample,
                                                         env_version) if env_version >= 3 else env_name,
                    env_version=env_version,
                    env_input=env_input,
                    render=False,  # change this to True if you want to visualize how arcs are added during evaluation
                    target_arcs=env.input_target_arcs,
                    get_action=lambda x: sess.run(pi, feed_dict={x_ph: x[None, :],
                                                                 temperature_ph: eval_temp})[0],
                    # number of samples to draw when simulate demand
                    n_sample=5000,
                    num_episodes=eval_episodes,
                    seed=seed,
                    save_all_eval=save_all_eval
                )

        # Perform PPO update!
        update()

        # # # Log into tensorboard
        log_key_to_tb(tb_logger, logger, epoch, key="EpRet", with_min_and_max=True)
        log_key_to_tb(tb_logger, logger, epoch, key="EpLen", with_min_and_max=False)
        log_key_to_tb(tb_logger, logger, epoch, key="VVals", with_min_and_max=True)
        log_key_to_tb(tb_logger, logger, epoch, key="LossPi", with_min_and_max=False)
        log_key_to_tb(tb_logger, logger, epoch, key="LossV", with_min_and_max=False)
        log_key_to_tb(tb_logger, logger, epoch, key="DeltaLossPi", with_min_and_max=False)
        log_key_to_tb(tb_logger, logger, epoch, key="DeltaLossV", with_min_and_max=False)
        log_key_to_tb(tb_logger, logger, epoch, key="Entropy", with_min_and_max=False)
        log_key_to_tb(tb_logger, logger, epoch, key="KL", with_min_and_max=False)
        log_key_to_tb(tb_logger, logger, epoch, key="ClipFrac", with_min_and_max=False)
        log_key_to_tb(tb_logger, logger, epoch, key="StopIter", with_min_and_max=False)
        tb_logger.log_scalar(tag="TotalEnvInteracts", value=(epoch + 1) * steps_per_epoch, step=epoch)
        tb_logger.log_scalar(tag="Time", value=time.time() - start_time, step=epoch)
        tb_logger.log_scalar(tag="epoch_temp", value=current_temp, step=epoch)
        if env_version >= 4:
            log_key_to_tb(tb_logger, logger, epoch, key="EpDummyCount", with_min_and_max=False)
            log_key_to_tb(tb_logger, logger, epoch, key="EpTotalArcs", with_min_and_max=False)

            if 'EpDummyStepsNormalized' in logger.epoch_dict.keys():
                if len(logger.epoch_dict['EpDummyStepsNormalized']) > 0:
                    log_key_to_tb(tb_logger, logger, epoch, key="EpDummyStepsNormalized", with_min_and_max=False)

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch + 1) * steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time() - start_time)
        logger.log_tabular('EpochTemp', current_temp)
        if env_version >= 4:
            logger.log_tabular('EpDummyCount', with_min_and_max=True)
            if 'EpDummyStepsNormalized' in logger.epoch_dict.keys():
                if len(logger.epoch_dict['EpDummyStepsNormalized']) > 0:
                    logger.log_tabular('EpDummyStepsNormalized', average_only=True)
            logger.log_tabular('EpTotalArcs', average_only=True)

        logger.dump_tabular()

        if early_stop_epochs > 0:
            # check for early stop
            if saved:
                # start to count the episodes elapsed after a "saved" event
                early_stop_count_started = True

                # reset the count to 0
                episode_count_after_saved = 0

            else:
                # check whether we should count this episode, i.e., whether early_stop_count_started == True
                if early_stop_count_started:
                    episode_count_after_saved += 1
                    if episode_count_after_saved > early_stop_epochs:
                        logger.log('Early Stopped at epoch {}.'.format(epoch), color='cyan')
                        break


def _get_current_temperature(epoch, epochs, train_starting_temp):
    current_temp = train_starting_temp

    if train_starting_temp > 1.0:
        temp_gap = train_starting_temp - 1.0
        exploring_epochs = np.floor(epochs / 3.0 * 2)
        temp_delta = temp_gap / exploring_epochs
        if epoch < exploring_epochs:
            current_temp = train_starting_temp - temp_delta * epoch
            print('epoch: {} | current_temp: {}'.format(epoch, current_temp))
        else:
            current_temp = 1.0

    return current_temp


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)  # number of layers
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    ppo(lambda: gym.make(args.env), actor_critic=core.mlp_actor_critic,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l), gamma=args.gamma,
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs)
