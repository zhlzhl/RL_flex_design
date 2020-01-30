#!/bin/bash

#tarcs=(44 48 52 56 60)
tarcs=(44)
for (( i=0; i<${#tarcs[@]}; i++ )); do
python -m spinup.run_simple \
    --algo ppo  \
    --env Flexibility20x20T"${tarcs[i]}"_SP1-v0 \
    --exp_name F20x20T"${tarcs[i]}"_SP1_PPO_CH2048-512_SPE96000_ITR80_EP1000_ET0-5_TST1-0  \
    --cpu 8 \
    --epochs 1000  \
    --steps_per_epoch 96000  \
    --save_freq 40  \
    --custom_h 2048-512 \
    --act tf.nn.relu  \
    --train_pi_iters 80  \
    --train_v_iters 80 \
    --eval_temp 0.5 \
    --train_starting_temp 1.0 \
    --n_sample 50 \
    --do_checkpoint_eval; done
