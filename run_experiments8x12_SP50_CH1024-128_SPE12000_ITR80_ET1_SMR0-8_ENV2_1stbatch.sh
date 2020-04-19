#!/bin/bash

tarcs=(21 22 23 24 25 26)
for (( i=0; i<${#tarcs[@]}; i++ )); do
python -m spinup.run_simple \
    --algo ppo  \
    --env_name Flexibility8x12T"${tarcs[i]}"_SP50-v0 \
    --exp_name F8x12T"${tarcs[i]}"_SP50_PPO_CH1024-128_SPE12000_ITR80_ES400_ET1_ENV2  \
    --cpu 8 \
    --epochs 400  \
    --steps_per_epoch 12000  \
    --save_freq 10  \
    --custom_h 1024-128 \
    --act tf.nn.relu  \
    --eval_temp 1.0 \
    --env_version 2 \
    --num_runs 1 \
    --do_checkpoint_eval; done
