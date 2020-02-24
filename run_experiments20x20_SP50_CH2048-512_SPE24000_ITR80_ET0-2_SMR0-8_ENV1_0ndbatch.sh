#!/bin/bash

tarcs=(20 22 24 26 28 30 32 34 36)
for (( i=0; i<${#tarcs[@]}; i++ )); do
python -m spinup.run_simple \
    --algo ppo  \
    --env_name Flexibility20x20T"${tarcs[i]}"_SP50-v0 \
    --exp_name F20x20T"${tarcs[i]}"_SP50_PPO_EP800_ET0-2_SMR0-8_ENV1  \
    --cpu 8 \
    --epochs 800  \
    --steps_per_epoch 24000  \
    --save_freq 10  \
    --custom_h 2048-512 \
    --act tf.nn.relu  \
    --eval_temp 0.2 \
    --num_runs 4 \
    --do_checkpoint_eval; done
