#!/bin/bash

tarcs=(42 44 46 48 50 52 54 56 58 60)
for (( i=0; i<${#tarcs[@]}; i++ )); do
python -m spinup.run_simple \
    --algo ppo  \
    --env Flexibility20x20T"${tarcs[i]}"_SP50-v0 \
    --exp_name F20x20T"${tarcs[i]}"_SP50_PPO_EP700_ET0-2_CENV \
    --cpu 8 \
    --epochs 700  \
    --steps_per_epoch 24000  \
    --save_freq 10  \
    --custom_h 2048-512 \
    --act tf.nn.relu  \
    --eval_temp 0.2 \
    --do_checkpoint_eval; done
