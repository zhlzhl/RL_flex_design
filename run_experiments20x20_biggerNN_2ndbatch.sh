#!/bin/bash

tarcs=(44 48 52 56 60)
for (( i=0; i<${#tarcs[@]}; i++ )); do
python -m spinup.run_simple \
    --algo ppo  \
    --env Flexibility20x20T"${tarcs[i]}"-v0 \
    --exp_name Flexibility20x20T"${tarcs[i]}"_PPO  \
    --cpu 8 \
    --epochs 240  \
    --steps_per_epoch 6000  \
    --save_freq 10  \
    --custom_h 2048-512 \
    --act tf.nn.relu  \
    --do_checkpoint_eval; done
