#!/bin/bash

#tarcs=(16 18 20 22 24)
tarcs=(25 26 27 28 29 30 31 32 33 34 35 36)
for (( i=0; i<${#tarcs[@]}; i++ )); do
python -m spinup.run_simple \
    --algo ppo  \
    --env_name Flexibility10x10T"${tarcs[i]}"_SP50-v0 \
    --exp_name F10x10T"${tarcs[i]}"_SP50_PPO_CH1024-128_SPE12000_EP1500_ET0-2_SMR0-8_ENV1_2ndrun  \
    --cpu 8 \
    --epochs 1500  \
    --steps_per_epoch 12000  \
    --save_freq 10  \
    --custom_h 1024-128 \
    --act tf.nn.relu  \
    --eval_temp 0.2 \
    --num_runs 4 \
    --do_checkpoint_eval; done
