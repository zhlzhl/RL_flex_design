#!/bin/bash

tarcs=(9 10 11 12 13 14 15 16 17 18 19 20 21)
# tarcs=(27 28 29 30 31 32 33 34 35 36)
for (( i=0; i<${#tarcs[@]}; i++ )); do
python -m spinup.run_simple \
    --algo ppo  \
    --env Flexibility10x10T"${tarcs[i]}"_SP50-v0 \
    --exp_name F10x10T"${tarcs[i]}"_SP50_PPO_CH1024-128_SPE12000_ITR80_EP200_ET0-2_CENV0-8_GM1  \
    --cpu 8 \
    --epochs 200  \
    --steps_per_epoch 12000  \
    --save_freq 10  \
    --custom_h 1024-128 \
    --act tf.nn.relu  \
    --eval_temp 0.2 \
    --gamma 1.0 \
    --do_checkpoint_eval; done
