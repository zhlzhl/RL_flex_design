#!/bin/bash

#tarcs=(16 18 20 22 24)
tarcs=(26 28 30 32 34)
for (( i=0; i<${#tarcs[@]}; i++ )); do
python -m spinup.run_simple \
    --algo ppo  \
    --env Flexibility10x10T"${tarcs[i]}"-v0 \
    --exp_name F10x10T"${tarcs[i]}"_PPO_CH1024-128_SPE6000_ITR80_EP500  \
    --cpu 8 \
    --epochs 500  \
    --steps_per_epoch 6000  \
    --save_freq 10  \
    --custom_h 1024-128 \
    --act tf.nn.relu  \
    --do_checkpoint_eval; done
