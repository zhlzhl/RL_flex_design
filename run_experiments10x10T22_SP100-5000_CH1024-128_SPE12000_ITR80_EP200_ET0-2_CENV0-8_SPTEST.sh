#!/bin/bash

sample=(100 5000)
# tarcs=(27 28 29 30 31 32 33 34 35 36)
for (( i=0; i<${#sample[@]}; i++ )); do
python -m spinup.run_simple \
    --algo ppo  \
    --env Flexibility10x10T22_SP"${sample[i]}"-v0 \
    --exp_name F10x10T22_SP"${sample[i]}"_PPO_CH1024-128_SPE12000_ITR80_EP200_ET0-2_CENV0-8_SPTEST  \
    --cpu 1 \
    --epochs 200  \
    --steps_per_epoch 12000  \
    --save_freq 10  \
    --custom_h 1024-128 \
    --act tf.nn.relu  \
    --eval_temp 0.2 \
    --do_checkpoint_eval; done
