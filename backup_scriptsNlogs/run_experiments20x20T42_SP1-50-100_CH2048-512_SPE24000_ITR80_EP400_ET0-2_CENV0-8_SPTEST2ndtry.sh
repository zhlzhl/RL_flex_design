#!/bin/bash

sample=(1 50 100)
for (( i=0; i<${#sample[@]}; i++ )); do
python -m spinup.run_simple \
    --algo ppo  \
    --env Flexibility20x20T42_SP"${sample[i]}"-v0 \
    --exp_name F20x20T42_SP"${sample[i]}"_PPO_EP400_ET0-2_CENV0-8_SPTEST2ndtry  \
    --cpu 2 \
    --epochs 400  \
    --steps_per_epoch 24000  \
    --save_freq 10  \
    --custom_h 2048-512 \
    --act tf.nn.relu  \
    --eval_temp 0.2 \
    --do_checkpoint_eval; done
