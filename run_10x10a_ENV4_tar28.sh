#!/bin/bash

#tarcs=(10 11 12 13 14 15 16 17 18 19 20) # #tarcs=(10, 13, 16, 19, 22, 25, 28, 31, 34, 37)
#for (( i=0; i<${#tarcs[@]}; i++ )); do
env_version=4
tar=28
python -m spinup.run_flexibility \
    --algo ppo  \
    --env_name F10x10-v${env_version} \
    --exp_name F10x10_CH1024-128_ENV${env_version}_tar${tar}  \
    --cpu 2 \
    --epochs 400  \
    --custom_h 1024-128 \
    --env_version ${env_version} \
    --env_input input_ran10x10a_cv0.8.pkl \
    --target_arcs  ${tar} \
    --steps_per_epoch 24000  \
    --seed 24040 24050 \
    --save_freq 8  \
    --do_checkpoint_eval;
#done
