#!/bin/bash

#tarcs=(10 11 12 13 14 15 16 17 18 19 20)
#for (( i=0; i<${#tarcs[@]}; i++ )); do
python -m spinup.run_flexibility \
    --algo ppo  \
    --env_name Flexibility4x64_SP50-v3 \
    --exp_name Flexibility4x64_SP50_CH1536-256_SPE12000_ENV3  \
    --cpu 8 \
    --epochs 800  \
    --steps_per_epoch 12000  \
    --save_freq 10  \
    --custom_h 1536-256 \
    --env_version 3 \
    --env_input input_ran4x64_cv0.8.pkl \
    --do_checkpoint_eval;
#done
