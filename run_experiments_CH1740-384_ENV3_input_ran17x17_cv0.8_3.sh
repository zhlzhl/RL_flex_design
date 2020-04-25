#!/bin/bash

#tarcs=(10 11 12 13 14 15 16 17 18 19 20)
#for (( i=0; i<${#tarcs[@]}; i++ )); do
python -m spinup.run_flexibility \
    --algo ppo  \
    --env_name Flexibility17x17_-v3 \
    --exp_name Flexibility17x17_CH1740-384_ENV3  \
    --cpu 2 \
    --epochs 2000  \
    --steps_per_epoch 12000  \
    --save_freq 10  \
    --custom_h 1740-384 \
    --env_version 3 \
    --env_n_sample 50 \
    --env_input input_ran17x17_cv0.8.pkl \
    --target_arcs 47 50 53 56 \
    --do_checkpoint_eval;
#done
