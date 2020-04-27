#!/bin/bash

#tarcs=(10 11 12 13 14 15 16 17 18 19 20)
#for (( i=0; i<${#tarcs[@]}; i++ )); do
python -m spinup.run_flexibility \
    --algo ppo  \
    --env_name Flexibility10x10-v3 \
    --exp_name Flexibility10x10_CH1024-128_ENV3  \
    --cpu 8 \
    --epochs 400  \
    --steps_per_epoch 12000  \
    --save_freq 10  \
    --custom_h 1024-128 \
    --act tf.nn.relu  \
    --eval_temp 1.0 \
    --env_version 3 \
    --env_n_sample 50 \
    --env_input input_ran10x10a_cv0.8.pkl \
    --target_arcs 28 31\
    --do_checkpoint_eval;
#done
