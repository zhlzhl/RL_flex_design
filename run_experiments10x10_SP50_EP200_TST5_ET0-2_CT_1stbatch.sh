#!/bin/bash
python -m spinup.run_simple \
    --algo ppo  \
    --env Flexibility10x10_SP50-v0 \
    --exp_name F10x10_SP50_EP200_ET0-2_CT  \
    --cpu 2 \
    --epochs 200  \
    --steps_per_epoch 12000  \
    --save_freq 10  \
    --custom_h 1024-128 \
    --act tf.nn.relu  \
    --eval_temp 0.2 \
    --train_starting_temp 5 \
    --target_arcs 16 17 18 19 20 21 22 23 24 25 26 \
    --train_continuously \
    --do_checkpoint_eval;
