#!/bin/bash
python -m spinup.run_simple \
    --algo ppo  \
    --env Flexibility10x10_SP50-v0 \
    --exp_name F10x10_SP50_EP200_TST3_ET0-2_TL  \
    --cpu 2 \
    --epochs 200  \
    --steps_per_epoch 12000  \
    --save_freq 10  \
    --custom_h 1024-128 \
    --act tf.nn.relu  \
    --do_checkpoint_eval \
    --eval_temp 0.2 \
    --train_starting_temp 3 \
    --target_arcs 17 18 19 20 21 22 23 24 25 26 \
    --train_continuously \
    --transfer_learning_base_model_path /home/user/git/spinningup/data/2020-02-07_F10x10_SP50_EP200_ET0-2_CT_tar16/2020-02-07_10-29-02-F10x10_SP50_EP200_ET0-2_CT_tar16_s0;
