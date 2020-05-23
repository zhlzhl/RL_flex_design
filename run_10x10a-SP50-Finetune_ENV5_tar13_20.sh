#!/bin/bash

python -m spinup.run_flexibility  \
                             --algo ppo  \
                             --env_name F10x10a-SP50-Finetune-v5  \
                             --exp_name F10x10a-SP50-Finetune_ENV5_tar13  \
                             --cpu 8 \
                             --epochs 700  \
                             --custom_h 1024-128  \
                             --env_version 5  \
                             --target_arcs 13  \
                             --env_input input_ran10x10a_cv0.8.pkl  \
                             --seed 20  \
                             --save_freq 10   \
                             --steps_per_epoch 16000  \
                             --do_checkpoint_eval \
                             --finetune  \
                             --finetune_model_path /home/user/git/RL_flex_design/data/F10x10a-SP50-META_ENV5_tar13/F10x10a-SP50-META_ENV5_tar13_s0  \
                             --gamma 0.99  \
                             --lam 0.999;