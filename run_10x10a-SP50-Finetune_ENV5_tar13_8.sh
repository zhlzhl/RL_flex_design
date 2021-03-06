#!/bin/bash

python -m spinup.run_flexibility  \
                             --algo ppo  \
                             --env_name F10x10a-SP50-Finetune-v5  \
                             --exp_name F10x10a-SP50-Finetune_CH1024-128_ENV5_tar13  \
                             --cpu 8 \
                             --epochs 800  \
                             --custom_h 1024-128  \
                             --env_version 5  \
                             --env_input input_ran10x10a_cv0.8.pkl  \
                             --target_arcs  13  \
                             --seed 80  \
                             --save_freq 10   \
                             --steps_per_epoch 10400  \
                             --do_checkpoint_eval \
                             --early_stop 60  \
                             --gamma 0.99  \
                             --finetune  \
                             --finetune_model_path /home/user/git/RL_flex_design/data/F10x10a-SP50-META_CH1024-128_ENV5_tar0/F10x10a-SP50-META_CH1024-128_ENV5_tar0_s80  \
                             --lam 0.999;