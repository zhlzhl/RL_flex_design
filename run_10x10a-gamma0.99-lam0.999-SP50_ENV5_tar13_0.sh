#!/bin/bash

python -m spinup.run_flexibility  \
                             --algo ppo  \
                             --env_name F10x10a-gamma0.99-lam0.999-SP50-v5  \
                             --exp_name F10x10a-gamma0.99-lam0.999-SP50_CH1024-128_ENV5_tar13  \
                             --cpu 16 \
                             --epochs 50  \
                             --custom_h 1024-128  \
                             --env_version 5  \
                             --env_input input_ran10x10a_cv0.8.pkl  \
                             --target_arcs  13  \
                             --seed 0  \
                             --save_freq 10   \
                             --steps_per_epoch 10400  \
                             --do_checkpoint_eval \
                             --gamma 0.99  \
                             --lam 0.999;