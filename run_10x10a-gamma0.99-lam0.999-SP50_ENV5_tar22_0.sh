#!/bin/bash

python -m spinup.run_flexibility  \
                             --algo ppo  \
                             --env_name F10x10a-gamma0.99-lam0.999-SP50-v5  \
                             --exp_name F10x10a-gamma0.99-lam0.999-SP50_CH1024-128_ENV5_tar22  \
                             --cpu 2 \
                             --epochs 360  \
                             --custom_h 1024-128  \
                             --env_version 5  \
                             --env_input input_ran10x10a_cv0.8.pkl  \
                             --target_arcs  22  \
                             --seed 700000  \
                             --save_freq 10   \
                             --steps_per_epoch 17600  \
                             --do_checkpoint_eval \
                             --early_stop -1  \
                             --save_freq 40  \
                             --save_all_eval  \
                             --gamma 0.99  \
                             --lam 0.999;