#!/bin/bash

python -m spinup.run_flexibility  \
                             --algo ppo  \
                             --env_name F8x16JG-SP50-v5  \
                             --exp_name F8x16JG-SP50_CH1024-128_ENV5_tar22  \
                             --cpu 8 \
                             --epochs 300  \
                             --custom_h 1024-128  \
                             --env_version 5  \
                             --env_input input_JGm8n16_cv0.8.pkl  \
                             --target_arcs  22  \
                             --seed 100  \
                             --save_freq 30   \
                             --steps_per_epoch 17600  \
                             --do_checkpoint_eval \
                             --early_stop 60  \
                             --save_all_eval  \
                             --gamma 0.99  \
                             --lam 0.999;