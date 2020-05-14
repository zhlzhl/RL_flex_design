#!/bin/bash

python -m spinup.run_flexibility   \
                                   --algo ppo    \
                                   --env_name F8x16JG-gamma0.99-lam0.999-SP50-v5   \
                                   --exp_name F8x16JG-gamma0.99-lam0.999-SP50_CH1024-128_ENV5    \
                                   --cpu 8   \
                                   --epochs 200    \
                                   --custom_h 1024-128   \
                                   --env_version 5   \
                                   --env_input input_JGm8n16_cv0.8.pkl   \
                                   --target_arcs  28 31    \
                                   --seed 0 10 20 30 40 50   \
                                   --save_freq 20    \
                                   --steps_per_epoch 23200   \
                                   --do_checkpoint_eval  \
                                   --early_stop -1  \
                                   --save_all_eval  \
                                   --gamma 0.99   \
                                   --lam 0.999;