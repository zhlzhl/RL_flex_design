#!/bin/bash

python -m spinup.run_flexibility   \
                                   --algo ppo    \
                                   --env_name F10x26-gamma0.99-lam0.999-SP50-v5   \
                                   --exp_name F10x26-gamma0.99-lam0.999-SP50_CH1024-128_ENV5    \
                                   --cpu 8   \
                                   --epochs 800    \
                                   --custom_h 1024-128   \
                                   --env_version 5   \
                                   --env_input input_ran10x26_cv0.8.pkl   \
                                   --target_arcs  41 32    \
                                   --seed 0 10 20 30 40 50   \
                                   --save_freq 10    \
                                   --steps_per_epoch 28800   \
                                   --do_checkpoint_eval  \
                                   --gamma 0.99   \
                                   --lam 0.999;