#!/bin/bash

python -m spinup.run_flexibility   \
                                   --algo ppo    \
                                   --env_name F10x26-VR-SP50-v5   \
                                   --exp_name F10x26-VR-SP50_CH1640-332_ENV5_tar41    \
                                   --cpu 6   \
                                   --epochs 800    \
                                   --custom_h 1024-128   \
                                   --env_version 5   \
                                   --env_input input_ran10x26_cv0.8.pkl   \
                                   --target_arcs  41    \
                                   --seed 400 410 420 430   \
                                   --save_freq 10    \
                                   --steps_per_epoch 49200   \
                                   --do_checkpoint_eval  \
                                   --env_subtract_full_flex  \
                                   --custom_h 1640-332   \
                                   --early_stop 60  \
                                   --gamma 0.99   \
                                   --lam 0.999;