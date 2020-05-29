#!/bin/bash

python -m spinup.run_flexibility   \
                                   --algo ppo    \
                                   --env_name F10x10Obermeyerm-SP50-v5   \
                                   --exp_name F10x10Obermeyerm-SP50_CH1024-128_ENV5_tar19    \
                                   --cpu 8   \
                                   --epochs 300    \
                                   --custom_h 1024-128   \
                                   --env_version 5   \
                                   --env_input input_Obermeyerm10n10_cvNone.pkl   \
                                   --target_arcs  19    \
                                   --seed 0 10 20 30 40 50   \
                                   --save_freq 30    \
                                   --steps_per_epoch 15200   \
                                   --do_checkpoint_eval  \
                                   --early_stop 60  \
                                   --gamma 0.99   \
                                   --lam 0.999;