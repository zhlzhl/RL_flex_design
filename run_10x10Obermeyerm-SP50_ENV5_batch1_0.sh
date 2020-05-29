#!/bin/bash

python -m spinup.run_flexibility   \
                                   --algo ppo    \
                                   --env_name F10x10Obermeyerm-SP50-v5   \
                                   --exp_name F10x10Obermeyerm-SP50_CH1024-128_ENV5_tar10    \
                                   --cpu 8   \
                                   --epochs 300    \
                                   --custom_h 1024-128   \
                                   --env_version 5   \
                                   --env_input input_Obermeyerm10n10_cvNone.pkl   \
                                   --target_arcs  10    \
                                   --seed 100 110 120 130 140 150   \
                                   --save_freq 30    \
                                   --steps_per_epoch 8000   \
                                   --do_checkpoint_eval  \
                                   --early_stop 60  \
                                   --gamma 0.99   \
                                   --lam 0.999;