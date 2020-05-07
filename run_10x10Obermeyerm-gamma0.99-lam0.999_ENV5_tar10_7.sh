#!/bin/bash

python -m spinup.run_flexibility  \
                             --algo ppo  \
                             --env_name F10x10Obermeyerm-gamma0.99-lam0.999-v5  \
                             --exp_name F10x10Obermeyerm-gamma0.99-lam0.999_CH1024-128_ENV5_tar10  \
                             --cpu 2 \
                             --epochs 800  \
                             --custom_h 1024-128  \
                             --env_version 5  \
                             --env_input input_Obermeyerm10n10_cvNone.pkl  \
                             --target_arcs  10  \
                             --seed 270  \
                             --save_freq 10   \
                             --steps_per_epoch 8000  \
                             --do_checkpoint_eval  --gamma 0.99    --lam 0.999;