#!/bin/bash

python -m spinup.run_flexibility  \
                             --algo ppo  \
                             --env_name F10x10a-SP50-META-v5  \
                             --exp_name F10x10a-SP50-META_CH1024-128_ENV5_tar0  \
                             --cpu 8 \
                             --epochs 800  \
                             --custom_h 1024-128  \
                             --env_version 5  \
                             --env_input input_ran10x10a_cv0.8.pkl  \
                             --target_arcs  13  \
                             --seed 70  \
                             --save_freq 10   \
                             --steps_per_epoch 24000  \
                             --do_checkpoint_eval \
                             --early_stop 30  \
                             --gamma 0.99  \
                             --meta_learning  \
                             --lam 0.999;