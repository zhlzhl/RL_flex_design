#!/bin/bash

python -m spinup.run_flexibility  \
                             --algo ppo  \
                             --env_name F10x26-VR-SP20-META-v5  \
                             --exp_name F10x26-VR-SP20-META_CH1640-332_ENV5_tar0  \
                             --cpu 8 \
                             --epochs 100  \
                             --custom_h 1640-332  \
                             --env_version 5  \
                             --env_input input_ran10x26_cv0.8.pkl  \
                             --target_arcs  44  \
                             --seed 100  \
                             --save_freq 20   \
                             --steps_per_epoch 48000  \
                             --do_checkpoint_eval \
                             --env_subtract_full_flex  \
                             --env_n_sample 20  \
                             --early_stop -1  \
                             --gamma 0.99  \
                             --custom_h 1640-332  \
                             --meta_learning  \
                             --lam 0.999;