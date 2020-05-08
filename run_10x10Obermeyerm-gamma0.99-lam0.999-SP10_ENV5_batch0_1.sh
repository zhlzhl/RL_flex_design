#!/bin/bash

python -m spinup.run_flexibility   \
                                   --algo ppo    \
                                   --env_name F10x10Obermeyerm-gamma0.99-lam0.999-SP10-v5   \
                                   --exp_name F10x10Obermeyerm-gamma0.99-lam0.999-SP10_CH1024-128_ENV5    \
                                   --cpu 2   \
                                   --epochs 800    \
                                   --custom_h 1024-128   \
                                   --env_version 5   \
                                   --env_input input_Obermeyerm10n10_cvNone.pkl   \
                                   --target_arcs  16 19    \
                                   --seed 0 10 20 30 40 50   \
                                   --save_freq 10    \
                                   --steps_per_epoch 13600   \
                                   --do_checkpoint_eval  \
                                   --env_n_sample 10  \
                                   --gamma 0.99   \
                                   --lam 0.999;