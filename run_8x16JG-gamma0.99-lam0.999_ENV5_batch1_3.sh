#!/bin/bash

python -m spinup.run_flexibility   \
                                   --algo ppo    \
                                   --env_name F8x16JG-gamma0.99-lam0.999-v5   \
                                   --exp_name F8x16JG-gamma0.99-lam0.999_CH1024-128_ENV5    \
                                   --cpu 2   \
                                   --epochs 800    \
                                   --custom_h 1024-128   \
                                   --env_version 5   \
                                   --env_input input_JGm8n16_cv0.8.pkl   \
                                   --target_arcs  34 37    \
                                   --seed 100 110 120   \
                                   --save_freq 10    \
                                   --steps_per_epoch 28000   \
                                   --do_checkpoint_eval  --gamma 0.99    --lam 0.999;