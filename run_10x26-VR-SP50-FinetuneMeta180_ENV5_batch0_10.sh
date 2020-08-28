#!/bin/bash

python -m spinup.run_flexibility   \
                                   --algo ppo    \
                                   --env_name F10x26-VR-SP50-FinetuneMeta180-v5   \
                                   --exp_name F10x26-VR-SP50-FinetuneMeta180_CH1640-332_ENV5    \
                                   --cpu 8   \
                                   --epochs 800    \
                                   --custom_h 1640-332   \
                                   --env_version 5   \
                                   --env_input input_ran10x26_cv0.8.pkl   \
                                   --target_arcs  38 41 44 47    \
                                   --seed 100 101 102 103 104 105 106 107   \
                                   --save_freq 10    \
                                   --steps_per_epoch 56400   \
                                   --do_checkpoint_eval  \
                                   --env_subtract_full_flex  \
                                   --early_stop 40  \
                                   --finetune  \
                                   --finetune_model_path /home/user/git/RL_flex_design/data/10x26_MAML/F10x26-VR-SP20-META_CH1640-332_ENV5_tar0/F10x26-VR-SP20-META_CH1640-332_ENV5_tar0_s100/simple_save180  \
                                   --gamma 0.99   \
                                   --lam 0.999;