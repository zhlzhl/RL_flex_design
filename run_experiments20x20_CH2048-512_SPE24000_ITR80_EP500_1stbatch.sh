#!/bin/bash

tarcs=(20 24 28 32 36 40)
for (( i=0; i<${#tarcs[@]}; i++ )); do
python -m spinup.run_simple \
    --algo ppo  \
    --env Flexibility20x20T"${tarcs[i]}"-v0 \
    --exp_name F20x20T"${tarcs[i]}"_PPO_CH2048-512_SPE24000_ITR80_EP350  \
    --cpu 8 \
    --epochs 500  \
    --steps_per_epoch 24000  \
    --save_freq 10  \
    --custom_h 2048-512 \
    --act tf.nn.relu  \
    --train_pi_iters 80  \
    --train_v_iters 80 \
    --do_checkpoint_eval; done
