#!/bin/bash

# Censorship Bias Dataset with different censorship rates, same range, same representation, different means

python3 run_survival.py --model coxph --dataset synthetic --experiment_name censor_bias7 --seed 1 --num_trials 5 \
    --N 1000 --G 2 --D 1 \
    --repr 0.5 0.5 --censorship_repr 0.1 0.9 \
    --mean 2 0 --std 1 1 \
    --scale 1 1 --shape 1 1 \
    --censorship_mean 0 0 --censorship_temp 1 1 --censorship_times 0 1 0 1