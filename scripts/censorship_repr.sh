#!/bin/bash

# Extreme Censorship Bias Dataset


# python run_survival.py --model coxph --dataset synthetic --experiment_name censorship_repr --seed 1 --num_trials 5 \
#     --N 1000 --G 2 --D 1 \
#     --repr 0.5 0.5 --censorship_repr 0.1 0.1 \
#     --mean 0 1 --std 1 1 \
#     --scale 3 1 --shape 3 1 \
#     --censorship_mean 0 0 --censorship_temp 1 1 --censorship_times 0 1 0 1

python3 run_survival.py --model coxph --dataset synthetic --experiment_name censorship_repr --seed 1 --num_trials 5 \
    --N 1000 --G 2 --D 1 \
    --repr 0.5 0.5 --censorship_repr 0.01 0.99 \
    --mean 0 1 --std 1 1 \
    --scale 3 1 --shape 1 1 \
    --censorship_mean 0 0 --censorship_temp 1 1 --censorship_times 0.9 1 0.2 0.3