#!/bin/bash

# Equal Synthetic Dataset

python run_survival.py --model coxph --dataset synthetic --experiment_name equal_1 --seed 1 \
    --N 10000 --G 2 --D 1 \
    --repr 0.5 0.5 --censorship_repr 0.5 0.5 \
    --mean 0 0 --std 1 1 \
    --scale 1 1 --shape 1 1 \
    --censorship_mean 0 0 --censorship_temp 1 1 --censorship_times 0 1 0 1

python run_survival.py --model coxph --dataset synthetic --experiment_name equal_2 --seed 2 \
    --N 10000 --G 2 --D 1 \
    --repr 0.5 0.5 --censorship_repr 0.5 0.5 \
    --mean 0 0 --std 1 1 \
    --scale 1 1 --shape 1 1 \
    --censorship_mean 0 0 --censorship_temp 1 1 --censorship_times 0 1 0 1

python run_survival.py --model coxph --dataset synthetic --experiment_name equal_3 --seed 3 \
    --N 10000 --G 2 --D 1 \
    --repr 0.5 0.5 --censorship_repr 0.5 0.5 \
    --mean 0 0 --std 1 1 \
    --scale 1 1 --shape 1 1 \
    --censorship_mean 0 0 --censorship_temp 1 1 --censorship_times 0 1 0 1

python run_survival.py --model coxph --dataset synthetic --experiment_name equal_4 --seed 4 \
    --N 10000 --G 2 --D 1 \
    --repr 0.5 0.5 --censorship_repr 0.5 0.5 \
    --mean 0 0 --std 1 1 \
    --scale 1 1 --shape 1 1 \
    --censorship_mean 0 0 --censorship_temp 1 1 --censorship_times 0 1 0 1

python run_survival.py --model coxph --dataset synthetic --experiment_name equal_5 --seed 5 \
    --N 10000 --G 2 --D 1 \
    --repr 0.5 0.5 --censorship_repr 0.5 0.5 \
    --mean 0 0 --std 1 1 \
    --scale 1 1 --shape 1 1 \
    --censorship_mean 0 0 --censorship_temp 1 1 --censorship_times 0 1 0 1