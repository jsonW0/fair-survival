#!/bin/bash
# python run_survival.py --model "coxph" --dataset "whas500"
# python run_survival.py --model "coxph" --dataset "flchain"



python run_survival.py --model coxph --dataset synthetic \
    --N 1000 --G 2 --D 1 \
    --repr 0.5 0.5 --censorship_repr 0.5 0.5 \
    --mean 0 0 --std 1 1 \
    --scale 1 1 --shape 1 1 \
    --censorship_mean 0 0 --censorship_temp 1 1 --censorship_times 0 1 0 1