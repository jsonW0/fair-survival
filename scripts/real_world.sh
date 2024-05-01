#!/bin/bash

# Real World Datasets

python run_survival.py --model "coxph" --dataset "flchain"
python run_survival.py --model "coxph" --dataset "whas500"
python run_survival.py --model "coxph" --dataset "aids"
# python run_survival.py --model "coxph" --dataset "breast_cancer"
python run_survival.py --model "coxph" --dataset "gbsg2"
python run_survival.py --model "coxph" --dataset "veterans_lung_cancer"