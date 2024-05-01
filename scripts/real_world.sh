#!/bin/bash

# Real World Datasets

python run_survival.py --model "uniform" --dataset "flchain"
python run_survival.py --model "uniform" --dataset "whas500"
python run_survival.py --model "uniform" --dataset "aids"
python run_survival.py --model "uniform" --dataset "gbsg2"
python run_survival.py --model "uniform" --dataset "veterans_lung_cancer"

python run_survival.py --model "coxph" --dataset "flchain"
python run_survival.py --model "coxph" --dataset "whas500"
python run_survival.py --model "coxph" --dataset "aids"
python run_survival.py --model "coxph" --dataset "gbsg2"
python run_survival.py --model "coxph" --dataset "veterans_lung_cancer"

python run_survival.py --model "randomforest" --dataset "flchain"
python run_survival.py --model "randomforest" --dataset "whas500"
python run_survival.py --model "randomforest" --dataset "aids"
python run_survival.py --model "randomforest" --dataset "gbsg2"
python run_survival.py --model "randomforest" --dataset "veterans_lung_cancer"