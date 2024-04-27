import os
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sksurv.linear_model import CoxPHSurvivalAnalysis
from dataset_utils import load_dataset
from dataset_utils import preprocess_dataset
import metrics

'''
Script run_survival.py

Loads a model, dataset, and runs evaluation metrics.

Example Usage:
python run_survival.py --model "coxph" --dataset "whas500"
'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', action='store', type=str, required=True, help='Name of model')
    parser.add_argument('--dataset', action='store', type=str, required=True, help='Name of survival dataset')
    parser.add_argument('--seed', action='store', type=int, required=False, default=226, help='Seed')
    parser.add_argument('--experiment_name', action='store', type=str, required=False, help='Name of Experiment')
    args = parser.parse_args()
    if args.experiment_name is None:
        args.experiment_name = f"{args.model}_{args.dataset}_{args.seed}"
    os.makedirs(f"results/{args.experiment_name}/",exist_ok=True)
    with open(f"results/{args.experiment_name}/{args.experiment_name}_args.txt","w") as f:
        f.write(str(args))

    # Load survival dataset
    dataset = load_dataset(args.dataset)
    X_train, X_test, Y_train, Y_test, G_train, G_test = preprocess_dataset(dataset)

    # Train an estimator
    estimator = CoxPHSurvivalAnalysis(alpha=0.1).fit(X_train, Y_train.to_records(index=False))

    # Evaluate the model
    surv_funcs = estimator.predict_survival_function(X_test)
    for fn in surv_funcs:
        plt.step(fn.x, fn(fn.x), where="post")
    plt.ylim(0, 1)
    plt.savefig(f"results/{args.experiment_name}/{args.experiment_name}_survival_function.png")

    train_risk_scores = estimator.predict(X_train)
    test_risk_scores = estimator.predict(X_test)

    with open(f"results/{args.experiment_name}/{args.experiment_name}_metrics.txt","w") as f:
        concordance_index = metrics.concordance_index_censored(Y_test["event_indicator"],Y_test["event_time"],test_risk_scores)
        # concordance_index = estimator.score(np.array(X_test), Y_test.to_records(index=False))
        keya_individual = metrics.keya_individual_fairness(np.array(X_test),test_risk_scores)
        
        f.write(f"Concordance Index: {concordance_index}\n")
        f.write(f"Keya Individual: {keya_individual}\n")

if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    print(f"Experiment took {end_time-start_time} seconds.")