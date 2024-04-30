import os
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sksurv.linear_model import CoxPHSurvivalAnalysis
from dataset_utils import load_dataset, preprocess_dataset, generate_synthetic_dataset
import metrics

'''
Script run_survival.py

Loads a model, dataset, and runs evaluation metrics.

Example Usage:
python run_survival.py --model "coxph" --dataset "whas500"
'''

def main():
    parser = argparse.ArgumentParser()
    # Experiment parameters
    parser.add_argument('--model', action='store', type=str, required=True, help='Name of model')
    parser.add_argument('--dataset', action='store', type=str, required=True, help='Name of survival dataset')
    parser.add_argument('--seed', action='store', type=int, required=False, default=226, help='Seed')
    parser.add_argument('--experiment_name', action='store', type=str, required=False, help='Name of Experiment')
    # Synthetic dataset parameters
    parser.add_argument('--N', action='store', type=int, required=False, default=1000, help='Number of individuals in synthetic dataset')
    parser.add_argument('--G', action='store', type=int, required=False, default=2, help='Number of demographic groups in synthetic dataset')
    parser.add_argument('--D', action='store', type=int, required=False, default=1, help='Number of features dimensions in synthetic dataset')
    parser.add_argument('--repr', action='store', type=float, nargs='+', required=False, default=[0.5,0.5], help='Probability of belonging to each demographic group')
    parser.add_argument('--censorship_repr', action='store', type=float, nargs='+', required=False, default=[0.5,0.5], help='Base probability of censorship for each demographic group')
    parser.add_argument('--mean', action='store', type=float, nargs='+', required=False, default=[0.,0.], help='Mean feature vector for each demographic group')
    parser.add_argument('--std', action='store', type=float, nargs='+', required=False, default=[1.,1.], help='Covariance matrix of feature vector for each demographic group')
    parser.add_argument('--scale', action='store', type=float, nargs='+', required=False, default=[1.,1.], help='Scale parameter for each demographic group survival time')
    parser.add_argument('--shape', action='store', type=float, nargs='+', required=False, default=[1.,1.], help='Shape parameter for each demographic group survival time')
    parser.add_argument('--censorship_mean', action='store', type=float, nargs='+', required=False, default=[0.,0.], help='Mean vector of uncensored data for each demographic group')
    parser.add_argument('--censorship_temp', action='store', type=float, nargs='+', required=False, default=[1.,1.], help='Temperature parameter for sampling based on distance to censorship_mean, for each demographic group')
    parser.add_argument('--censorship_times', action='store', type=float, nargs='+', required=False, default=[0.,1.,0.,1.], help='Lower and upper percentiles for right-censoring the times for each demographic group')

    args = parser.parse_args()
    if args.experiment_name is None:
        args.experiment_name = f"{args.model}_{args.dataset}_{args.seed}"
    os.makedirs(f"results/{args.experiment_name}/",exist_ok=True)
    with open(f"results/{args.experiment_name}/{args.experiment_name}_args.txt","w") as f:
        f.write(str(args))

    # Load survival 
    if args.dataset=="synthetic":
        dataset = generate_synthetic_dataset(N=args.N,G=args.G,D=args.D,repr=np.array(args.repr).reshape(args.G),censorship_repr=np.array(args.censorship_repr).reshape(args.G),mean=np.array(args.mean).reshape(args.G,args.D),std=np.array(args.std).reshape(args.G,args.D,args.D),scale=np.array(args.scale).reshape(args.G),shape=np.array(args.shape).reshape(args.G),censorship_mean=np.array(args.censorship_mean).reshape(args.G,args.D),censorship_temp=np.array(args.censorship_temp).reshape(args.G),censorship_times=np.array(args.censorship_times).reshape(args.G,2),seed=args.seed)
        dataset.to_csv(f"results/{args.experiment_name}/synthetic.csv")
    else:
        dataset = load_dataset(args.dataset)
    X_train, X_test, Y_train, Y_test, G_train, G_test = preprocess_dataset(dataset)

    # Train an estimator
    if args.model == "coxph":
        estimator = CoxPHSurvivalAnalysis(alpha=0.1).fit(X_train, Y_train.to_records(index=False))
    else:
        raise NotImplementedError(f"{args.model} has not been implemented.")

    # Evaluate the model
    surv_funcs = estimator.predict_survival_function(X_test)
    plt.figure()
    for fn in surv_funcs:
        plt.step(fn.x, fn(fn.x), where="post", alpha=0.05, c="tab:blue")
    plt.ylim(0, 1)
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.title("Survival Functions")
    plt.grid()
    plt.savefig(f"results/{args.experiment_name}/{args.experiment_name}_survival_function.png")

    train_risk_scores = estimator.predict(X_train)
    test_risk_scores = estimator.predict(X_test)

    with open(f"results/{args.experiment_name}/{args.experiment_name}_metrics.txt","w") as f:
        # Accuracy Metrics
        concordance_index_censored = metrics.concordance_index_censored(Y_test["event_indicator"],Y_test["event_time"],test_risk_scores)
        concordance_index_ipcw = metrics.concordance_index_ipcw(Y_train.to_records(index=False),Y_test.to_records(index=False),test_risk_scores)
        brier_score = metrics.brier_score(Y_train.to_records(index=False),Y_test.to_records(index=False),np.tile(test_risk_scores[:,None],(1,100)),np.linspace(Y_test["event_time"].min()+1e-6,Y_test["event_time"].max()-1e-6,100))
        integrated_brier_score = metrics.integrated_brier_score(Y_train.to_records(index=False),Y_test.to_records(index=False),np.tile(test_risk_scores[:,None],(1,100)),np.linspace(Y_test["event_time"].min()+1e-6,Y_test["event_time"].max()-1e-6,100))
        cumulative_dynamic_auc = metrics.cumulative_dynamic_auc(Y_train.to_records(index=False),Y_test.to_records(index=False),test_risk_scores,np.linspace(Y_test["event_time"].min()+1e-6,Y_test["event_time"].max()-1e-6,100))
        plt.figure()
        plt.plot(brier_score[0],brier_score[1])
        plt.xlabel("Time")
        plt.ylabel("Brier Score")
        plt.title("Brier Score")
        plt.grid()
        plt.savefig(f"results/{args.experiment_name}/{args.experiment_name}_brier.png")
        plt.figure()
        plt.plot(np.linspace(Y_test["event_time"].min()+1e-6,Y_test["event_time"].max()-1e-6,100),cumulative_dynamic_auc[0])
        plt.xlabel("Time")
        plt.ylabel("Cumulative Dynamic AUC")
        plt.title("Cumulative Dynamic AUC")
        plt.grid()
        plt.savefig(f"results/{args.experiment_name}/{args.experiment_name}_cumulative_dynamic_auc.png")

        # Fairness metrics
        keya_individual_total, keya_individual_max = metrics.keya_individual_fairness(np.array(X_test),np.exp(test_risk_scores))
        keya_group = metrics.keya_group_fairness(np.exp(test_risk_scores),G_test.to_numpy()==np.unique(G_test.to_numpy())[:,None])
        keya_intersectional = metrics.keya_intersectional_fairness(np.exp(test_risk_scores),G_test.to_numpy()==np.unique(G_test.to_numpy())[:,None])

        # Reporting out
        print(f"Concordance Index Censored: {concordance_index_censored}")
        print(f"Concordance Index IPCW: {concordance_index_ipcw}")
        print(f"Brier Score: {brier_score}")
        print(f"Integrated Brier Score: {integrated_brier_score}")
        print(f"Cumulative Dynamic AUC: {cumulative_dynamic_auc}")
        print(f"Keya Individual: {keya_individual_total, keya_individual_max}")
        print(f"Keya Group: {keya_group}")
        print(f"Keya Intersectional: {keya_intersectional}")
        f.write(f"Concordance Index Censored: {concordance_index_censored}\n")
        f.write(f"Concordance Index IPCW: {concordance_index_ipcw}\n")
        f.write(f"Brier Score: {brier_score}\n")
        f.write(f"Integrated Brier Score: {integrated_brier_score}\n")
        f.write(f"Cumulative Dynamic AUC: {cumulative_dynamic_auc}\n")
        f.write(f"Keya Individual: {keya_individual_total, keya_individual_max}\n")
        f.write(f"Keya Group: {keya_group}\n")
        f.write(f"Keya Intersectional: {keya_intersectional}\n")

if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    print(f"Experiment took {end_time-start_time} seconds.")