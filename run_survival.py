import os
import sys
import time
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pygwalker as pyg
from sksurv.util import check_y_survival
from survival_models import FittedUniformBaseline
from sksurv.nonparametric import SurvivalFunctionEstimator
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.svm import FastSurvivalSVM
from dataset_utils import load_dataset, preprocess_dataset, generate_synthetic_dataset
import metrics

class HidePrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
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
    parser.add_argument('--num_trials', action='store', type=int, required=False, default=1, help='Number of trials')
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
    with open(f"results/{args.experiment_name}/{args.experiment_name}_args.pickle","wb") as f:
        pickle.dump(args,f)

    results = {
        "concordance_index_censored": [],
        "concordance_index_ipcw": [],
        "integrated_brier_score": [],
        "cumulative_dynamic_auc": [],
        "keya_individual_total": [], 
        "keya_individual_max": [], 
        "keya_group": [], 
        "keya_intersectional": [],
        "rahman_censorship_individual_total": [],
        "rahman_censorship_individual_max": [],
        "rahman_censorship_group_total": [],
        "rahman_censorship_group_max": [],
        "average_equal_opportunity": [],
        "max_equal_opportunity": [],
        "adversarial_censorship_train": [],
        "adversarial_censorship_test": [],
        "adversarial_censorship_train_base": [],
        "adversarial_censorship_test_base": [],
        "adversarial_censorship_train_plus": [],
        "adversarial_censorship_test_plus": [],
    }
    for trial in range(args.num_trials):
        #################################################################################

        # Load survival 
        if args.dataset=="synthetic":
            dataset, true_times = generate_synthetic_dataset(N=args.N,G=args.G,D=args.D,repr=np.array(args.repr).reshape(args.G),censorship_repr=np.array(args.censorship_repr).reshape(args.G),mean=np.array(args.mean).reshape(args.G,args.D),std=np.array(args.std).reshape(args.G,args.D,args.D),scale=np.array(args.scale).reshape(args.G),shape=np.array(args.shape).reshape(args.G),censorship_mean=np.array(args.censorship_mean).reshape(args.G,args.D),censorship_temp=np.array(args.censorship_temp).reshape(args.G),censorship_times=np.array(args.censorship_times).reshape(args.G,2),seed=args.seed+trial)
        else:
            dataset = load_dataset(args.dataset)
            true_times = None
        dataset.to_csv(f"results/{args.experiment_name}/dataset_{trial}.csv")
        # with HidePrint():
        #     walker = pyg.walk(dataset)
        #     with open(f"results/{args.experiment_name}/dataset_{trial}.html","w") as f:
        #         f.write(walker.to_html())
        X_train, X_test, Y_train, Y_test, G_train, G_test, indices_train, indices_test = preprocess_dataset(dataset)
        
        train_true_times = true_times[indices_train] if true_times is not None else None
        test_true_times = true_times[indices_test] if true_times is not None else None

        #################################################################################

        # Train an estimator
        if args.model == "coxph":
            estimator = CoxPHSurvivalAnalysis(alpha=0.1).fit(X_train, Y_train.to_records(index=False))
        elif args.model == "uniform":
            estimator = FittedUniformBaseline(alpha=0.1).fit(X_train, Y_train.to_records(index=False))
        elif args.model == "randomforest":
            estimator = RandomSurvivalForest(n_estimators=100).fit(X_train, Y_train.to_records(index=False))
        else:
            raise NotImplementedError(f"{args.model} has not been implemented.")

        #################################################################################

        # Evaluate the model
        surv_funcs = estimator.predict_survival_function(X_test)
        plt.figure()
        for fn in surv_funcs:
            plt.step(fn.x, fn(fn.x), where="post", alpha=0.5, c="tab:blue")
        plt.ylim(0, 1)
        plt.xlabel("Time")
        plt.ylabel("Survival Probability")
        plt.title("Survival Functions")
        plt.grid()
        plt.savefig(f"results/{args.experiment_name}/{args.experiment_name}_survival_function_{trial}.png")

        # train_risk_scores = np.exp(estimator.predict(X_train)
        # test_risk_scores = np.exp(estimator.predict(X_test))
        train_risk_scores = estimator.predict(X_train)
        test_risk_scores = estimator.predict(X_test)

        train_half_life = np.array([fn.x[np.argmax(fn(fn.x)<=0.5)] for fn in estimator.predict_survival_function(X_train)])
        test_half_life = np.array([fn.x[np.argmax(fn(fn.x)<=0.5)] for fn in estimator.predict_survival_function(X_test)])

        # Some wizardry to crop the times correctly
        min_time = Y_train[Y_train["event_indicator"]]["event_time"].min()
        max_time = Y_train[Y_train["event_indicator"]]["event_time"].max()
        _, test_time = check_y_survival(Y_test[Y_test["event_time"]<max_time].to_records(index=False))
        min_time2 = test_time.min()
        max_time2 = test_time.max()

        with open(f"results/{args.experiment_name}/{args.experiment_name}_metrics_{trial}.txt","w") as f:
            # Accuracy Metrics
            concordance_index_censored = metrics.concordance_index_censored(Y_test["event_indicator"],Y_test["event_time"],test_risk_scores)
            concordance_index_ipcw = metrics.concordance_index_ipcw(Y_train.to_records(index=False),Y_test.to_records(index=False),test_risk_scores,tau=max_time)
            brier_score = metrics.brier_score(Y_train.to_records(index=False),Y_test[Y_test["event_time"]<max_time].to_records(index=False),np.tile(test_risk_scores[Y_test["event_time"]<max_time][:,None],(1,100)),np.linspace(min_time2,max_time2,102)[1:-1])
            integrated_brier_score = metrics.integrated_brier_score(Y_train.to_records(index=False),Y_test[Y_test["event_time"]<max_time].to_records(index=False),np.tile(test_risk_scores[Y_test["event_time"]<max_time][:,None],(1,100)),np.linspace(min_time2,max_time2,102)[1:-1])
            cumulative_dynamic_auc = metrics.cumulative_dynamic_auc(Y_train.to_records(index=False),Y_test[Y_test["event_time"]<max_time].to_records(index=False),test_risk_scores[Y_test["event_time"]<max_time],np.linspace(min_time2,max_time2,122)[11:-11])
            plt.figure()
            plt.plot(brier_score[0],brier_score[1])
            plt.xlabel("Time")
            plt.ylabel("Brier Score")
            plt.title("Brier Score")
            plt.grid()
            plt.savefig(f"results/{args.experiment_name}/{args.experiment_name}_brier_{trial}.png")
            plt.figure()
            plt.plot(np.linspace(Y_train["event_time"].min(),Y_train["event_time"].max(),102)[1:-1],cumulative_dynamic_auc[0])
            plt.xlabel("Time")
            plt.ylabel("Cumulative Dynamic AUC")
            plt.title("Cumulative Dynamic AUC")
            plt.grid()
            plt.savefig(f"results/{args.experiment_name}/{args.experiment_name}_cumulative_dynamic_auc_{trial}.png")

            # Fairness metrics
            keya_individual_total, keya_individual_max = metrics.keya_individual_fairness(X_test.to_numpy(),test_risk_scores)
            keya_group = metrics.keya_group_fairness(test_risk_scores,G_test.to_numpy()==np.unique(G_test.to_numpy())[:,None])
            keya_intersectional = metrics.keya_intersectional_fairness(test_risk_scores,G_test.to_numpy()==np.unique(G_test.to_numpy())[:,None])
            rahman_censorship_individual_total, rahman_censorship_individual_max = metrics.rahman_censorship_individual_fairness(X_test.to_numpy(),test_risk_scores,Y_test["event_time"].to_numpy(),Y_test["event_indicator"].to_numpy())
            rahman_censorship_group_total, rahman_censorship_group_max = metrics.rahman_censorship_group_fairness(X_test.to_numpy(),test_risk_scores,G_test.to_numpy()==np.unique(G_test.to_numpy())[:,None],Y_test["event_time"].to_numpy(),Y_test["event_indicator"].to_numpy())
            average_equal_opportunity, max_equal_opportunity = metrics.equal_opportunity(X_test.to_numpy(),G_test.to_numpy(),test_true_times if test_true_times is not None else Y_test["event_time"].to_numpy(),test_half_life,5)
            adversarial_censorship = metrics.adversarial_censorship_fairness(X_train.to_numpy(),X_test.to_numpy(),Y_train["event_indicator"].to_numpy(),Y_test["event_indicator"].to_numpy(),train_risk_scores,test_risk_scores)

            # Reporting out
            print(f"Concordance Index Censored: {concordance_index_censored}")
            print(f"Concordance Index IPCW: {concordance_index_ipcw}")
            # print(f"Brier Score: {brier_score}")
            print(f"Integrated Brier Score: {integrated_brier_score}")
            print(f"Cumulative Dynamic AUC: {cumulative_dynamic_auc[-1]}")
            print(f"Keya Individual: {keya_individual_total, keya_individual_max}")
            print(f"Keya Group: {keya_group}")
            print(f"Keya Intersectional: {keya_intersectional}")
            print(f"Rahman Individual: {rahman_censorship_individual_total, rahman_censorship_individual_max}")
            print(f"Rahman Group: {rahman_censorship_group_total, rahman_censorship_group_max}")
            print(f"Equal Opportunity: {average_equal_opportunity, max_equal_opportunity}")
            print(f"Adversarial Censorship: {adversarial_censorship}")
            f.write(f"Concordance Index Censored: {concordance_index_censored}\n")
            f.write(f"Concordance Index IPCW: {concordance_index_ipcw}\n")
            f.write(f"Brier Score: {brier_score}\n")
            f.write(f"Integrated Brier Score: {integrated_brier_score}\n")
            f.write(f"Cumulative Dynamic AUC: {cumulative_dynamic_auc}\n")
            f.write(f"Keya Individual Total: {keya_individual_total}\n")
            f.write(f"Keya Individual Max: {keya_individual_max}\n")
            f.write(f"Keya Group: {keya_group}\n")
            f.write(f"Keya Intersectional: {keya_intersectional}\n")
            f.write(f"Rahman Individual Total: {rahman_censorship_individual_total}\n")
            f.write(f"Rahman Individual Max: {rahman_censorship_individual_max}\n")
            f.write(f"Rahman Group Total: {rahman_censorship_group_total}\n")
            f.write(f"Rahman Group Max: {rahman_censorship_group_max}\n")
            f.write(f"Equal Opportunity: {average_equal_opportunity, max_equal_opportunity}\n")
            f.write(f"Adversarial Censorship: {adversarial_censorship}\n")

            # Aggregating over trials
            results["concordance_index_censored"].append(concordance_index_censored[0])
            results["concordance_index_ipcw"].append(concordance_index_ipcw[0])
            # results["brier_score"].append(brier_score)
            results["integrated_brier_score"].append(integrated_brier_score)
            results["cumulative_dynamic_auc"].append(cumulative_dynamic_auc[-1])
            results["keya_individual_total"].append(keya_individual_total)
            results["keya_individual_max"].append(keya_individual_max)
            results["keya_group"].append(keya_group)
            results["keya_intersectional"].append(keya_intersectional)
            results["rahman_censorship_individual_total"].append(rahman_censorship_individual_total)
            results["rahman_censorship_individual_max"].append(rahman_censorship_individual_max)
            results["rahman_censorship_group_total"].append(rahman_censorship_group_total)
            results["rahman_censorship_group_max"].append(rahman_censorship_group_max)
            results["average_equal_opportunity"].append(average_equal_opportunity)
            results["max_equal_opportunity"].append(max_equal_opportunity)
            results["adversarial_censorship_train"].append(adversarial_censorship[0]),
            results["adversarial_censorship_test"].append(adversarial_censorship[1]),
            results["adversarial_censorship_train_base"].append(adversarial_censorship[2]),
            results["adversarial_censorship_test_base"].append(adversarial_censorship[4]),
            results["adversarial_censorship_train_plus"].append(adversarial_censorship[3]),
            results["adversarial_censorship_test_plus"].append(adversarial_censorship[5]),

    # Aggregate results
    for key in list(results.keys()):
        results[key+"__mean"] = np.nanmean(results[key]) if not isinstance(results[key][0],tuple) else tuple(np.nanmean([results[key][j][i] for j in range(args.num_trials)]) for i in range(len(results[key][0])))
        results[key+"__std"] = np.nanstd(results[key]) if not isinstance(results[key][0],tuple) else tuple(np.nanstd([results[key][j][i] for j in range(args.num_trials)]) for i in range(len(results[key][0])))
        # compute as 95% CI as +-2std/sqrt(n)

    with open(f"results/{args.experiment_name}/{args.experiment_name}_results.txt","w") as f:
        print(f"\n\n================================\n{args.experiment_name}\n================================")
        f.write(f"{args.experiment_name}\n")
        for key,value in results.items():
            print(f"{key}: {value}")
            f.write(f"{key}: {value}\n")
    with open(f"results/{args.experiment_name}/{args.experiment_name}_results.pickle","wb") as f:
        pickle.dump({key:value for key,value in results.items() if key.endswith("__mean") or key.endswith("__std")},f)

if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    print(f"Experiment took {end_time-start_time} seconds.")