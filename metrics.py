from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, brier_score, integrated_brier_score, cumulative_dynamic_auc
from typing import Iterable, Optional, Callable
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_distances

def adversarial_censorship_fairness(X_train: Iterable[Iterable[float]], X_test: Iterable[Iterable[float]], censored_train: Iterable[int], censored_test: Iterable[int], t_hat_train: Iterable[float], t_hat_test: Iterable[float]):
    '''
    Computes the accuracies of two instances of logistic regression model, one trained on (X, censored) and another trained on ((X, t_hat), censored)
    
    Args: 
        X_train: array-like, shape = (n_samples, n_features)
            Data matrix (train)

        X_test: array-like, shape = (n_samples, n_features)
            Data matrix (test)

        censored_train: array-like, shape = (n_samples)
            Binary vector representing delta, whether a data point was censored (train)
        
        censored_test: array-like, shape = (n_samples)
            Binary vector representing delta, whether a data point was censored (test)

        t_hat_train: array-like, shape = (n_samples)
            Vector containing the 50% percentile of the survival analysis PDF (train)
        
        t_hat_test: array-like, shape = (n_samples)
            Vector containing the 50% percentile of the survival analysis PDF (test)

    Returns:
        (accuracy2_train-accuracy1_train, accuracy2_test-accuracy1_test, accuracy1_train, accuracy2_train, accuracy1_test, accuracy2_test): array-like, shape = (6)
            Train and test accuracies of model without access to t_hat and model with access to t_hat
    '''
    model1 = LogisticRegression()
    model1.fit(X_train, censored_train)
    predictions1_train = model1.predict(X_train)
    accuracy1_train = accuracy_score(censored_train, predictions1_train)
    predictions1_test = model1.predict(X_test)
    accuracy1_test = accuracy_score(censored_test, predictions1_test)

    X_train_with_t_hat = np.column_stack((X_train, t_hat_train))
    X_test_with_t_hat = np.column_stack((X_test, t_hat_test))
    model2 = LogisticRegression()
    model2.fit(X_train_with_t_hat, censored_train)
    predictions2_train = model2.predict(X_train_with_t_hat)
    accuracy2_train = accuracy_score(censored_train, predictions2_train)
    predictions2_test = model2.predict(X_test_with_t_hat)
    accuracy2_test = accuracy_score(censored_test, predictions2_test)

    return accuracy2_train-accuracy1_train, accuracy2_test-accuracy1_test, accuracy1_train, accuracy2_train, accuracy1_test, accuracy2_test

def equal_opportunity(X: Iterable[Iterable[float]], groups: Iterable[int], t: Iterable[int], t_hat: Iterable[float], num_bins: int):
    '''
    Computes the equal opportunity probabilities of the discretized bins of the survival times 
    
    Pr[t_hat=k|i=A,t=k]=Pr[t_hat=k|i=B,t=k]=Pr[t_hat=k|i=C,t=k], then holds for all k
    Compute the max of the max pairwise group deviations
    
    Args: 
        X: array-like, shape = (n_samples, n_features)
            Data matrix
        
        groups: array-like, shape = (n_samples)
            Binary vector representing the sensitive attributes of each data point

        t: array-like, shape = (n_samples)
            Vector containing the true survival times (even for censored data since this is generated synthetic dataset)
        
        t_hat: array-like, shape = (n_samples)
            Vector containing the 50% percentile of the survival analysis PDF
        
        num_bins: int
            Number of bins to discretize the survival time range into
        
    Returns:
        average_max_deviation: float
            Average max deviation in equal probability
        max_max_deviation: float
            Max max deviation in equal probability
    '''
    combined = np.concatenate((t, t_hat))
    bins = np.linspace(np.min(combined), np.max(combined), num=num_bins+1)
    digitized_t = np.digitize(t, bins) - 1  # bin indices from 0 to num_bins-1 for true times
    digitized_t_hat = np.digitize(t_hat, bins) - 1  # bin indices for predicted times

    eo_probs = np.zeros((num_bins, len(np.unique(groups))))
    for i in range(num_bins):
        for j,group in enumerate(np.unique(groups)):
            eo_probs[i,j] = np.mean(digitized_t[(digitized_t==i) & (groups==group)]==digitized_t_hat[(digitized_t==i) & (groups==group)]) if np.sum((digitized_t==i) & (groups==group))>0 else 0
    return np.mean([np.max(row) - np.min(row) for row in eo_probs]),np.max([np.max(row) - np.min(row) for row in eo_probs])

def keya_individual_fairness(X: Iterable[Iterable[float]], estimate: Iterable[float], alpha: Optional[float] = 1., distance: Optional[Callable] = lambda x,y: np.linalg.norm(x-y,ord=2)):
    '''
    Computes individual fairness as proposed by Keya et al. 2021. I.e., the sum of positive difference between estimate space and attribute space.

    $$\sum_{i,j}\max(0,|h_i-h_j|-||x_i-x_j||_2)$$

    Args:
        X: array-like, shape = (n_samples, n_features)
            Data matrix

        estimate: array-like, shape = (n_samples,)
            Estimate for each individual (can be exponentiated base risk, survival probability at time t, etc.)
        
        alpha: float
            Scaling factor to compare distance in estimate space and feature space. Default: 1.0
        
        distance: Callable
            Specify the distance in attribute space. Default: Euclidean distance

    Returns:
        total_deviation: float
            The total deviation from individual fairness (summed, as in the original paper)
        max_deviation float:
            The max deviation from individual fairness
    '''
    total_deviation = 0
    max_deviation = 0
    for i in range(X.shape[0]):
        for j in range(i+1):
            deviation = max(0,np.abs(estimate[i]-estimate[j])-alpha*distance(X[i],X[j]))
            total_deviation+=deviation
            max_deviation = max(max_deviation,deviation)
    return total_deviation, max_deviation

def keya_group_fairness(estimate: Iterable[float], group_membership: Iterable[Iterable[bool]]):
    '''
    Computes group fairness as proposed by Keya et al. 2021. I.e., the max difference in group estimate and overall estimate.

    $$\max_{a\in A}|\overline{h}(a)-\E_{x\in \mathcal{X}}[\overline{h}(x)]|$$ where $\overline{h}$ is the average hazard.

    Args:
        estimate: array-like, shape = (n_samples,)
            Estimate for each individual (can be exponentiated base risk, survival probability at time t, etc.)

        group_membership: array-like, shape = (n_groups, n_samples)
            List of demographic group memberships (`S[i][j]` is `True` when individual `j` is in group `i`)

    Returns:
        float: the computed group fairness.
    '''
    avg_estimate = np.mean(estimate)
    group_avg_estimates = [np.mean(estimate[group_membership[i]]) for i in range(group_membership.shape[0])]
    statistic = np.max(np.abs(group_avg_estimates-avg_estimate))
    return statistic

def keya_intersectional_fairness(estimate: Iterable[float], group_membership: Iterable[Iterable[bool]]):
    '''
    Computes intersectional fairness as proposed by Keya et al. 2021. I.e., the max difference (in log space) between any two groups.

    $$\max_{a,b\in A}|\overline{h}(a)-\overline{h}(b)|$$ where $\overline{h}$ is the average hazard.

    Args:
        estimate: array-like, shape = (n_samples,)
            Estimate for each individual (can be exponentiated base risk, survival probability at time t, etc.)

        group_membership: array-like, shape = (n_groups, n_samples)
            List of demographic group memberships (`S[i][j]` is `True` when individual `j` is in group `i`)

    Returns:
        float: the computed group fairness.
    '''
    group_avg_estimates = [np.mean(estimate[group_membership[i]]) for i in range(group_membership.shape[0])]
    max_diff = 0.
    for i in range(group_membership.shape[0]):
        for j in range(i+1,group_membership.shape[0]):
            max_diff = max(max_diff,np.abs(np.log(group_avg_estimates[i])-np.log(group_avg_estimates[j])))
    return max_diff

def rahman_censorship_individual_fairness(X: Iterable[Iterable[float]], estimate: Iterable[float], event_time: Iterable[float], event_indicator: Iterable[float], alpha: Optional[float] = 1., distance: Optional[Callable] = lambda x,y: np.linalg.norm(x-y,ord=2)):
    '''
    Computes censorship-based individual fairness as proposed by Rahman et al. 2023.

    Args:
        X: array-like, shape = (n_samples, n_features)
            Data matrix

        estimate: array-like, shape = (n_samples,)
            Estimate for each individual (can be exponentiated base risk, survival probability at time t, etc.)
        
        event_time: array-like, shape = (n_samples,)
            Observed event_time for each individual

        event_indicator: array-like, shape = (n_samples,)
            List of booleans that are True when the individual's event was uncensored and False when the individual's event was censored

        alpha: float
            Scaling factor to compare distance in estimate space and feature space. Default: 1.0
        
        distance: Callable
            Specify the distance in attribute space. Default: Euclidean distance

    Returns:
        total_deviation: float
            The total deviation from individual fairness
        max_deviation float:
            The max deviation from individual fairness
    '''
    total_deviation = 0
    max_deviation = 0
    for i in np.argwhere(~event_indicator): #censored
        for j in np.argwhere(event_indicator): # uncensored
            if event_time[i]<=event_time[j]:
                deviation = max(0,np.abs(estimate[i]-estimate[j])-alpha*distance(X[i],X[j]))
                total_deviation+=deviation
                max_deviation = max(max_deviation,deviation)
    return float(total_deviation/(len(np.argwhere(~event_indicator))*len(np.argwhere(event_indicator)))), float(max_deviation)


def rahman_censorship_group_fairness(X: Iterable[Iterable[float]], estimate: Iterable[float], group_membership: Iterable[Iterable[bool]], event_time: Iterable[float], event_indicator: Iterable[float], alpha: Optional[float] = 1., distance: Optional[Callable] = lambda x,y: np.linalg.norm(x-y,ord=2)):
    '''
    Computes censorship-based individual fairness as proposed by Rahman et al. 2023.

    Args:
        X: array-like, shape = (n_samples, n_features)
            Data matrix

        estimate: array-like, shape = (n_samples,)
            Estimate for each individual (can be exponentiated base risk, survival probability at time t, etc.)

        group_membership: array-like, shape = (n_groups, n_samples)
            List of demographic group memberships (`S[i][j]` is `True` when individual `j` is in group `i`)
        
        event_time: array-like, shape = (n_samples,)
            Observed event_time for each individual

        event_indicator: array-like, shape = (n_samples,)
            List of booleans that are True when the individual's event was uncensored and False when the individual's event was censored

        alpha: float
            Scaling factor to compare distance in estimate space and feature space. Default: 1.0
        
        distance: Callable
            Specify the distance in attribute space. Default: Euclidean distance

    Returns:
        total_deviation: float
            The total deviation from individual fairness
        max_deviation float:
            The max deviation from individual fairness
    '''
    total_deviation = 0
    max_deviation = 0
    for group in range(group_membership.shape[0]):
        for i in np.argwhere((~event_indicator)[group_membership[group]]): #censored
            for j in np.argwhere(event_indicator[group_membership[group]]): # uncensored
                if event_time[i]<=event_time[j]:
                    deviation = max(0,np.abs(estimate[group_membership[group]][i]-estimate[group_membership[group]][j])-alpha*distance(X[group_membership[group]][i],X[group_membership[group]][j]))
                    total_deviation+=deviation
                    max_deviation = max(max_deviation,deviation)
    return float(total_deviation/(len(np.argwhere(~event_indicator))*len(np.argwhere(event_indicator)))), float(max_deviation)