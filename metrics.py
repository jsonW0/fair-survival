from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, brier_score, integrated_brier_score, cumulative_dynamic_auc
from typing import Iterable
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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
        (accuracy1, accuracy2): array-like, shape = (2)
            Test accuracies of model without access to t_hat and model with access to t_hat
    '''
    model1 = LogisticRegression()
    model1.fit(X_train, censored_train)
    predictions1 = model1.predict(X_test)
    accuracy1 = accuracy_score(censored_test, predictions1)

    X_train_with_t_hat = np.column_stack((X_train, t_hat_train))
    X_test_with_t_hat = np.column_stack((X_test, t_hat_test))
    model2 = LogisticRegression()
    model2.fit(X_train_with_t_hat, censored_train)
    predictions2 = model2.predict(X_test_with_t_hat)
    accuracy2 = accuracy_score(censored_test, predictions2)

    return accuracy1, accuracy2

def equal_opportunity(X: Iterable[Iterable[float]], g: Iterable[int], t: Iterable[int], t_hat: Iterable[float], num_bins: int):
    '''
    Computes the equal opportunity probabilities of the discretized bins of the survival times 
    
    Args: 
        X: array-like, shape = (n_samples, n_features)
            Data matrix
        
        g: array-like, shape = (n_samples)
            Binary vector representing the sensitive attributes of each data point

        t: array-like, shape = (n_samples)
            Vector containing the true survival times (even for censored data since this is generated synthetic dataset)
        
        t_hat: array-like, shape = (n_samples)
            Vector containing the 50% percentile of the survival analysis PDF
        
        num_bins: int
            Number of bins to discretize the survival time range into
        
    Returns:
        eo_probs: array-like, shape = (num_bins, 2)
            Equal opportunity probabilities for each survival time bin, 2 for each bin (1 for each value of the sensitive attribute)
    '''
    X = np.array(X)
    g = np.array(g)
    t = np.array(t)
    t_hat = np.array(t_hat)

    combined = np.concatenate((t, t_hat))
    bins = np.linspace(np.min(combined), np.max(combined), num=num_bins+1)
    digitized_t = np.digitize(t, bins) - 1  # bin indices from 0 to num_bins-1 for true times
    digitized_t_hat = np.digitize(t_hat, bins) - 1  # bin indices for predicted times

    eo_probs = np.zeros((num_bins, 2))

    for i in range(num_bins):
        for j in range(2):
            mask = (digitized_t == i) & (g == j)
            if np.sum(mask) > 0:
                correct_predictions = np.sum(digitized_t[mask] == digitized_t_hat[mask])
                total_predictions = np.sum(mask)
                eo_probs[i, j] = correct_predictions / total_predictions if total_predictions > 0 else 0

    return eo_probs

def keya_individual_fairness(X: Iterable[Iterable[float]], individual_risk: Iterable[float]):
    '''
    Computes individual fairness as proposed by Keya et al. 2021. I.e., the sum of positive difference between risk space and individual space. The distance in individual space is the Euclidean distance.

    $$\sum_{i,j}\max(0,|h_i-h_j|-||x_i-x_j||_2)$$

    Args:
        X: array-like, shape = (n_samples, n_features)
            Data matrix

        risk: array-like, shape = (n_samples,)
            Estimated individual risk of experiencing an event

    Returns:
        total: float
            The total deviation from individual fairness (summed, as in the original paper)
        max_deviation float:
            The max deviation from individual fairness
    '''
    individual_risk = np.exp(individual_risk)
    total_deviation = 0
    max_deviation = 0
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            deviation = max(0,np.abs(individual_risk[i]-individual_risk[j])-np.linalg.norm(X[i]-X[j],ord=2))
            total_deviation+=deviation
            max_deviation = max(max_deviation,deviation)
    return total_deviation, max_deviation

def keya_group_fairness(individual_risk: Iterable[float], groups: Iterable[float]):
    '''
    Computes group fairness as proposed by Keya et al. 2021. I.e., the max difference in group average risk and overall average risk.

    $$\max_{a\in A}|\overline{h}(a)-\E_{x\in \mathcal{X}}[\overline{h}(x)]|$$ where $\overline{h}$ is the average hazard.

    Args:
        individual_risk: array-like, shape = (n_samples,)
            Estimated individual risk of experiencing an event

        groups: array-like, shape = (n_samples,)
            List of group membership for each individual

    Returns:
        float: the computed group fairness.
    '''
    individual_risk = np.exp(individual_risk)
    avg_risk = np.mean(individual_risk)
    group_avg_risks = [np.mean(individual_risk[groups==group]) for group in np.unique(groups)]
    statistic = np.max(np.abs(group_avg_risks-avg_risk))
    return statistic

def keya_intersectional_fairness(S: Iterable[Iterable[bool]], individual_risk: Iterable[float]):
    '''
    Computes intersectional fairness as proposed by Keya et al. 2021. I.e., the max difference (in log space) between any two groups.

    $$\max_{a,b\in A}|\overline{h}(a)-\overline{h}(b)|$$ where $\overline{h}$ is the average hazard.

    Args:
        S: array-like, shape = (n_groups, n_samples)
            List of demographic group memberships

        individual_risk: array-like, shape = (n_samples,)
            Estimated individual risk of experiencing an event

    Returns:
        float: the computed group fairness.
    '''
    individual_risk = np.exp(individual_risk)
    group_avg_risks = [np.mean(individual_risk[S[i]]) for i in range(S.shape[0])]
    max_diff = 0.
    for i in range(S.shape[0]):
        for j in range(i+1,S.shape[0]):
            max_diff = max(max_diff,np.abs(np.log(group_avg_risks[i])-np.log(group_avg_risks[j])))
    return max_diff