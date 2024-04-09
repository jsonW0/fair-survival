from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, brier_score, integrated_brier_score, cumulative_dynamic_auc
from typing import Iterable
import numpy as np

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
        float: the computed individual fairness.
    '''
    individual_risk = np.exp(individual_risk)
    total = 0
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            total+=max(0,np.abs(individual_risk[i]-individual_risk[j])-np.linalg.norm(X[i]-X[j],ord=2))
    return total

def keya_group_fairness(S: Iterable[Iterable[bool]], individual_risk: Iterable[float]):
    '''
    Computes group fairness as proposed by Keya et al. 2021. I.e., the max difference in group average risk and overall average risk.

    $$\max_{a\in A}|\overline{h}(a)-\E_{x\in \mathcal{X}}[\overline{h}(x)]|$$ where $\overline{h}$ is the average hazard.

    Args:
        S: array-like, shape = (n_groups, n_samples)
            List of demographic group memberships (`S[i][j]` is `True` when individual `j` is in group `i`)

        individual_risk: array-like, shape = (n_samples,)
            Estimated individual risk of experiencing an event

    Returns:
        float: the computed group fairness.
    '''
    individual_risk = np.exp(individual_risk)
    avg_risk = np.mean(individual_risk)
    group_avg_risks = [np.mean(individual_risk[S[i]]) for i in range(S.shape[0])]
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
    avg_risk = np.mean(individual_risk)
    group_avg_risks = [np.mean(individual_risk[S[i]]) for i in range(S.shape[0])]
    max_diff = 0.
    for i in range(S.shape[0]):
        for j in range(i+1,S.shape[0]):
            max_diff = max(max_diff,np.abs(np.log(group_avg_risks[i])-np.log(group_avg_risks[j])))
    return max_diff
