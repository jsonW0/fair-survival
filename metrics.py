from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, brier_score, integrated_brier_score, cumulative_dynamic_auc
from typing import Iterable

def individual_fairness(event_indicator: Iterable[bool], event_time: Iterable[float], estimate: Iterable[float]):
    '''
    Computes individual fairness as proposed by Keya et al. 2021. I.e., the sum of positive difference between hazard space and individual space. The distance in individual space is the Euclidean distance.

    $$\sum_{i,j}\max(0,|h_i-h_j|-||x_i-x_j||_2)$$

    Args:
        event_indicator: array-like, shape = (n_samples,)
            Boolean array denotes whether an event occurred

        event_time: array-like, shape = (n_samples,)
            Array containing the time of an event or time of censoring

        estimate: array-like, shape = (n_samples,)
            Estimated risk of experiencing an event

    Returns:
        float: the computed individual fairness.

    Raises:
        NotImplementedError: Dataset name was unknown and the .csv file was not found.
    '''
    
