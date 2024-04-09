# Expanded Group Fairness for Survival Analysis
Lillian Sun and Jason Wang

Survival analysis is the statistical problem of modeling the time until an event of interest occurs, such as death, failure, or recovery. It is widely used in various domains, including healthcare, engineering, and social sciences. However, survival analysis often faces the challenge of censorship, where the event of interest is not observed for some individuals within the study period (due to study termination or lack of follow-ups). Now, instead of observing $(x,y)$ pairs (where $y$ is output), we observe $(x,T,\delta)$ triplets, where $T$ denotes the survival time and $\delta$ is an indicator for the event occurring or the data-point being censored (i.e., the study ends and the true survival time is in the unobservable future). The presence of censorship in survival analysis data can lead to biased estimates and inaccurate predictions if not properly accounted for. For example, it may be that only the longer survival times of a particular demographic group were censored compared to another demographic group.

## Setup
```
conda create -n fair-survival
conda activate fair-survival
conda install python=3.10
conda install -c sebp scikit-survival
pip install jupyter
pip install plotly
pip install seaborn
```

## Background

In survival analysis, we observe $(x,T,\delta)$ triplets where $\delta$ is `True` if the event was actually observed and `False` if censored.
- The hazard function for an individual is the instantaneous probability of event occurrence at a time given survival up to that time: $\lim_{\Delta t\to 0}\frac{\Pr[T\in[t,t+\Delta t)|T\geq t]}{\Delta t}$.
- The cumulative hazard function is the cumulative hazard until a specified time: $H(t)=\int_0^t h(u)du$.
- The survival function for an individual is the probability to survive that time: $S(t)=\Pr[T>t]=\exp(-H(t))$.

## Code Organization

We use coding conventions based off of [scikit-survival](https://scikit-survival.readthedocs.io/en/stable).

In particular, that means that a survival analysis model must follow this API:
- Fitting procedure invoked by `.fit(X,Y)`
- Hazard function recovered from `.predict(X)`
- Cumulative hazard function recovered from `.predict_cumulative_hazard(X)`
- Survival function recovered from `.predict_survival_function(X)`

Our code is organized as follows:
- `dataset_utils.py` details the loading and processing of a survival analysis dataset.
- `metrics.py` details the metrics (accuracy, fairness) for evaluation on a survival analysis dataset and the survival predictions.
- `data/` is a folder dedicated to holding any custom datasets.
- `models/` is a folder dedicated to holding any saved models.

## Citations
Keya et al. 2021: https://github.com/kkeya1/FairSurv