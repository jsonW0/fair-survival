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
pip install pygwalker
```

## Running Experiments

Run `bash scripts/[name_of_script].sh` to run an experiment.

Visualize the experiment results by running `python summarize_results.py`. That will produce `results/results.html` based on all experiments saved under `results`. This html is an interactive pygwalker visualizer.

## Background

In survival analysis, we observe $(x,T,\delta)$ triplets where $\delta$ is `True` if the event was actually observed and `False` if censored.
- The hazard function for an individual is the instantaneous probability of event occurrence at a time given survival up to that time: $h(t)=\lim_{\Delta t\to 0}\frac{\Pr[T\in[t,t+\Delta t)|T\geq t]}{\Delta t}$.
- The cumulative hazard function is the cumulative hazard until a specified time: $H(t)=\int_0^t h(u)du$.
- The survival function for an individual is the probability to survive that time: $S(t)=\Pr[T>t]=\exp(-H(t))$.

## Code Organization

We use coding conventions based off of [scikit-survival](https://scikit-survival.readthedocs.io/en/stable).

In particular, that means that a survival analysis model must follow this API:
- Fitting procedure invoked by `.fit(X,Y)`
- The baseline hazard rate recovered from `.cum_baseline_hazard_`, often determined via Breslow's estimator.
- The time-independent (individual-dependent) risk score recovered from `.predict(X)`
- The time-dependent cumulative hazard function recovered from `.predict_cumulative_hazard(X)`
- The time-dependent survival function recovered from `.predict_survival_function(X)`

For example, the Cox PH model's `.predict(X)` returns the $\beta^Tx$ that is part of the cumulative hazard function $H(t)=H_0(t)\exp(\beta^Tx)$.

Our code is organized as follows:
- `run_survival.py` is the script for running an experiment.
- `dataset_utils.py` details the loading and processing of a survival analysis dataset.
- `metrics.py` details the metrics (accuracy, fairness) for evaluation on a survival analysis dataset and the survival predictions.
- `survival_models.py` contains survival analysis models.
- `summarize_results.py` summarizes the results in `results/`.
- `scripts/` is a folder dedicated to holding any experiment bash scripts.
- `results/` is a folder deidcated to storing all results of the experiments