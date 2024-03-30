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

## Code Organization

We use coding conventions based off of [scikit-survival](https://scikit-survival.readthedocs.io/en/stable).

`dataset_utils.py` details the loading and processing of a survival analysis dataset.

`metrics.py` details the metrics (accuracy, fairness) for evaluation on a survival analysis dataset and the survival predictions.

`data/` is a folder dedicated to holding any custom datasets.

`models/` is a folder dedicated to holding any saved models.