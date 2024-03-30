# Expanded Group Fairness for Survival Analysis
Lillian Sun and Jason Wang


Some code taken from https://github.com/umbc-sanjaylab/FISA_KDD22

## Setup
```
conda create -n fair-survival
conda activate fair-survival
conda install python=3.10
conda install -c sebp scikit-survival
pip install jupyter
```

## Code Organization

`dataset_utils.py` details the loading and processing of a survival analysis dataset.

`metrics.py` details the metrics (accuracy, fairness) for evaluation on a survival analysis dataset and the survival predictions.

`data/` is a folder dedicated to holding any custom datasets.

`models/` is a folder dedicated to holding any saved models.