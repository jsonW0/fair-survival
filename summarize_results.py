import os
import sys
import pickle
import numpy as np
import pandas as pd
import pygwalker as pyg
class HidePrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

data = {}
for directory in os.listdir("results"):
    try:
        with open(f"results/{directory}/{directory}_args.pickle","rb") as f:
            args = pickle.load(f)
        with open(f"results/{directory}/{directory}_results.pickle","rb") as f:
            results = pickle.load(f)
        data[directory] = vars(args) | results
    except:
        print(f"Failed to read results/{directory}")
results = pd.DataFrame.from_dict(data).T
with HidePrint():
    walker = pyg.walk(results)
    with open(f"results/results.html","w") as f:
        f.write(walker.to_html())