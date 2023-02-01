"""Data generators."""
# Code inspired from BrainHack Marseille 2022 and Frites python package.
# Authors: Christian Ferreyra, chrisferreyra13@gmail.com
# Date: 01/2023

import numpy as np


def generate_data(n_trials=300, n_roi=6, n_times=600):
    # generate the data
    x = np.random.rand(n_trials, n_roi, n_times)
    roi = np.array([f"r{r}" for r in range(n_roi)])
    trials = np.random.rand(n_trials)
    times = np.arange(n_times)

    return x, trials, times, roi


def set_lin_relationship(x, relationships, alpha=0.6):
    # last item is the dependent var
    for r in relationships:
        vars, sl = r
        vars = [int(v) for v in vars.split(',') if v.isnumeric()]
        win = np.hanning(sl.stop-sl.start).reshape(1, -1)
        win = alpha*win
        x[:, vars[-1], sl] += np.sum(x[:, vars[:-1], sl], axis=1)*win

    return x
