"""Useful functions for computing higher order interactions."""
# Code source from BrainHack Marseille 2022 and Frites python package.
# Modified: Christian Ferreyra, chrisferreyra13@gmail.com
# Date: 01/2023

import numpy as np
import itertools


def get_combinations(n, k, roi, task_related=False):
    """Get combinations."""
    combs = np.array(list(itertools.combinations(np.arange(n), k)))

    # add target (behaviour) as a final columns
    if task_related:
        combs = np.c_[combs, np.full((combs.shape[0],), n)]

    # build region names
    roi_st = ['-'.join(r) for r in roi[combs].tolist()]

    return combs, roi_st
