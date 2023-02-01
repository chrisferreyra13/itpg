"""Useful vizualization functions for higher order interactions."""
# Code inspired from BrainHack Marseille 2022 and Frites python package.
# Authors: Christian Ferreyra, chrisferreyra13@gmail.com
# Date: 01/2023

import numpy as np
import matplotlib.pyplot as plt
from frites import set_mpl_style
set_mpl_style()


def plot_oinfo(oinfo):
    """Plot O-Information in time per multiplet"""
    # print(oinfo)
    vmin, vmax = np.nanpercentile(oinfo.data, [1, 99])
    minmax = max(abs(vmin), abs(vmax))
    vmin, vmax = -minmax, minmax

    # plot the results
    df = oinfo.to_pandas()
    plt.pcolormesh(
        df.columns, df.index, df.values, cmap='RdBu_r', vmin=vmin, vmax=vmax
    )
    plt.colorbar()
    plt.xlabel('Times')
    plt.axvline(0., color='k')

    for n_k, k in enumerate(oinfo['roi'].data):
        plt.gca().get_yticklabels()[n_k]

    plt.show()
