"""Useful vizualization functions for higher order interactions."""
# Code source from BrainHack Marseille 2022 and Frites python package.
# Modified: Christian Ferreyra, chrisferreyra13@gmail.com
# Date: 01/2023

import numpy as np
import matplotlib.pyplot as plt
from frites import set_mpl_style

from cfutils.workflow import get_figs_path

set_mpl_style()


def plot_oinfo(oinfo, output_path=None, fig_name='oinfo_map'):
    """Plot O-Information in time per multiplet."""
    # print(oinfo)
    vmin, vmax = np.nanpercentile(oinfo.data, [1, 99])
    minmax = max(abs(vmin), abs(vmax))
    vmin, vmax = -minmax, minmax

    # plot the results
    df = oinfo.to_pandas()
    fig, axs = plt.subplots(1, 1)
    m = axs.pcolormesh(
        df.columns, df.index, df.values, cmap='RdBu_r', vmin=vmin, vmax=vmax
    )
    plt.colorbar(m)
    axs.set_xlabel('Times')
    axs.axvline(0., color='k')

    if output_path:
        fig_path = get_figs_path(output_path, fig_name)
        plt.savefig(fig_path+'.png', dpi='figure', format='png')
        plt.close(fig)
        return fig_path
    else:
        plt.show(block=True)
        return None
