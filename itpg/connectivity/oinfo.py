"""O-Information."""
# Code inspired from BrainHack Marseille 2022 and Frites python package.
# Modified: Christian Ferreyra, chrisferreyra13@gmail.com
# Date: 01/2023

import numpy as np
from ..entropy.entropy_gaussian import entropy_gauss_nd, entropy_gauss


def compute_oinfo(x, ind):
    """Compute the O-info.

    Parameters
    ----------
    x : ndarray, shape (..., n_vars, n_trials)
        Multidimensional data array.
    ind : list
        Indices for tensor computations.

    Returns
    -------
    float
        O-Information.
    """
    nvars = x.shape[-2]
    o = (nvars - 2) * entropy_gauss_nd(x)
    o += (entropy_gauss_nd(x[..., np.newaxis, :])
          - entropy_gauss_nd(x[..., ind, :])).sum(-1)

    return o


def compute_oinfo_loop(x):
    nvars, _ = x.shape

    # (n - 2) * H(X^n)
    o = (nvars - 2) * entropy_gauss(x)

    for j in range(nvars):
        # sum_{j=1...n}( H(X_{j}) - H(X_{-j}^n) )
        o += entropy_gauss(x[[j], :]) - entropy_gauss(np.delete(x, j, axis=0))

    return o

# @jax.jit
# def oinfo_jax_tensor(x, ind):
#     nvars, _ = x.shape
#     o = (nvars - 2) * jnb_ent_g_nd(x[np.newaxis, ...])
#     o += (jnb_ent_g_nd(x[:, np.newaxis, :]) -
#           jnb_ent_g_nd(x[ind[:, 1:], :])).sum(0)
#     return o
