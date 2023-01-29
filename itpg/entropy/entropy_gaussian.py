"""Gaussian entropy."""
# Code inspired from BrainHack Marseille 2022 and Frites python package.
# Authors: Christian Ferreyra, chrisferreyra13@gmail.com
# Date: 01/2023

import numpy as np


def entropy_gauss_nd(x):
    """Entropy of a tensor of shape (..., n_vars, n_trials)."""
    nvarx, ntrl = x.shape[-2], x.shape[-1]

    # covariance
    c = np.einsum('...ij, ...kj->...ik', x, x)
    c /= float(ntrl - 1.)
    chc = np.linalg.cholesky(c)

    # entropy in nats
    hx = (np.log(np.einsum('...ii->...i', chc)).sum(-1)
          + 0.5 * nvarx * (np.log(2 * np.pi) + 1.0))

    return hx


def entropy_gauss(x):
    nvarx, ntrl = x.shape

    # covariance
    c = np.dot(x, x.T) / float(ntrl - 1)
    chc = np.linalg.cholesky(c)

    # entropy in nats
    hx = np.sum(np.log(np.diag(chc))) + 0.5 * nvarx * (
        np.log(2 * np.pi) + 1.0)
    return hx


def entropy_gauss_loop(x):
    h = []
    for k in range(x.shape[0]):
        h.append(entropy_gauss(x[k, ...]))
    return h


# @jax.jit
# def jnb_ent_g_nd(x):
#     _, nvarx, ntrl = x.shape

#     # covariance
#     c = jnp.einsum('...ij, ...kj->...ik', x, x)
#     c /= float(ntrl - 1.)
#     chc = jnp.linalg.cholesky(c)

#     # entropy in nats
#     hx = jnp.log(jnp.einsum('...ii->...i', chc)).sum(-1) + 0.5 * nvarx * (
#         jnp.log(2 * jnp.pi) + 1.0)
#     return hx
