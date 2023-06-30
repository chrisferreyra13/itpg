"""Gaussian entropy."""
# Source code from Frites python package.
# Modified: Christian Ferreyra, chrisferreyra13@gmail.com
# Date: 01/2023

import numpy as np


def entropy_gauss_nd(x):
    """Entropy of a gaussian tensor of shape (..., n_vars, n_trials)."""
    nvarx, ntrl = x.shape[-2], x.shape[-1]
    # sample covariance
    # the variables are gaussian with zero mean
    # so cov(x,x) = sum(xx^T)/N-1
    c = np.einsum('...ij, ...kj->...ik', x, x)
    c /= float(ntrl - 1.)
    # c = L*L.H in order to compute the determinant
    chc = np.linalg.cholesky(c)
    # |c|=|chc|^2, |chc|=(product of the diagonal elements of chc)

    # entropy in nats
    # 0.5*log(|chc|^2) = sum(log(diag(chc))) --> log of product is sum of log
    hx = (np.log(np.einsum('...ii->...i', chc)).sum(-1)
          + 0.5 * nvarx * (np.log(2 * np.pi) + 1.0))

    return hx


def entropy_gauss(x):
    """Entropy of a gaussian random process of shape (n_vars, v_trials)."""
    nvarx, ntrl = x.shape
    # sample covariance
    # the variables are gaussian with zero mean
    c = np.dot(x, x.T) / float(ntrl - 1)
    # c = chc*chc^(conjugate transpose)
    chc = np.linalg.cholesky(c)
    # |c|=|chc|^2, |chc|=(product of the diagonal elements of chc)

    # entropy in nats
    # 0.5*log(|chc|^2) = sum(log(diag(chc))) --> log of product is sum of log
    hx = np.sum(np.log(np.diag(chc))) + 0.5 * nvarx * (
        np.log(2 * np.pi) + 1.0)

    return hx


def entropy_gauss_loop(x):
    """Entropy of a gaussian random process."""
    # x with shape (n_times, n_vars, n_trials)
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
