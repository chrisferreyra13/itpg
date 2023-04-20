"""Permutation entropy for time series or a time series map."""
# Authors: Christian Ferreyra, chrisferreyra13@gmail.com
# Date: 2023
# License: MIT License

import numpy as np


def ordinal_patterns(x, dim=3, tau=1, return_probs=True):
    """Compute ordinal patters (permutations) from an array.

    Parameters
    ----------
    x : ndarray, shape(n_points,)
        Array to compute symbolization and get ordinal patterns.
    dim : int, optional (default 3)
        Embedding dimension.
    tau : int, optional (default 1)
        Embedding delay.
    return_probs : bool, optional
        If True, return the probabilities.

    Returns
    -------
    ndarray
        Ordinal patterns.
    ndarray
        Probabilities of each pattern (permutation).
    """
    n = x.shape[0]
    x = np.array([x])

    # create partitions, matrix of 'dim' rows and n-(dim-1)*tau columns
    partitions = np.concatenate(
        [[x[0, i:i+dim*tau:tau]] for i in range(n-(dim-1)*tau)]
    )
    # get the order of each column and the sort again to get the permutation
    order = np.apply_along_axis(np.argsort, 1, partitions)
    patterns = np.apply_along_axis(np.argsort, 1, order)

    if not return_probs:
        return patterns
    else:
        # compute probs based on observation frequency
        _, patterns_count = np.unique(
            patterns, return_counts=True, axis=0)
        probs = patterns_count/len(partitions)

        return patterns, probs


def permutation_entropy(x, dim=3, tau=1, base='2', normalized=True):
    """Compute permutation entropy.

    Note: Permutation entropy defined in Bandt and Pompe 2002.

    Parameters
    ----------
    x : ndarray, shape(n_points,)
        Array to compute symbolization and get ordinal patterns.
    dim : int, optional (default 3)
        Embedding dimension.
    tau : int, optional (default 1)
        Embedding delay.
    base : str, optional (default '2')
        Logarithm base for Shannon's entropy. Either '2' or 'e'.
    normalized : bool, optional (default True)
        If True, return normalized permutation entropy based of the maximum
        number of permutations.

    Returns
    -------
    float
        Permutation entropy value.

    Raises
    ------
    TypeError
        If 'x' is not a numpy array.
    TypeError
        If 2 is provided instead of '2'.
    ValueError
        If base value is neither '2' or 'e'.
    """

    if not isinstance(x, np.ndarray):
        raise TypeError("Array should be numpy array")

    if base not in ['2', 'e']:
        if base == 2:
            raise TypeError("Base 2 parameter should be a string")
        else:
            raise ValueError("Base parameter should be '2' or 'e'")

    _, probs = ordinal_patterns(x, dim, tau)

    # point to log function
    if base == '2':
        log_func = np.log2
    else:
        log_func = np.log

    if normalized:
        # pe_max is log(dim!) which is the max number of permutations
        pe_max = log_func(float(np.math.factorial(dim)))
        pe = -np.sum(probs*log_func(probs))
        return pe/pe_max
    else:
        return -np.sum(probs*log_func(probs))


def permutation_entropy_map(x, fs, tw=1, dim=3, base='2', normalized=True):
    """Compute permutation entropy of a time series map.

    Note: Permutation entropy defined in Bandt and Pompe 2002.

    Parameters
    ----------
    x : ndarray, shape(n_points,)
        Array to compute symbolization and get ordinal patterns.
    fs : float
        Sampling frequency.
    tw : float, optional (default 1)
        Time window in seconds to split the time series.
    dim : int, optional (default 3)
        Embedding dimension.
    tau : int, optional (default 1)
        Embedding delay.
    base : str, optional (default '2')
        Logarithm base for Shannon's entropy. Either '2' or 'e'.
    normalized : bool, optional (default True)
        If True, return normalized permutation entropy based of the maximum
        number of permutations.

    Returns
    -------
    float
        Permutation entropy map.
    """
    n_rows, n_times = x.shape
    pe_map = np.zeros((n_rows, n_times))
    slices = [slice(int(k*tw*fs), int((k+1)*tw*fs))
              for k in range(int(n_times/(tw*fs)))]
    for j, time_series in enumerate(x):
        for s in slices:
            pe_map[j, s] = permutation_entropy(
                time_series[s], dim=dim, base=base, normalized=normalized)

    return pe_map
