"""Tests for permutation entropy functions."""
# Authors: Christian Ferreyra, chrisferreyra13@gmail.com
# Date: 2023
# License: MIT License


import numpy as np

from ..permutation_entropy import (
    permutation_entropy, permutation_entropy_map, ordinal_patterns)


def test_permutation_entropy():
    """Test permutation entropy of a time series."""
    # original time series used by the authors
    x = np.array([4, 7, 9, 10, 6, 11, 3])

    pe_true = 1.5219

    # test without normalization
    pe = permutation_entropy(x, dim=3, base='2', normalized=False)
    np.testing.assert_almost_equal(pe, pe_true, decimal=3)

    # test with normalization
    pe = permutation_entropy(x, dim=3, base='2', normalized=True)
    np.testing.assert_almost_equal(pe, pe_true, decimal=3)


def test_permutation_entropy_map():
    """Test permutation entropy map of a time series map."""
    # original time series used by the authors
    x_row = np.array([4, 7, 9, 10, 6, 11, 3])
    x = np.array([x_row, x_row, x_row])
    fs = 1
    tw = len(x_row)
    pe_true = 1.5219*np.ones(x.shape)

    # test without normalization
    pe = permutation_entropy_map(
        x, fs, tw=tw, dim=3, base='2', normalized=False)
    np.testing.assert_almost_equal(pe, pe_true, decimal=3)

    # test with normalization
    pe = permutation_entropy_map(
        x, fs, tw=tw, dim=3, base='2', normalized=False)
    np.testing.assert_almost_equal(pe, pe_true, decimal=3)


def test_ordinal_patterns():
    """Test computation of ordinal patterns."""
    # original time series used by the authors
    x = np.array([4, 7, 9, 10, 6, 11, 3])

    patterns_true = np.array(
        [[0, 1, 2], [0, 1, 2], [1, 2, 0], [1, 0, 2], [1, 2, 0]])
    probs_true = np.array([2/5, 1/5, 2/5])

    # test just patterns
    patterns = ordinal_patterns(x, dim=3, tau=1, return_probs=False)
    np.testing.assert_array_equal(patterns, patterns_true)

    # test also the probs
    patterns, probs = ordinal_patterns(x, dim=3, tau=1, return_probs=True)
    np.testing.assert_array_equal(patterns, patterns_true)
    np.testing.assert_array_equal(probs, probs_true)
