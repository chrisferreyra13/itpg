"""Connectivity of higher order interactions."""
# Code inspired from BrainHack Marseille 2022 and Frites python package.
# Modified: Christian Ferreyra, chrisferreyra13@gmail.com
# Date: 01/2023

import numpy as np
import xarray as xr

from frites.conn import conn_io
from frites.io import logger, check_attrs
from frites.core import copnorm_nd

from .utils import get_combinations
from .oinfo import compute_oinfo


def conn_hoi(data, y=None, times=None, roi=None, minsize=3, maxsize=5,
             verbose=None):
    """Dynamic, possibly task-related, higher-order interactions.

    Parameters
    ----------
    data : array_like
        Electrophysiological data. Several input types are supported:

            * Standard NumPy arrays of shape (n_epochs, n_roi, n_times)
            * mne.Epochs
            * xarray.DataArray of shape (n_epochs, n_roi, n_times)

    y : array_like
        The feature of shape (n_trials,) for estimating task-related O-info.
    roi : array_like | None
        Array of region of interest name of shape (n_roi,)
    times : array_like | None
        Array of time points of shape (n_times,)
    minsize, maxsize : int | 3, 5
        Minimum and maximum size of the multiplets

    Returns
    -------
    oinfo : array_like
        The O-info array of shape (n_multiplets, n_times) where positive values
        reflect redundant dominated interactions and negative values stand for
        synergistic dominated interactions.
    """
    # ________________________________ INPUTS _________________________________
    # inputs conversion
    is_task_related = isinstance(y, (str, list, np.ndarray, tuple))
    kw_links = {'directed': False, 'net': False}
    data, cfg = conn_io(
        data, y=y, times=times, roi=roi, name='DynOinfo', verbose=verbose,
        kw_links=kw_links
    )

    # extract variables
    x, attrs = data.data, cfg['attrs']
    y, roi, times = data['y'].data, data['roi'].data, data['times'].data
    n_roi = len(roi)

    # get the maximum size of the multiplets investigated
    if not isinstance(maxsize, int):
        maxsize = n_roi
    maxsize = max(1, maxsize)
    assert maxsize > minsize

    logger.info(f"Compute the {'task-related ' * is_task_related} HOI "
                f"(min={minsize}; max={maxsize})")

    # ________________________________ O-INFO _________________________________
    logger.info("    Copnorm the data")

    # for task-related, add behavior along spatial dimension
    if is_task_related:
        y = np.tile(y.reshape(-1, 1, 1), (1, 1, len(times)))
        x = np.concatenate((x, y), axis=1)
        roi = np.r_[roi, ['beh']]

    # copnorm and demean the data
    x = copnorm_nd(x.copy(), axis=0)
    x = (x - x.mean(axis=0, keepdims=True))

    # make the data (n_times, n_roi, n_trials)
    x = x.transpose(2, 1, 0)

    oinfo, roi_o = [], []
    for msize in range(minsize, maxsize + 1):
        # ------------------------------ INDICES ------------------------------
        ish = msize if not is_task_related else msize + 1
        ind = np.zeros((ish, ish), dtype=int)
        vec = np.arange(ish)
        for shift in range(ish):
            ind[shift, :] = np.roll(vec, -shift)
        ind = ind[:, 1:]

        # ----------------------------- MULTIPLETS ----------------------------
        logger.info(f"    Multiplets of size {msize}")
        combs, _roi_o = get_combinations(
            n_roi, msize, roi, task_related=is_task_related)
        roi_o += _roi_o

        # ------------------------------- O-INFO ------------------------------
        for mult in combs:
            _oinfo = compute_oinfo(x[:, mult, :], ind)
            oinfo += [_oinfo]
    oinfo = np.stack(oinfo, 0)

    # _______________________________ OUTPUTS _________________________________
    attrs.update(dict(
        task_related=is_task_related, minsize=minsize, maxsize=maxsize
    ))
    oinfo = xr.DataArray(
        oinfo, dims=('roi', 'times'), coords=(roi_o, times), name="Oinfo",
        attrs=check_attrs(attrs)
    )

    return oinfo
