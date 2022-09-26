""" 

Pascal M. Keller <pmk46@mrao.cam.ac.uk> 2021/22
Cavendish Astrophysics, University of Cambridge, UK

Tools for working with closure phase data

"""

import warnings
import numpy as np
from itertools import product


def remove_nan_slices(data, axis, return_index=False):
    """Remove all-nan slices from data array

    Parameters
    ----------
    data : ndarray
        data array
    axis : int
        axis along which to remove all-nan slices
    return_index : bool, optional
        If True, return indices of 'good' data along axis, by default False

    Returns
    -------
    ndarray
        data with all-nan slices along axis removed
    """

    naxes = len(data.shape)
    data = np.moveaxis(data, axis, 0)
    axes = tuple(np.arange(1, naxes))
    bad_data = np.all(np.isnan(data), axis=axes)
    data = np.moveaxis(data[~bad_data], 0, axis)

    if return_index:
        return data, np.where(~bad_data)
    else:
        return data


def flag_repeated_slices(data, axis, raxis):
    """Flag slices which repeate, i.e. slices which are
    equal to one of their neighbouring slices along a given axis.

    Parameters
    ----------
    data : ndarray
        data array
    axis : int
        axis along which to flag repeated slices
    raxis : int
        axis along which to compare repeats

    Returns
    -------
    data array
        flags with same shape as data
    """

    # rearrange axes
    axis = tuple(np.atleast_1d(axis))
    raxis = tuple(np.atleast_1d(raxis))
    axes = raxis + axis
    new_axes = np.arange(-1, -len(axes) - 1, -1)
    data = np.moveaxis(data, axes, new_axes)
    flags = np.zeros_like(data)
    shape = data.shape

    # axes to iterate over
    iter_axes = np.arange(0, len(shape) - len(axes))
    iter_ranges = (range(shape[i]) for i in range(len(iter_axes)))

    # iterate over axes
    for idx in product(*iter_ranges):

        # iterate over slices
        for i in range(shape[-len(axes)] - 1):
            slices = data[idx]

            # compare current slice with next slice
            if np.isclose(slices[i], slices[i + 1], equal_nan=True).all():
                flags[idx][[i, i + 1]] = True

    data = np.moveaxis(data, new_axes, axes)

    return np.moveaxis(flags, new_axes, axes).astype(bool)


def select(data, axval, vmin, vmax, axis=-1):
    """Select subset of data given axis values and range

    Parameters
    ----------
    data : ndarray
        data array
    axval : 1D array
        associated axis values
    vmin : float
        minimum value
    vmax : float
        maximum value
    axis : int
        axis along which to select subset, by default -1

    Returns
    -------
    ndarray
        subset of data array, such that new axis values lie within range [vmin, vmax]
    """

    data = np.moveaxis(data, axis, 0)
    axval = np.asarray(axval)
    idx = np.where((axval <= vmax) & (axval >= vmin))[0]

    return np.moveaxis(data[idx], 0, axis), axval[idx]


def phase_diff(phi1, phi2):
    """Compute absolute phase difference

    Parameters
    ----------
    phi1 : ndarray
        phase1
    phi2 : ndarray
        phase2

    Returns
    -------
    ndarray
        Absolute phase differences
    """

    dphi = np.abs(phi1 - phi2)

    if np.size(dphi) > 1:
        index = np.where(dphi > np.pi)
        dphi[index] = 2 * np.pi - dphi[index]

    elif dphi > np.pi:
        dphi = 2 * np.pi - dphi

    return dphi


def geomed(data, weights=None, axis=0):
    """Geometric Median

    Parameters
    ----------
    data : ndarray
        data to be averaged
    weights : ndarray, optional
        data weights, by default None
    axis : int, optional
        axis along which to compute average, by default 0

    Returns
    -------
    ndarray
        geometric median of data
    """

    with warnings.catch_warnings():

        # reshape phase data
        N = np.size(axis)
        data = np.moveaxis(data, axis, np.arange(N))
        shape = [np.prod(data.shape[:N])] + list(data.shape[N:])
        data = data.reshape(shape)

        # compute pointwise distances in data
        d = np.abs(data[np.newaxis, :] - data[:, np.newaxis])

        # set weights
        if not isinstance(weights, np.ndarray):
            weights = np.ones(d.shape[1])
        weights = np.abs(weights)

        # compute average
        d = np.nanmean(np.moveaxis(d, 1, -1) * weights, axis=-1)

        # mask nan-values
        d = np.ma.masked_array(d, np.isnan(d))

        try:
            # get data point that minimises its distance with all other data points
            index = np.nanargmin(d, axis=0)
            index = (index,) + tuple(np.indices(index.shape))
            data = data[index]
        except:
            data = np.nan * np.ones(shape[1:])

    return data
