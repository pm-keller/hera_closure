""" 
Modules for computing delay spectra
"""

import numpy as np
import scipy.signal

import astropy.units as u
from astropy.cosmology import Planck18

f_HI = 1.4204057517667e9 * u.Hz  # H I rest frequency [Hz]
c = 2.99792458e8 * u.m / u.s  # speed of light [m/s]
kb = 1.3806503e-23 * u.J / u.K  # Boltzmann [J/K]

def welch(
    v1, v2, fs=1, nperseg=None, noverlap=None, W="blackmanharris", inverse=False, shift=False):
    """Compute Cross-Power Spectrum using Welch's method

    Args:
        v1 (array): input array 1
        v2 (array): input array 2
        fs (float): sampling frequency. Defaults to 1.
        nperseg (int): number of data points per segment. Defaults to None.
        noverlap (int): number of overlapping data points between segments. Defaults to None.
        W (array or str): window function. Defaults to None.
        inverse (bool): If True, perform inverse FFT. Defaults to False.
        shift (bool): If True, shift FFT to centre. Defaults to False.

    Returns:
        array: cross-power spectrum of v1 and v2
    """
    
    if inverse:
        v1 = np.flip(v1, axis=-1)
        v2 = np.flip(v2, axis=-1)
        
    if nperseg == None:
        nperseg = v1.shape[-1]
    
    # compute cross spectral density
    csd = scipy.signal.csd(v1, v2, fs=fs, window=W, nperseg=nperseg, noverlap=noverlap, scaling="density", return_onesided=False, average="mean", detrend=False)[1]
    
    # shift to centre
    if shift:
        csd = np.fft.fftshift(csd, axes=-1)
        
    return csd * csd.shape[-1] / fs



def xps_mat(v1, v2, fs):
    """Compute cross power spectra between two sets of data

    Args:
        v1 (ndarray): data array 1
        v2 (ndarray): data array 2
        fs (float): sampling frequency

    Returns:
        ndarray: cross power spectrum matrix.
        The element at [i, j] is the cross-power of v1[i] and v2[j].
    """
    N = len(v1)
    
    mat = np.array([
        [
            welch(v1[i], v2[j], fs=fs, nperseg=v1.shape[-1], W="blackmanharris", inverse=False, shift=True)
            for i in range(N)
        ]
        for j in range(N)
    ], 
        dtype="complex128")

    return mat

"""
# Noise only
def xps_err(v, fs):
    # Compute differences
    dv = (v[np.newaxis, :] - v[:, np.newaxis]) / 2

    # Create pairs of independent closure phase differences
    indices = [[0, 1, 2, 3], [0, 2, 1, 3], [0, 3, 1, 2]]
    dv = np.array([[dv[i, j], dv[k, l]] for [i, j, k, l] in indices])

    # Compute cross-power spectrum for each pair of independent closure phase differences
    dv_xps = np.array([xps_mat(dv[i, 0], dv[i, 1], fs) for i in range(3)])
    return dv_xps
"""

# Noise + Signal-Noise cross-terms
def xps_err(v, fs):
    """Compute uncertainties for cross power spectrum
    
    Args:
        v (ndarray): first axis must contain four independent samples of data which are used for differencing
        fs (float): sampling frequency
    
    Returns:
        ndarray: cross power spectrum uncertainties
    """
    N = len(v)

    # cross power
    v_xps = np.array([[xps_mat(v[i], v[j], fs) for i in range(N)] for j in range(N)])

    # differences
    indices = [[0, 1, 2, 3], [0, 2, 1, 3], [0, 3, 1, 2]]
    dv_xps = np.array([(v_xps[i, j] - v_xps[k, l]) / (2 * np.sqrt(2)) for [i, j, k, l] in indices])

    return dv_xps

def ftoz(fobs, fem=f_HI):
    """
    Frequency to redshift
    """
    return (fem - fobs) / fobs

def get_delays(n, fs=10.24*u.MHz**-1):
        """
        Get delay array
        """
        return np.fft.fftshift(np.fft.fftfreq(n, 1 / fs))

def get_k_parallel(delay, freq, cosmo=Planck18):
    """
    Parallel k
    """
    z = ftoz(freq)

    return 2 * np.pi * delay * f_HI * cosmo.H(z) / cosmo.h / c / (1 + z) ** 2

def get_k_perpendicular(bls, freq, cosmo=Planck18):
    """
    Perpendicular k
    """
    lam = c / freq
    z = ftoz(freq, f_HI)

    return 2 * np.pi * np.abs(bls) / lam / cosmo.comoving_distance(z) / cosmo.h