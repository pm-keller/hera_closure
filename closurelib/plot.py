""" 

Pascal M. Keller <pmk46@mrao.cam.ac.uk> 2021/22
Cavendish Astrophysics, University of Cambridge, UK

Modules for plotting closure phase data

"""

import numpy as np
import astropy.units as u

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from closurelib import dspec

NoneType = type(None)


def spectrogram(data, lst, frq, ax=None, return_im=False, **kwargs):
    """
    Plot spectrogram of data. LST and frequency are given in hours and MHz respectively.
    """

    if isinstance(ax, NoneType):
        fig, ax = plt.subplots(1, 1)

    im = ax.imshow(
        data,
        interpolation=None,
        extent=(np.min(frq), np.max(frq), np.max(lst), np.min(lst)),
        aspect="auto",
        **kwargs
    )

    plt.setp(
        ax,
        xlabel="Frequency (MHz)",
        ylabel="LST (h)",
        xlim=[min(frq), max(frq)],
        ylim=[max(lst), min(lst)],
    )

    ax.minorticks_on()

    if return_im:
        return ax, im
    else:
        return ax


def bar(ax, x, y, yerr, colour="k"):
    """Plot a box error bar

    Args:
        ax (class): matplotlib ax
        x (ndarray): x-axis values
        y (ndarray): y-axis values
        yerr (ndarray): y-axis error
    """

    width = 0.3 * np.mean(x[1:] - x[:-1])
    height = 2 * yerr
    coord = np.array([x - width / 2.0, y - yerr])

    for i in range(len(yerr)):
        ax.add_patch(Rectangle(coord[:, i], width, height[i], alpha=0.1, color=colour))


def xps_plot(power, error, freq, fs=10.24*u.MHz**-1, nsig=1, bls=None, linthresh=1e-4, color="k", onesided=False, cosmo=False, tau_ax=False, ax=None):
    """
    Plot a closure phase delay power-spectrum
    """
    # get delays
    delay = dspec.get_delays(power.shape[-1], fs=fs)

    # average in |k//|-bins
    if onesided:
        idx = np.where(delay >= 0)[0]
        delay = (delay - np.flip(delay))[idx] / 2
        power = (power + np.flip(power, axis=-1))[idx] / 2
        error = np.sqrt((error**2 + np.flip(error, axis=-1)**2) / 4)[idx]
        xlim = np.array([0, 5])
    else:
        xlim = np.array([-5, 5])

    # convert delays to k parallel
    k = dspec.get_k_parallel(delay, freq).to(u.Mpc ** -1).value
    idx = np.where(k > 1)

    # create figure and layout
    if isinstance(ax, NoneType):
        fig, ax = plt.subplots(1, 1)

    # cosmological power spectrum (triangle plot)
    if cosmo:
        assert not isinstance(bls, NoneType), "Baseline length not specified!"
        k_perp = dspec.get_k_perpendicular(bls, freq).to(u.Mpc ** -1).value
        k = np.sqrt(k**2 + k_perp**2)
        power *= k**3 / (2 * np.pi**2)

        ax.scatter(k, power.real, marker="o", s=50, color=color, label="Positive",)

    else:
        # plot power
        ax.scatter(k, power.real, marker="o", s=50, color=color, label="Real",)
        ax.scatter(k, power.imag, marker="o", s=50, c="None", edgecolor=color, alpha=0.5, label="Imaginary")

    if not isinstance(error, NoneType):
        rms = np.sqrt(np.mean(error.real[idx]**2))

        if cosmo:
            k_1 = k
            error *= k_1**3 / (2 * np.pi**2)
            rms *= k_1**3 / (2 * np.pi**2)
        else:
            k_1 = k[idx]
            rms *= np.ones_like(k_1)            

        # plot error bars
        bar(ax, k, power, error * nsig)

        # plot RMS lines
        ax.plot(k_1, rms, color="k", alpha=1, linestyle="--", label="RMS")
        ax.plot(k_1, rms, color="k", alpha=1, linestyle="--")
        ax.plot(k_1, -rms, color="k", alpha=1, linestyle="--")
        ax.plot(k_1, -rms, color="k", alpha=1, linestyle="--")

    ax.hlines(np.min(k), np.max(k), 0, linewidth=0.5)

    # set axes limits and tick labels
    ax.set_xlim([np.min(k), np.max(k)])
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)

    # delay axis
    dax = ax.twiny()
    dax.set_xlim(xlim)

    if tau_ax:
        dax.set_xlabel(r"$\tau$ ($\mu\mathrm{s}$)", fontsize=18)
        dax.tick_params(axis="x", labelsize=14)
    else:
        dax.set_xticklabels([])

    # yscale
    ax.set_yscale("symlog", linthresh=linthresh)

    # ticks, grid and layout
    plt.tight_layout()
    ax.minorticks_on()
    dax.minorticks_on()
    ax.grid(linestyle="--", alpha=0.5)

    return ax, rms