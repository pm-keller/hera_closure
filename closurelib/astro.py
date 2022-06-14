""" 
Astronomy Tools
"""


import numpy as np
import scipy.integrate as integrate


"""constants in SI units"""
kb = 1.3806503e-23  # Boltzmann [J/K]
c = 2.99792458e8  # speed of light [m/s]
f_HI = 1.4204057517667e9  # H I rest frequency [Hz]
l_HI = c / f_HI  # H I rest wavelength [m]
pc = 3.085678e16  # one parsec in metres [m]

"""standard terms in cosmology"""
H0 = 100  # Hubbles constant [km/s/Mpc]
h = H0 / 100.0  # H0 = h * 100 km/s/Mpc
Omega_M = 0.308  # density parameter
Omega_vac = 1.0 - Omega_M  # vacuum density
Omega_K = 0.0  # curvature parameter (0 - flat)
Omega_L = 0.692  # cosmological constant


def lconvert(l, IN, OUT):
    """Convert units of length

    Args:
        l (float): length
        IN (str): unit of length l
        OUT (str): unit to convert l to

    Raises:
        Exception: IN or OUT not contained in unit dictionary

    Returns:
        float: legth l in units specified in OUT
    """

    global pc, h
    units = {"m": 1e0, "Mpc": pc * 1e6, "Mpc/h": pc * 1e6 / h}

    if IN not in units or OUT not in units:
        raise Exception(
            "IN or OUT not specified in unit dictionary! Choose from {}".format(units)
        )
    else:
        return l * units[IN] / units[OUT]


def ftoz(fobs, fem=f_HI):
    """Compute redshift

    Args:
        fobs (float): observerd frequency
        fem (float): emmitted frequency. Defaults to f_HI (H I line).

    Returns:
        float: redshift of source
    """

    return (fem - fobs) / fobs


def ltoz(lobs, lem=l_HI):
    """Compute redshift

    Args:
        lobs (float): observerd wavelength
        lem (float): emmitted wavelength. Defaults to l_HI (H I line).

    Returns:
        float: redshift of source
    """

    return (lobs - lem) / lem


def Ec(z, Omega_M=Omega_M, Omega_K=Omega_K, Omega_L=Omega_L):
    """Standard term in cosmology

    Args:
        z (float): redshift
        Omega_M (float, optional): density parameter. Defaults to Omega_M.
        Omega_K (float, optional): curvature parameter. Defaults to Omega_K.
        Omega_L (float, optional): cosmological constant. Defaults to Omega_L.

    Returns:
        float: E(z)
    """

    return np.sqrt(Omega_M * (1 + z) ** 3 + Omega_K * (1 + z) ** 2 + Omega_L)


def Dc(z, unit="Mpc/h", H0=H0, **kwargs):
    """Compute comoving distance

    Args:
        z (float): redshift
        unit (str): units of Dc. Defaults to "Mpc/h"
        H0 (float, optional): Hubbles constant. Defaults to H0.

    Returns:
        float: comoving distance of source at redshift z
    """

    z = np.array(z, ndmin=1)
    res = []

    # Hubble time [s] and distance [m]
    tH = 1 / (H0 * 1e3)
    DH = c * tH
    
    global h
    h = H0 / 100

    # integrate E(z) to compute Dc
    for i in z:
        E = lambda z: 1.0 / Ec(z, **kwargs)
        I = DH * integrate.quad(E, 0, i)[0]
        res.append(lconvert(I, "Mpc", unit))

    return np.squeeze(res)


def dDc(B, fc, f0=f_HI, unit="Mpc/h", H0=H0, **kwargs):
    """Compute comoving depth

    Args:
        z (float): redshift
        fc (float): centre frequency
        f0 (float, optional): rest frequency. Defaults to f_HI.
        unit (str, optional): units of comoving depth. Defaults to "Mpc/h".
        H0 (float, optional): Hubbles constant. Defaults to H0.

    Returns:
        float: comoving depth
    """

    # Hubble time [s] and distance [m]
    tH = 1 / (H0 * 1e3)
    DH = c * tH

    global h
    h = H0 / 100

    # redshift
    z = ftoz(fc)

    # comoving depth corresponding to centre frequency f0 and bandwidth B
    dD = B * DH * (1 + z) ** 2 / f_HI / Ec(z, **kwargs)

    return lconvert(dD, "Mpc", unit)