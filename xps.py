""" 

Pascal M. Keller <pmk46@mrao.cam.ac.uk> 2021/22
Cavendish Astrophysics, University of Cambridge, UK

Compute cross-spectral densities

"""


import h5py
import numpy as np
import argparse
import astropy.units as u

from closurelib import dspec


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--inpath", help="Path to closure data.", type=str)
parser.add_argument("-o", "--outpath", help="Path to write cross-power spectra to.", type=str)
parser.add_argument("-s", "--scalingpath", help="File with scaling coefficients", type=str)
parser.add_argument("--fs", help="sampling frequency", type=float, default=10.24*1e-6)
parser.add_argument("-m", "--model", help="use model data.", action="store_true")
parser.add_argument("--fcut_low", help="Number of channels to cut from lower end of band.", type=int, default=0)
parser.add_argument("--fcut_high", help="Number of channels to cut from upper end of band.", type=int, default=0)

args = parser.parse_args()

if args.model:
    ifmodel = " model"
else:
    ifmodel = ""

# load data
with h5py.File(args.inpath, "r") as f:
    frq = f["FRQ"][()]
    lst = f["LST"][()]

    N = len(frq)
    frq = frq[:N//2]

    # trim band
    if args.fcut_high == 0:
        args.fcut_high = -N
    eicp1_xx = f[f"eicp XX (2){ifmodel}"][()][..., args.fcut_low:-args.fcut_high]
    eicp2_xx = f[f"eicp XX (4){ifmodel}"][()][..., args.fcut_low:-args.fcut_high]
    eicp1_yy = f[f"eicp YY (2){ifmodel}"][()][..., args.fcut_low:-args.fcut_high]
    eicp2_yy = f[f"eicp YY (4){ifmodel}"][()][..., args.fcut_low:-args.fcut_high]
    trlist_xx = f["triads XX"][()]
    trlist_yy = f["triads YY"][()]
    
# load scaling coefficients
scaling_array = np.loadtxt(args.scalingpath)

# compute noise realisations
noise_xx = (eicp1_xx[1] - eicp1_xx[0]) / np.sqrt(2)
noise_yy = (eicp1_yy[1] - eicp1_yy[0]) / np.sqrt(2)

# variance along frequency axis
var_xx = np.nanvar(noise_xx, axis=-1)
var_yy = np.nanvar(noise_yy, axis=-1)

# inverse variance weights
w_var_xx = 1 / np.moveaxis([np.outer(var_xx[:, i], var_xx[:, i]) for i in range(var_xx.shape[1])], 0, -1)
w_var_yy = 1 / np.moveaxis([np.outer(var_yy[:, i], var_yy[:, i]) for i in range(var_yy.shape[1])], 0, -1)

w_var_xx[np.where(np.isinf(w_var_xx))] = np.nan
w_var_yy[np.where(np.isinf(w_var_yy))] = np.nan

w_cross_xx = w_var_xx.copy()
w_cross_yy = w_var_yy.copy()

# set weight of diagonal triad pairs to zero (inter/cross triad)
for i in range(w_var_xx.shape[0]):
    for j in range(w_var_xx.shape[1]):
        if (trlist_xx[i] == trlist_xx[j]).sum() > 1:
            w_cross_xx[i, j] = 0

for i in range(w_var_yy.shape[0]):
    for j in range(w_var_yy.shape[1]):
        if (trlist_yy[i] == trlist_yy[j]).sum() > 1:
            w_cross_yy[i, j] = 0

# cross spectral densities between triads in different JD bins
xps_xx = dspec.xps_mat(eicp1_xx[0], eicp1_xx[1], fs=args.fs)
xps_yy = dspec.xps_mat(eicp1_yy[0], eicp1_yy[1], fs=args.fs)

# compute cross-power spectrum errors
noise_xx = dspec.xps_err(eicp2_xx, fs=args.fs)
noise_yy = dspec.xps_err(eicp2_yy, fs=args.fs)

# compute average triad pairs and time
xps_xx_avg = np.nansum(np.moveaxis(xps_xx, -1, 0) * w_cross_xx, axis=(1, 2, 3)) / np.nansum(w_cross_xx)
xps_yy_avg = np.nansum(np.moveaxis(xps_yy, -1, 0) * w_cross_yy, axis=(1, 2, 3)) / np.nansum(w_cross_yy)
noise_xx_avg = np.nansum(np.moveaxis(noise_xx, -1, 0) * w_cross_xx, axis=(2, 3, 4)) / np.nansum(w_cross_xx)
noise_yy_avg = np.nansum(np.moveaxis(noise_yy, -1, 0) * w_cross_yy, axis=(2, 3, 4)) / np.nansum(w_cross_yy)

# compute average over polarisations
w_xx = np.nansum(w_cross_xx) / (np.nansum(w_cross_xx) + np.nansum(w_cross_yy))
w_yy = np.nansum(w_cross_yy) / (np.nansum(w_cross_xx) + np.nansum(w_cross_yy))
xps_avg = (w_xx * xps_xx_avg + w_yy * xps_yy_avg)
noise_avg = (w_xx * noise_xx_avg + w_yy * noise_yy_avg)

err_avg = np.sqrt(np.mean(noise_avg.real**2, axis=1))
err_xx_avg = np.sqrt(np.mean(noise_xx_avg.real**2, axis=1))
err_yy_avg = np.sqrt(np.mean(noise_yy_avg.real**2, axis=1))

delay = dspec.get_delays(n=xps_avg.shape[-1], fs=args.fs/u.MHz)

# downsample (::2) and write to file
dnames = ["LST", "delay", f"XPS XX{ifmodel}", f"XPS YY{ifmodel}", f"ERR XX{ifmodel}", f"ERR YY{ifmodel}", f"XPS XX AVG{ifmodel}", f"XPS YY AVG{ifmodel}", f"ERR XX AVG{ifmodel}", 
f"ERR YY AVG{ifmodel}", f"Noise XX AVG{ifmodel}", f"Noise YY AVG{ifmodel}", f"XPS AVG{ifmodel}", f"ERR AVG{ifmodel}", f"Noise AVG{ifmodel}", f"weights var XX{ifmodel}", 
f"weights var YY{ifmodel}", f"weights cross XX{ifmodel}", f"weights cross YY{ifmodel}", f"scaling{ifmodel}"]

data_list = [lst, delay[..., ::2], xps_xx[..., ::2], xps_yy[..., ::2], noise_xx[::2], noise_yy[::2], xps_xx_avg[..., ::2], xps_yy_avg[..., ::2], err_xx_avg[..., ::2], err_yy_avg[..., ::2], noise_xx_avg[::2], noise_yy_avg[::2],
xps_avg[..., ::2], err_avg[..., ::2], noise_avg[::2], w_var_xx, w_var_yy, w_cross_xx, w_cross_yy, scaling_array]

f = h5py.File(args.outpath, "a")

for dname, data in zip(dnames, data_list):
    if dname in f.keys():
        del f[dname]
    f.create_dataset(dname, data=data)

f.close()