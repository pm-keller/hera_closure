
"""
Compute cross-power spectrum errors
"""

import h5py
import numpy as np
import argparse
from itertools import product 

from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cppath", help="Path to closure data.", type=str)
parser.add_argument("-x", "--xpspath", help="Path to cross-power spectr.", type=str)
parser.add_argument("-s", "--scalingpath", help="File with scaling coefficients", type=str)
parser.add_argument("--fs", help="sampling frequency", type=float, default=10.24*1e-6)

args = parser.parse_args()

# load data
with h5py.File(args.cppath, "r") as f:
    lst = f["LST"][()]
    eicp1_xx = f["eicp XX (2)"][()]
    eicp2_xx = f["eicp XX (4)"][()]
    eicp1_yy = f["eicp YY (2)"][()]
    eicp2_yy = f["eicp YY (4)"][()]
    cov_xx = f["cov XX"][()]
    cov_yy = f["cov YY"][()]

with h5py.File(args.xpspath, "r") as f:
    lst = f["LST"][()]
    xps_xx = f["XPS XX"][()]
    xps_yy = f["XPS YY"][()]
    w_xx = f["weights cross XX"][()]
    w_yy = f["weights cross YY"][()]

Nd = eicp1_xx.shape[-1]

# load scaling coefficients
scaling_array = np.loadtxt(args.scalingpath)

# flag NaN-triads
tr_flags_xx = np.isnan(eicp2_xx).any(axis=(0, 2, 3))
eicp1_xx = eicp1_xx[:, ~tr_flags_xx]

tr_flags_yy = np.isnan(eicp2_yy).any(axis=(0, 2, 3))
eicp1_yy = eicp1_yy[:, ~tr_flags_yy]

# flag galactic plane
glx_flags = (lst > 6.25) & (lst < 10.75)
eicp1_xx = eicp1_xx[:, :, ~glx_flags]
eicp1_yy = eicp1_yy[:, :, ~glx_flags]
eicp2_xx = eicp2_xx[:, :, ~glx_flags]
eicp2_yy = eicp2_yy[:, :, ~glx_flags]
scaling_array = scaling_array[~glx_flags]
lst = lst[~glx_flags]

# noise-only variance
def noise_var(w, cov, A, idx):
    w = w / np.nansum(w)
    trange = 4 * (range(w.shape[0]),)
    return [w[i, j, idx] * w[k, l, idx] * cov[idx, i, k] * cov[idx, j, l] * A[idx]**2 for i, j, k, l in product(*trange)]

# noise-signal cross-terms
def xterm(w, cov, xps, A, idx):
    w = w / np.nansum(w)
    trange = 4 * (range(w.shape[0]),)
    xps = xps.real
    xps[np.where(xps < 0)] = 0
    return [w[i, j, idx] * w[k, l, idx] * (cov[idx, i, k] * (xps[j, l, idx] - np.sqrt(cov[idx, j, j] * cov[idx, l, l] / 4 / np.pi)) + cov[idx, j, l] * (xps[i, k, idx] - np.sqrt(cov[idx, j, j] * cov[idx, l, l] / 4 / np.pi))) * A[idx]**2 for i, j, k, l in product(*trange)]

def func1(idx):
    return noise_var(w_xx, cov_xx, scaling_array[:, 0], idx)

def func2(idx):
    return noise_var(w_yy, cov_yy, scaling_array[:, 1], idx)

def func3(idx):
    return xterm(w_xx, cov_xx, xps_xx, scaling_array[:, 0], idx)

def func4(idx):
    return xterm(w_yy, cov_yy, xps_yy, scaling_array[:, 1], idx)

norm = Nd**2 / 2 / args.fs**4
print(1)
with Pool(processes=16) as pool:
    var_xx = np.nansum(pool.map(func1, np.arange(len(lst)))) * norm
print(2)
with Pool(processes=16) as pool:
    var_yy = np.nansum(pool.map(func2, np.arange(len(lst)))) * norm
print(3)
with Pool(processes=16) as pool:
    xterm_xx = np.nansum(pool.map(func3, np.arange(len(lst))), axis=(0, 1)) * norm
print(4)
with Pool(processes=16) as pool:
    xterm_yy = np.nansum(pool.map(func4, np.arange(len(lst))), axis=(0, 1)) * norm

# write to file
f = h5py.File(args.xpspath, "a")
if "VAR XX" in f.keys():
    del f["VAR XX"]
if "VAR YY" in f.keys():
    del f["VAR YY"]
if "XTERM XX" in f.keys():
    del f["XTERM XX"]
if "XTERM YY" in f.keys():
    del f["XTERM YY"]
f.create_dataset("VAR XX", data=var_xx)
f.create_dataset("VAR YY", data=var_yy)
f.create_dataset("XTERM XX", data=xterm_xx)
f.create_dataset("XTERM YY", data=xterm_yy)
f.close()
