#!/lustre/aoc/projects/hera/pkeller/anaconda3/bin/python
#SBATCH -J xps_cov
#SBATCH -o xps_cov.out
#SBATCH -t 1:00:00
#SBATCH --mem=32G
#SBATCH --mail-type=ALL
#SBATCH --mail-user pmk46@cam.ac.uk

"""
Compute Triad Covariances
"""

import os
import h5py
import numpy as np
from itertools import product

import sys
sys.path.append("/users/pkeller/code/ClosurePhaseAnalysis/")

from library import dspec

# data path
path = "/lustre/aoc/projects/hera/pkeller/data/H1C_IDR3.2/sample/EQ14_FC_B2_AVG.h5"
xps_path = "/lustre/aoc/projects/hera/pkeller/data/H1C_IDR3.2/sample/EQ14_FC_B2_XPS.h5"

# load covariance data
with h5py.File(path, "r") as f:
    frq = f["FRQ"][()]
    lst = f["LST"][()]
    cov_xx = f["cov XX"][()]
    cov_yy = f["cov YY"][()]

# load cross-power spectral data
with h5py.File(xps_path, "r") as f:
    xps_xx = f["XPS intra-JD XX"][()]
    xps_yy = f["XPS intra-JD YY"][()]

Nlst = len(lst)
Ntrx = xps_xx.shape[0]
Ntry = xps_yy.shape[0]
Ndx = xps_xx.shape[-1]
Ndy = xps_yy.shape[-1]

w_xx = np.zeros((Ntrx, Ntrx, Nlst, Ndx), dtype=complex)
w_yy = np.zeros((Ntry, Ntry, Nlst, Ndy), dtype=complex)

cov_xx_inv = np.array([np.linalg.inv(cov_xx[i]) for i in range(Nlst)])
cov_yy_inv = np.array([np.linalg.inv(cov_yy[i]) for i in range(Nlst)])

w_cov_xx = np.sum(cov_xx_inv, axis=1)
w_cov_yy = np.sum(cov_yy_inv, axis=1)

fs = 10.24 * 1e-6
cx = fs**2 / Ndx
cy = fs**2 / Ndy

for t, d, i, j in product(range(Nlst), range(Ndx), range(Ntrx), range(Ntrx)):    
    w_xx[i, j, t, d] = ((w_cov_xx[t, i] * w_cov_xx[t, j])**-1 + (cov_xx[t, i, i] * (xps_xx[j, j, t, d] * cx - cov_xx[t, j, j])) + (cov_xx[t, j, j] * (xps_xx[i, i, t, d] * cx - cov_xx[t, i, i])))**-1

for t, d, i, j in product(range(Nlst), range(Ndy), range(Ntry), range(Ntry)): 
    w_yy[i, j, t, d] = ((w_cov_yy[t, i] * w_cov_yy[t, j])**-1 + (cov_yy[t, i, i] * (xps_yy[j, j, t, d] * cy - cov_yy[t, j, j])) + (cov_yy[t, j, j] * (xps_yy[i, i, t, d] * cy - cov_yy[t, i, i])))**-1

# save to HDF5-file
f = h5py.File(xps_path, "a")

for cov, pol in zip([w_xx, w_yy], ["XX", "YY"]):
    dname = f"weights cov {pol}"
    if dname in f.keys():
        del f[dname]
    f.create_dataset(dname, data=cov)

f.close()