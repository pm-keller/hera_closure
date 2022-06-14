#!/lustre/aoc/projects/hera/pkeller/anaconda3/bin/python
#SBATCH -j xps
#SBATCH --mem-per-cpu=32G
#SBATCH --mail-type=ALL
#SBATCH -o xps.out
#SBATCH -l walltime=1:00:00

"""
Compute cross-spectral densities
"""

import os
import h5py
import numpy as np

import sys
sys.path.append("/users/pkeller/code/ClosurePhaseAnalysis/")

from library import dspec

# data path
inpath = "/lustre/aoc/projects/hera/pkeller/data/H1C_IDR3.2/sample/EQ14_FC_B2_AVG.h5"
outpath = "/lustre/aoc/projects/hera/pkeller/data/H1C_IDR3.2/sample/EQ14_FC_B2_XPS.h5"

# load data
with h5py.File(inpath, "r") as f:
    frq = f["FRQ"][()]
    lst = f["LST"][()]
    eicp_xx = f["eicp XX 1"][()]
    eicp_yy = f["eicp YY 1"][()]

# sums
eicp_xx = (eicp_xx[0] + eicp_xx[1]) / 2
eicp_yy = (eicp_yy[0] + eicp_yy[1]) / 2

# cross spectral densities between triads in different JD bins
xps_xx = dspec.xps(eicp_xx, eicp_xx, fs=10.24*1e-6)[1]
xps_yy = dspec.xps(eicp_yy, eicp_yy, fs=10.24*1e-6)[1]

# save to HDF5-file
f = h5py.File(outpath, "a")

for xps, pol in zip([xps_xx, xps_yy], ["XX", "YY"]):
    dname = f"XPS intra-JD {pol}"
    if dname in f.keys():
        del f[dname]
    f.create_dataset(dname, data=xps)

f.close()
