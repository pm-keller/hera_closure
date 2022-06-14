#!/lustre/aoc/projects/hera/pkeller/anaconda3/bin/python
#SBATCH -j xrfi
#SBATCH --mem-per-cpu=128G
#SBATCH --mail-type=ALL
#SBATCH -o xrfi.out
#SBATCH -l walltime=12:00:00

"""
RFI flagging
"""

import os
import h5py
import numpy as np

import sys
sys.path.append("/users/pkeller/code/ClosurePhaseAnalysis/")

from library import rfi
from library import cptools as cp

# data path
path = "/lustre/aoc/projects/hera/pkeller/data/H1C_IDR3.2/sample/EQ14_FC_B2.h5"

# load data
with h5py.File(path, "r") as f:
    jd = f["JD"][()]
    lst = f["LST"][()]
    frq = f["FRQ"][()]
    bispec = f["bispec"][()]
    flags = f["flags"][()]
    
# apply flags
bispec[flags] = np.nan

# number of time integrations for RFI flagging
N = 160

# mean over polarisations and triads
bispec = np.nanmean(bispec, axis=(0, 2))

# JD median
mbispec = cp.geomed(bispec, axis=0)

f = h5py.File(path, "a")
if "bispec RFI" in f.keys():
    del f["bispec RFI"]
f.create_dataset("bispec RFI", data=bispec)
f.close()

flags = np.zeros_like(bispec).astype(bool)

# RFI flagging
for i in range(bispec.shape[0]):
    print(f"processing Julian Day {jd[i]}, {i} of {len(jd)}\n")
    with open("/users/pkeller/code/H1C_IDR3.2/printfile.out", "a") as f:
        f.write(f"processing Julian Day {jd[i]}, {i} of {len(jd)}\n")
    
    
    if np.all(np.isnan(bispec[i])):
        flags[i] = True
        
    else:
        # median filtering 1
        newfl = True
        while(newfl):
            res = np.abs(np.angle(bispec * mbispec.conjugate()))
            z = res / (1.4826 * np.nanmedian(res, axis=0))
            idx = np.where((z[i] > 4) | ((z[i] > 2) & rfi.nbr_idx_2D(z[i]>4)))
                 
            if len(idx[0]) > 0:
                bispec[i][idx] = np.nan
                flags[i][idx] = True                    
            else:
                newfl = False
        
        # median filtering 2
        flags[i] += rfi.median_clip_2D(bispec[i])
        
        # polynomial filter
        for j in range(bispec.shape[1] // (N//2) + 1):
            s = slice(j*N//2, j*N//2+N)
            flags[i, s] += rfi.poly_2D_clip(bispec[i, s], zth=3, zth_adj=2)
        
      
        # broadband flags 1 (flag count)
        flags[i] = rfi.broadband_flags(flags[i], n=0.2, n_adj=0.1)

# write flags to file
f = h5py.File(path, "a")
if "RFI" in f.keys():
    del f["RFI"]
f.create_dataset("RFI", data=flags)
f.close()
