#!/lustre/aoc/projects/hera/pkeller/anaconda3/envs/closure_analysis/bin/python3
#SBATCH -J cp_spec
#SBATCH -o cp_spec.out
#SBATCH -t 12:00:00
#SBATCH --mem=128G
#SBATCH --mail-type=ALL
#SBATCH --mail-user pmk46@cam.ac.uk

"""
Initial Data Reduction
"""

import os
import h5py
import numpy as np

from multiprocessing import Pool

import sys

sys.path.append("/users/pkeller/code/H1C_IDR3.2/")

from closurelib import cptools as cp


# data paths
path = "/lustre/aoc/projects/hera/pkeller/data/H1C_IDR3.2/sample/EQ28_FA.h5"
flpath = "/lustre/aoc/projects/hera/pkeller/data/H1C_IDR3.2/sample/EQ28_FA_B2.h5"
outpath = "/lustre/aoc/projects/hera/pkeller/data/H1C_IDR3.2/sample/EQ28_FA_SPEC.h5"

# generate frequency array
fmin, fmax, fN = (100, 200, 1024)
frq = np.linspace(fmin, fmax, fN)


# load data
with h5py.File(flpath, "r") as f:
    flags = f["flags"][()]

with h5py.File(path, "r") as f:
    jd = f["JDax"][()]
    lst = f["LSTAx"][()]
    trlist = f["triads"][()]

if os.path.exists(outpath):
    os.remove(outpath)
fw = h5py.File(outpath, "a")

with h5py.File(path, "r") as f:
    bispec = f["bispec"][:, 0]


def mkspec(t):
    with h5py.File(path, "r") as f:
        bispec = f["bispec"][:, t]

    eicp = np.exp(1j * np.angle(bispec))
    del bispec

    # put in order (polarisation, JD, triad, LST, Frequency)
    eicp = np.moveaxis(eicp, (0, 1, 2, 3), (1, 2, 3, 0))

    # flag data
    eicp[flags] = np.nan

    # remove all-nan slices
    eicp = cp.remove_nan_slices(eicp, axis=1)
    eicp = cp.remove_nan_slices(eicp, axis=2)

    # average JD
    eicp = cp.geomed(eicp, axis=1)

    # average over triads
    eicp = np.nanmean(eicp, axis=1)
    
    return eicp

with Pool(processes=16) as pool:
    meicp = pool.map(mkspec, np.arange(len(lst)))

meicp = np.moveaxis(meicp, 0, -2)

fw.create_dataset("eicp", data=meicp)
fw.create_dataset("LST", data=lst)
fw.close()
