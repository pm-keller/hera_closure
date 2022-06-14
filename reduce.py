"""
Initial Data Reduction
"""


import os
import h5py
import argparse
import numpy as np

from closurelib import cptools as cp


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--inpath", help="Path to closure data.", type=str)
parser.add_argument(
    "-o", "--outpath", help="Path to write reduced closure data to.", type=str
)
parser.add_argument("--fmin", help="Minimum Frequency (MHz).", type=float)
parser.add_argument("--fmax", help="Maximum Frequency (MHz).", type=float)
args = parser.parse_args()


# generate frequency array
fmin, fmax, fN = (100, 200, 1024)
frq = np.linspace(fmin, fmax, fN)

# load data
with h5py.File(args.inpath, "r") as f:
    jd = f["JDax"][()]
    lst = f["LSTAx"][()]
    trlist = f["triads"][()]

# shape of reduced data set
Nlst = len(lst)

if os.path.exists(args.outpath):
    os.remove(args.outpath)
fw = h5py.File(args.outpath, "a")

for t in range(Nlst):
    with h5py.File(args.inpath, "r") as f:
        bispec = f["bispec"][:, t]
        flags = f["flags"][:, t]

    # update flags
    flags = flags | np.isnan(bispec)
    bispec[flags] = np.nan

    # select subband and rearrange axes in order (polarisation, JD, triad, LST, Frequency)
    bispec, sb = cp.select(bispec, frq, args.fmin, args.fmax, axis=2)
    bispec = np.moveaxis(bispec, (0, 1, 2, 3), (1, 2, 3, 0))

    # remove nan-only slices
    bispec, jd_idx = cp.remove_nan_slices(bispec, axis=1, return_index=True)
    bispec, tr_idx = cp.remove_nan_slices(bispec, axis=2, return_index=True)

    # add lst axis
    bispec = bispec[:, :, :, np.newaxis]

    if t == 0:
        jd = jd[jd_idx]
        trlist = trlist[tr_idx]
        shape = (2, len(jd), len(trlist), Nlst, len(sb))

        fw.create_dataset("JD", data=jd)
        fw.create_dataset("LST", data=lst)
        fw.create_dataset("FRQ", data=sb)
        fw.create_dataset("triads", data=trlist)
        fw.create_dataset("bispec", data=bispec, chunks=True, maxshape=shape)
    else:
        fw["bispec"].resize((fw["bispec"].shape[3] + 1), axis=3)
        fw["bispec"][:, :, :, -1:] = bispec

fw.close()
