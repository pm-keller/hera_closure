""" 
Compute triad-JD metric

This metric helps find closure phase data formed by non-correlating baselines 
on a per-night and per-triad basis.
"""


import numpy as np
import argparse
import h5py

from  multiprocessing import Pool
from closurelib import cptools as cp


parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", help="Path to closure data.", type=str)
args = parser.parse_args()

# load closure phase data
with h5py.File(args.path, "r") as f:
    bispec = f["bispec"][()]
    meicp = f["eicp trmed"][()]
    trlist = f["triads"][()]
    flags = f["JD-LST flags"][()]

# apply flags
idx = np.where(flags)
bispec[:, idx[0], :, idx[1]] = np.nan

eicp = np.exp(1j * np.angle(bispec))
del bispec

def metric(tr):
    """ 
    Triad metric
    """
    #return np.nanmedian(np.abs(np.angle(eicp[:, :, tr] * meicp.conjugate())), axis=(-2, -1)) * 2 / np.pi
    return np.nanmedian(np.abs(eicp[:, :, tr] - meicp), axis=(-2, -1)) 

with Pool(processes=16) as pool:
    z = pool.map(metric, np.arange(len(trlist)))
z = np.moveaxis(z, 0, 2)

# write to HDF5 file
f = h5py.File(args.path, "a")
dname = "triad-JD metric"
if dname in f.keys():
    del f[dname]
f.create_dataset(dname, data=z)
f.close()