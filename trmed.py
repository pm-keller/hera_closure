""" 
Compute geometric median of complex closure phase over triads
"""


import numpy as np
import argparse
import h5py

from multiprocessing import Pool
from closurelib import cptools as cp


parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", help="Path to closure data.", type=str)
args = parser.parse_args()

# load bispectrum data
with h5py.File(args.path, "r") as f:
    lst = f["LST"][()]
    bispec = f["bispec"][()]
    Nlst = len(lst)

# complex exponential of closure phase
eicp = np.exp(1j * np.angle(bispec))
del bispec

shape = eicp.shape[:2] + eicp.shape[3:]

# geometric median over triads
def geomed(idx):
    return cp.geomed(eicp[:, :, :, idx], axis=2)


with Pool(processes=16) as pool:
    meicp = pool.map(geomed, np.arange(Nlst))
meicp = np.moveaxis(meicp, 0, 2)

# write to HDF5 file
f = h5py.File(args.path, "a")
dname = "eicp trmed"
if dname in f.keys():
    del f[dname]
f.create_dataset(dname, data=meicp)
f.close()
