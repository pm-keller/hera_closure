""" 

Pascal M. Keller <pmk46@mrao.cam.ac.uk> 2021/22
Cavendish Astrophysics, University of Cambridge, UK

Compute geometric median of complex closure phase over triads

"""


import numpy as np
import argparse
import h5py

from multiprocessing import Pool
from closurelib import cptools as cp


parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", help="Path to closure data.", type=str)
parser.add_argument("-a", help="Retain amplitudes when averaging.", action="store_true", default=False)
args = parser.parse_args()

# load bispectrum data
with h5py.File(args.path, "r") as f:
    lst = f["LST"][()]
    bispec = f["bispec"][()]
    Nlst = len(lst)

if not args.a:
    # complex exponential of closure phase
    dname = "eicp trmed"
    data = np.exp(1j * np.angle(bispec))
    del bispec
else:
    dname = "bispec trmed"
    data = bispec.copy()
    del bispec

shape = data.shape[:2] + data.shape[3:]

# geometric median over triads
def geomed(idx):
    return cp.geomed(data[:, :, :, idx], axis=2)


with Pool(processes=8) as pool:
    mdata = pool.map(geomed, np.arange(Nlst))
mdata = np.moveaxis(mdata, 0, 2)

# write to HDF5 file
f = h5py.File(args.path, "a")
if dname in f.keys():
    del f[dname]
f.create_dataset(dname, data=mdata)
f.close()
