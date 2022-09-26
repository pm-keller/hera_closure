""" 

Pascal M. Keller <pmk46@mrao.cam.ac.uk> 2021/22
Cavendish Astrophysics, University of Cambridge, UK

Create flags based on nan-values in data

"""

import numpy as np
import argparse
import h5py


parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", help="Path to closure data.", type=str)
args = parser.parse_args()

nanflags = []

# load data
with h5py.File(args.path, "r") as f:
    shape = f["bispec"].shape

for i in range(shape[0]):
    for j in range(shape[1]):
        with h5py.File(args.path, "r") as f:
            nanflags.append(np.all(np.isnan(f["bispec"][i]), axis=(-2, -1)))

nanflags = np.array(nanflags)

# write to file
f = h5py.File(args.path, "a")

dname = f"nan-flags"
if dname in f.keys():
    del f[dname]
f.create_dataset(dname, data=nanflags)
    
f.close()