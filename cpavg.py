""" 
Average all closure phase spectrograms using inverse variance weights
"""


import h5py
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--inpath", help="Path to closure data.", type=str)
parser.add_argument("-o", "--outpath", help="Path to save averaged data to.", type=str)
parser.add_argument("-s", "--scalingpath", help="Path to scaling constants.", type=str)
parser.add_argument("-m", "--model", help="use model data.", action="store_true")
args = parser.parse_args()


# use model data?
if args.model:
    ifmodel = " model"
else:
    ifmodel = ""

# load data
with h5py.File(args.inpath, "r") as f:
    eicp_xx = f[f"eicp XX (2){ifmodel}"][()]
    eicp_yy = f[f"eicp YY (2){ifmodel}"][()]
    freq = f[f"FRQ"][()]
    lst = f[f"LST"][()]

# load scaling coefficients
scaling_array = np.loadtxt(args.scalingpath)

# apply scaling
for i, A in enumerate(scaling_array):
    eicp_xx[:, :, i] *= np.sqrt(A[0])
    eicp_yy[:, :, i] *= np.sqrt(A[1])

# compute noise realisation
noise_xx = (eicp_xx[1] - eicp_xx[0]) / np.sqrt(2)
noise_yy = (eicp_yy[1] - eicp_yy[0]) / np.sqrt(2)

# variance along frequency axis
ivar_xx = 1 / np.nanvar(noise_xx, axis=-1)
ivar_yy = 1 / np.nanvar(noise_yy, axis=-1)

# averag over triads
eicp_xx_travg = np.nansum(np.moveaxis(eicp_xx, -1, 0) * ivar_xx, axis=(1, 2))
eicp_yy_travg = np.nansum(np.moveaxis(eicp_yy, -1, 0) * ivar_yy, axis=(1, 2))

# average over triads and times
eicp_xx_avg = np.nansum(np.moveaxis(eicp_xx, -1, 0) * ivar_xx, axis=(1, 2, 3))
eicp_yy_avg = np.nansum(np.moveaxis(eicp_yy, -1, 0) * ivar_yy, axis=(1, 2, 3))

# average polarisations
eicp_travg = (eicp_xx_travg + eicp_yy_travg) / (np.nansum(ivar_xx) + np.nansum(ivar_yy, axis=0)) 
eicp_avg = (eicp_xx_avg + eicp_yy_avg) / (np.nansum(ivar_xx) + np.nansum(ivar_yy, axis=(0, 1))) 

# save to file
dnames = ["eicp travg", "eicp avg", "FRQ", "LST"]
data_list = [eicp_travg, eicp_avg, freq, lst]

f = h5py.File(args.outpath, "a")

for dname, data in zip(dnames, data_list):
    if dname in f.keys():
        del f[dname]
    f.create_dataset(dname, data=data)

f.close()