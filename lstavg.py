"""
LST averaging
"""

import os
import h5py
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--inpath", help="Path to closure data.", type=str)
parser.add_argument("-o", "--outpath", help="Path to write averaged data to.", type=str)
parser.add_argument("-n", help="Number of neighbouring LST's to average", type=int)
parser.add_argument("-m", "--model", help="use model data.", action="store_true")
args = parser.parse_args()

if args.model:
    ifmodel = " model"
else:
    ifmodel = ""

# load data
with h5py.File(args.inpath, "r") as f:
    jd = f["JD"][()]
    lst = f["LST"][()]
    frq = f["FRQ"][()]
    trflags_xx = f["triad flags XX"][()]
    trflags_yy = f["triad flags YY"][()]
    trlist_xx = f["triads XX"][()][~trflags_xx]
    trlist_yy = f["triads YY"][()][~trflags_yy]
    eicp1_xx = f[f"eicp jdmed (2) XX{ifmodel}"][()][:, ~trflags_xx]
    eicp2_xx = f[f"eicp jdmed (4) XX{ifmodel}"][()][:, ~trflags_xx]
    eicp1_yy = f[f"eicp jdmed (2) YY{ifmodel}"][()][:, ~trflags_yy]
    eicp2_yy = f[f"eicp jdmed (4) YY{ifmodel}"][()][:, ~trflags_yy]


# number of averaged LST integrations
Nlst = len(lst) // args.n

# average complex closure phase
eicp1_xx = np.array([np.nanmean(eicp1_xx[:, :, i*args.n:(i+1)*args.n], axis=2) for i in range(Nlst)])
eicp2_xx = np.array([np.nanmean(eicp2_xx[:, :, i*args.n:(i+1)*args.n], axis=2) for i in range(Nlst)])
eicp1_yy = np.array([np.nanmean(eicp1_yy[:, :, i*args.n:(i+1)*args.n], axis=2) for i in range(Nlst)])
eicp2_yy = np.array([np.nanmean(eicp2_yy[:, :, i*args.n:(i+1)*args.n], axis=2) for i in range(Nlst)])

# average LST time stamps
lst = np.array([np.mean(lst[i*args.n:(i+1)*args.n]) for i in range(Nlst)])

dnames = ["JD", "LST", "FRQ", f"eicp XX (2){ifmodel}", f"eicp XX (4){ifmodel}", f"eicp YY (2){ifmodel}", f"eicp YY (4){ifmodel}", "triads XX", "triads YY"]
data_list = [jd, lst, frq, np.moveaxis(eicp1_xx, 0, 2), np.moveaxis(eicp2_xx, 0, 2), np.moveaxis(eicp1_yy, 0, 2), np.moveaxis(eicp2_yy, 0, 2), trlist_xx, trlist_yy]

f = h5py.File(args.outpath, "a")

for dname, data in zip(dnames, data_list):
    if dname in f.keys():
        del f[dname]
    f.create_dataset(dname, data=data)

f.close()
