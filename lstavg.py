""" 

Pascal M. Keller <pmk46@mrao.cam.ac.uk> 2021/22
Cavendish Astrophysics, University of Cambridge, UK

LST averaging

"""

import os
import h5py
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--inpath", help="Path to closure data.", type=str)
parser.add_argument("-o", "--outpath", help="Path to write averaged data to.", type=str)
parser.add_argument(
    "-s", "--scalingpath", help="File with scaling coefficients", type=str
)
parser.add_argument(
    "-v", "--veffpath", help="File with effective visibilities", type=str
)
parser.add_argument("-n", help="Number of neighbouring LST's to average", type=int)
parser.add_argument("-m", "--model", help="Use model data.", action="store_true")
parser.add_argument("-f", "--flags", help="Apply LST flags.", type=str, default="None")

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
    trlist_xx = f["triads XX"][()]
    trlist_yy = f["triads YY"][()]
    eicp1_xx = f[f"eicp jdmed (2) XX{ifmodel}"][()]
    eicp2_xx = f[f"eicp jdmed (4) XX{ifmodel}"][()]
    eicp1_yy = f[f"eicp jdmed (2) YY{ifmodel}"][()]
    eicp2_yy = f[f"eicp jdmed (4) YY{ifmodel}"][()]

# load scaling coefficients
scaling_array = np.loadtxt(args.scalingpath)

# flag Veff > 5
veff = np.loadtxt(args.veffpath)
idx = np.where(veff < 5)
print(len(idx[1]), len(veff.T))
scaling_array[idx] *= np.nan

# manual flags
if args.flags != "None" and os.path.exists(args.flags):
    lst_flags = np.atleast_2d(np.loadtxt(args.flags))

    for lst_range in lst_flags:
        lst_min, lst_max = lst_range
        idx = np.where((lst > lst_min) & (lst < lst_max))
        print(len(idx[0]), scaling_array.shape[1])
        scaling_array[:, idx] *= np.nan

for i, A in enumerate(scaling_array.T):
    eicp1_xx[:, :, i] *= np.sqrt(A[0])
    eicp1_yy[:, :, i] *= np.sqrt(A[1])
    eicp2_xx[:, :, i] *= np.sqrt(A[0])
    eicp2_yy[:, :, i] *= np.sqrt(A[1])

# number of averaged LST integrations
Nlst = len(lst) // args.n

# average complex closure phase
eicp1_xx = np.array(
    [
        np.nanmean(eicp1_xx[:, :, i * args.n : (i + 1) * args.n], axis=2)
        for i in range(Nlst)
    ]
)
eicp2_xx = np.array(
    [
        np.nanmean(eicp2_xx[:, :, i * args.n : (i + 1) * args.n], axis=2)
        for i in range(Nlst)
    ]
)
eicp1_yy = np.array(
    [
        np.nanmean(eicp1_yy[:, :, i * args.n : (i + 1) * args.n], axis=2)
        for i in range(Nlst)
    ]
)
eicp2_yy = np.array(
    [
        np.nanmean(eicp2_yy[:, :, i * args.n : (i + 1) * args.n], axis=2)
        for i in range(Nlst)
    ]
)

# average LST time stamps
lst = np.array([np.mean(lst[i * args.n : (i + 1) * args.n]) for i in range(Nlst)])

dnames = [
    "JD",
    "LST",
    "FRQ",
    f"eicp XX (2){ifmodel}",
    f"eicp XX (4){ifmodel}",
    f"eicp YY (2){ifmodel}",
    f"eicp YY (4){ifmodel}",
    "triads XX",
    "triads YY",
]
data_list = [
    jd,
    lst,
    frq,
    np.moveaxis(eicp1_xx, 0, 2),
    np.moveaxis(eicp2_xx, 0, 2),
    np.moveaxis(eicp1_yy, 0, 2),
    np.moveaxis(eicp2_yy, 0, 2),
    trlist_xx,
    trlist_yy,
]

f = h5py.File(args.outpath, "a")

for dname, data in zip(dnames, data_list):
    if dname in f.keys():
        del f[dname]
    f.create_dataset(dname, data=data)

f.close()
