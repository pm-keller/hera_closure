#!/lustre/aoc/projects/hera/pkeller/anaconda3/bin/python
#SBATCH -j xps
#SBATCH --mem-per-cpu=32G
#SBATCH --mail-type=ALL
#SBATCH -o xps.out
#SBATCH -l walltime=1:00:00

"""
Compute Triad Covariances
"""

import h5py
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", help="Path to closure data.", type=str)
args = parser.parse_args()

# load data
with h5py.File(args.path, "r") as f:
    frq = f["FRQ"][()]
    lst = f["LST"][()]
    eicp_xx = f["eicp XX (2)"][()]
    eicp_yy = f["eicp YY (2)"][()]
    eicp2_xx = f["eicp XX (4)"][()]
    eicp2_yy = f["eicp YY (2)"][()]

# flag NaN-triads
tr_flags_xx = np.isnan(eicp2_xx).any(axis=(0, 2, 3))
eicp_xx = eicp_xx[:, ~tr_flags_xx]
eicp2_xx = eicp2_xx[:, ~tr_flags_xx]

tr_flags_yy = np.isnan(eicp2_yy).any(axis=(0, 2, 3))
eicp_yy = eicp_yy[:, ~tr_flags_yy]
eicp2_yy = eicp2_yy[:, ~tr_flags_yy]

# flag galactic plane
glx_flags = (lst > 6.25) & (lst < 10.75)
eicp_xx = eicp_xx[:, :, ~glx_flags]
eicp_yy = eicp_yy[:, :, ~glx_flags]
Nlst = len(lst[~glx_flags])

# compute closure phase differences
noise_xx = (eicp_xx[1] - eicp_xx[0]) / 2
noise_yy = (eicp_yy[1] - eicp_yy[0]) / 2

# compute covariance matrices
cov_xx = np.array([np.cov(noise_xx[:, i]) for i in range(Nlst)])
cov_yy = np.array([np.cov(noise_yy[:, i]) for i in range(Nlst)])

# save to HDF5-file
f = h5py.File(args.path, "a")

for cov, pol in zip([cov_xx, cov_yy], ["XX", "YY"]):
    dname = f"cov {pol}"
    if dname in f.keys():
        del f[dname]
    f.create_dataset(dname, data=cov)

f.close()