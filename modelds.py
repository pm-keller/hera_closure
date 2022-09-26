""" 

Pascal M. Keller <pmk46@mrao.cam.ac.uk> 2021/22
Cavendish Astrophysics, University of Cambridge, UK

Make model data set

"""

import numpy as np
import pandas as pd
import itertools
import argparse
import h5py

from scipy.interpolate import interp1d
from closurelib.triads import triads_to_baselines

import matplotlib.pyplot as plt

import sys
sys.path.append("/users/pkeller/code/ClosureSim/skysim")
from vis import VisData


parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gleam", help="Path to GLEAM visibilities.", type=str)
parser.add_argument("-b", "--bright", help="Path to model visibilities of bright sources.", type=str)
parser.add_argument("-d", "--data", help="Path to closure data.", type=str)
parser.add_argument("-s", "--std", help="Path noise model.", type=str)


# load noise standard deviation
with h5py.File(args.std) as f:
    lstn = f["LST"][()]
    sig = f["RMS"][()].real   
        
# get gleam visibilities
gleam = VisData()
gleam.read_vis(args.gleam, names=["LST"])

# get visibilities of bright sources
bright = VisData()
bright.read_vis(args.bright)

# add visibilities
vis = gleam.copy()
vis.vis += bright.vis
        
# LST array
lstm = vis.header_data["LST"]
lst_idx = np.where((lstn > np.min(lstm)) & (lstn < np.max(lstm)) & (~np.isnan(sig).all(-1)))
lstn = lstn[lst_idx]
        
# select noise RMS corresponding to model LST array
sig = sig[lst_idx]
sig = interp1d(lstn, sig, axis=0, fill_value="extrapolate")(lstm)
        
with h5py.File(args.data) as f:
    triads = f["triads"][()]
    shape = f["bispec"].shape
        
# create array for model bispectra
bispec = np.empty(shape, dtype=vis.vis.dtype)
        
# array of baseline pairs
bls = triads_to_baselines(triads)
        
# create noise array, where each baseline has its idependent noise
noise_shape = (bls.size,) + vis.vis.shape[1:]
mu = np.zeros(noise_shape, dtype=np.float32)
        
for i in range(shape[1]):
    noise = np.random.normal(mu, sig) + 1j * np.random.normal(mu, sig)
            
    for j, triad in enumerate(triads):
        bls_triad = triads_to_baselines(triad)
                
        # select correct noise components for a given triad
        idx1 = np.where(np.all(bls_triad[0] == bls, axis=-1))[0][0]
        idx2 = np.where(np.all(bls_triad[1] == bls, axis=-1))[0][0]
        idx3 = np.where(np.all(bls_triad[2] == bls, axis=-1))[0][0]

        # compute noisy visibilities
        visn = vis.copy()
        visn.vis = vis.vis + np.array([noise[idx1], noise[idx2], noise[idx3].conjugate()])      
                
        # compute bispectrum
        bispec[:, i, j] = visn.get_bispec()
        
# write to file
with h5py.File(args.data, "a") as f:
    dname = f"bispec model"
    if dname in f.keys():
        del f[dname]
    f.create_dataset(f"bispec model", data=bispec)