#!/lustre/aoc/projects/hera/pkeller/anaconda3/envs/closure_analysis/bin/python3
#SBATCH -p hera
#SBATCH -J mkcl
#SBATCH -o mkcl.out
#SBATCH -t 72:00:00
#SBATCH --mem=128G
#SBATCH --mail-type=ALL
#SBATCH --mail-user pmk46@cam.ac.uk

""" 

Pascal M. Keller <pmk46@mrao.cam.ac.uk> 2021/22
Cavendish Astrophysics, University of Cambridge, UK

Make closure phase or bispectrum data sets.

"""

import sys

sys.path.append("/users/pkeller/code/H1C_IDR3.2/")

import os
import glob
import pandas
import numpy as np
from closurelib import heradata
from closurelib import triads

# time delta [s]
delta = 10.7668166 / 3600

# list of HERA fields
lstdata_list = [{"start": 21.5, "stop": 0.0, "delta": delta, "name": "FA"},
                {"start": 0.75, "stop": 2.75, "delta": delta, "name": "FB"},
                {"start": 4.0, "stop": 6.25, "delta": delta, "name": "FC"},
                {"start": 6.25, "stop": 9.25, "delta": delta, "name": "FD"},
                {"start": 9.25, "stop": 14.75, "delta": delta, "name": "FE"}]

# list of triad classes
triad_list = ["EQ14", "EQ28"]

# select triad class and HERA field
triad = triad_list[1]
lstdata = lstdata_list[2]

# generate a name
name = f"{triad}_{lstdata['name']}"

# specify i/o directories 
outdir = "/lustre/aoc/projects/hera/pkeller/data/H1C_IDR3.2/"
stagedir = os.path.join(outdir, "tmp")
alignfname = os.path.join(outdir, "alignment", f"h1c_idr3_{lstdata['name']}.npz")

# generate a triad list
antfile = f"/lustre/aoc/projects/hera/pkeller/data/array/antlist_h1c_{lstdata['name']}.dat"
trfile = f"/lustre/aoc/projects/hera/pkeller/data/array/trlist_{triad}.dat"
trlist = np.loadtxt(trfile).astype(int)
antlist = np.loadtxt(antfile)
trlist = triads.trlistSub(trlist, antlist)

# prepare closure phase data file
datafname = os.path.join(outdir, "sample", f"{name}.h5")

if not os.path.isfile(datafname):
    heradata.prepClosureDS(alignfname, trlist, datafname, qname="bispec")

# add aligned closure phases to data file
fg = np.load(alignfname)["fnameg"]
print("align, shape: ", fg.shape)

for j in range(0, fg.shape[0]):
    with open("/users/pkeller/code/H1C_IDR3.2/printfile.out", "w") as f:
        print(f"processing day {j} of {fg.shape[0]}\n")
        f.write(f"processing day {j} of {fg.shape[0]}\n")

    for k in range(fg.shape[1]):
        heradata.addBispecDS(alignfname, outdir, datafname, trlist, j, k)

    for f in glob.iglob(r"/lustre/aoc/projects/hera/pkeller/data/H1C_IDR3.2/tmp/*", recursive=True):
        if os.path.isfile(f):
            os.remove(f)
            
