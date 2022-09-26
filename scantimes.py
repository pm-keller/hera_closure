#!/lustre/aoc/projects/hera/pkeller/anaconda3/bin/python
#SBATCH -j scantimes
#SBATCH --mem-per-cpu=32G
#SBATCH --mail-type=ALL
#SBATCH -o scantimes.out
#SBATCH -l walltime=48:00:00

""" 

Pascal M. Keller <pmk46@mrao.cam.ac.uk> 2021/22
Cavendish Astrophysics, University of Cambridge, UK

Scan all visibility files and save metadata to file.

"""

import sys

sys.path.append("/users/pkeller/code/H1C_IDR3.2/")

import os
import pandas
import numpy as np
from closurelib import heradata

jdrange = (2458026, 2458208)
jdlist = np.arange(*jdrange)
lstdata = {"start": 0.0, "stop": 24, "delta": 10.7668166 / 3600}
lstrange = (lstdata["start"], lstdata["stop"])
outdir = "/lustre/aoc/projects/hera/pkeller/data/H1C_IDR3.2/"

for i, jd in enumerate(jdlist):
    with open("/users/pkeller/code/H1C_IDR3.2/printfile.out", "a") as f:
        f.write(f"processing Julian Day {jd}, {i} of {len(jdlist)}\n")

    d = heradata.scanTimeLib((jd, jd + 1), lstrange, outdir)
    d.to_csv(os.path.join(outdir, "scans", f"{str(jd)}_scan.csv"))