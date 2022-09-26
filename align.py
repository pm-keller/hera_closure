#!/lustre/aoc/projects/hera/pkeller/anaconda3/bin/python
#SBATCH -j align
#SBATCH --mem-per-cpu=32G
#SBATCH --mail-type=ALL
#SBATCH -o align.out
#SBATCH -l walltime=1:00:00

""" 

Pascal M. Keller <pmk46@mrao.cam.ac.uk> 2021/22
Cavendish Astrophysics, University of Cambridge, UK

LST alignment

"""

import sys
sys.path.append("/users/pkeller/code/H1C_IDR3.2/")

import os
import pandas
from closurelib import heradata


indir = "/lustre/aoc/projects/hera/pkeller/data/H1C_IDR3.2/scans/"
outdir = "/lustre/aoc/projects/hera/pkeller/data/H1C_IDR3.2/alignment/"
lstdata = {"start": 4.0, "stop": 6.25, "delta": 10.7668166, "name": "FC"}

d = pandas.read_csv(os.path.join(indir, "h1c_idr3_scan.csv"))
alignfname = os.path.join(outdir, f"h1c_idr3_{lstdata['name']}.npz")
heradata.nearestTimeF(alignfname, d, [lstdata["start"], lstdata["stop"]-lstdata["delta"]/3600], lstdata["delta"])