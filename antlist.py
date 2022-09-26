""" 

Pascal M. Keller <pmk46@mrao.cam.ac.uk> 2021/22
Cavendish Astrophysics, University of Cambridge, UK

Generate a list of antennas which can be used for the analysis of a field

"""

import sys
sys.path.append("/users/pkeller/code/H1C_IDR3.2/")

import numpy as np
import h5py
import os
import shutil

from pyuvdata import UVData

from closurelib import libtools as librarian

delta = 10.7668166 / 3600
lstdata_list = [{"start": 21.5, "stop": 0.0, "delta": delta, "name": "FA"},
                {"start": 0.75, "stop": 2.75, "delta": delta, "name": "FB"},
                {"start": 4.0, "stop": 6.25, "delta": delta, "name": "FC"},
                {"start": 6.25, "stop": 9.25, "delta": delta, "name": "FD"},
                {"start": 9.25, "stop": 14.75, "delta": delta, "name": "FE"}]

indir = "/lustre/aoc/projects/hera/pkeller/data/H1C_IDR3.2/alignment/"
outdir = "/lustre/aoc/projects/hera/pkeller/data/array/"
stagedir = "/lustre/aoc/projects/hera/pkeller/data/H1C_IDR3.2/tmp/"

for lstdata in lstdata_list:
    alignfname = os.path.join(indir, f"h1c_idr3_{lstdata['name']}.npz")
    fg = np.load(alignfname)["fnameg"].astype(str)

    success, fpath = librarian.stageFile(fg[0, 0][:-2], stagedir)
    UV = UVData()
    UV.read(fpath)
    a = UV.ant_1_array
    shutil.rmtree(fpath)

    np.savetxt(os.path.join(outdir, f"antlist_h1c_{lstdata['name']}.dat"), a)
