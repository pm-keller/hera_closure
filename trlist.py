""" 

Pascal M. Keller <pmk46@mrao.cam.ac.uk> 2021/22
Cavendish Astrophysics, University of Cambridge, UK

Make a list of triads of a particular triad class

"""

import sys
sys.path.append("/users/pkeller/code/H1C_IDR3.2/")

import numpy
from closurelib import triads

# equilateral 14 metre and 29 metre triads
triad_list = [((14.6, 0), (-7.3, numpy.sqrt(3) / 2 * 14.6), (-7.3, -numpy.sqrt(3) / 2 * 14.6)),
          ((29.2, 0), (-14.6, numpy.sqrt(3) / 2 * 29.2), (-14.6, -numpy.sqrt(3) / 2 * 29.2))]
trnames = ["EQ14", "EQ29"]

infile = "/lustre/aoc/projects/hera/pkeller/data/array/antpos.dat"

for triad, trname in zip(triad_list, trnames):
    trfile = f"/lustre/aoc/projects/hera/pkeller/data/array/trlist_{trname}.dat"
    trlist = triads.mktrlistF(infile, trfile, triad, returntr=True)