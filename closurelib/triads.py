""" 
Functions for generating triad lists
"""


import numpy

# default directory to read and write files
root = "/lustre/aoc/projects/hera/pkeller/data/array/"


def antpairs(antpos, bln):
    blns = numpy.array([antpos - a for a in antpos])
    dd = numpy.linalg.norm(blns - bln, axis=-1) 
    return numpy.dstack(numpy.where(numpy.isclose(dd, 0.0, 1e0)))[0].tolist()

def istriad(a1, a2, a3):
    """
    Check if three antenna pairs form a triad
    """
    
    alist = numpy.unique(a1 + a2 + a3)

    return len(alist) == 3 and len(numpy.unique(a1, a2, a3)) == 3


def mktrlist(antpos, blns):
    """
    Return list of equilateral triads of length eqlen
    """

    apairs = [antpairs(antpos, bln) for bln in blns]
    trlist = []
    
    for a1 in apairs[0]:
        for a2 in apairs[1]:
            for a3 in apairs[2]:
                if istriad(a1, a2, a3):
                    tr = numpy.sort(numpy.unique(a1 + a2 + a3)).tolist()
                    trlist.append(tr)

    return numpy.unique(trlist, axis=0).tolist()

def mktrlistF(infile, outfile, blns, returntr=False):
    """
    Read antenna positions from "infile" and save traids to "outfile"
    """
    
    antpos = numpy.loadtxt(infile)
    antpos = numpy.delete(antpos, (2), axis=1)
    trlist = mktrlist(antpos, blns)
    
    if outfile:
        numpy.savetxt(outfile, trlist)
    if not outfile or returntr:
        return trlist

def trlistSub(trlist, antlist):
    """
    Select a subset of triads that contain certain antennas
    """
    
    trkeep = [True] * len(trlist)
    for i,tr in enumerate(trlist):
        for ant in tr:
            if ant not in antlist:
                trkeep[i] = False
                break
    
    return trlist[trkeep]
                
