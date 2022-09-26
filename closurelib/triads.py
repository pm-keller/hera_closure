""" 

Pascal M. Keller <pmk46@mrao.cam.ac.uk> 2021/22
Cavendish Astrophysics, University of Cambridge, UK

Functions for generating triad lists

"""


import numpy


def antpairs(antpos, bln):
    """
    Args: 
        antpos (list): list of antenna position vectors
        bln (list): list of baseline vectors of a given triad shape

    Returns:
        list: list of antenna pairs which form one of the baselines in bln.
    """

    blns = numpy.array([antpos - a for a in antpos])
    dd = numpy.linalg.norm(blns - bln, axis=-1) 

    return numpy.dstack(numpy.where(numpy.isclose(dd, 0.0, 1e0)))[0].tolist()


def istriad(a1, a2, a3):
    """
    Check if three antenna pairs (a1, a2, and a3) form a triad. 
    The variables a1, a2 and a3 are lists containing two antenna numbers
    e.g. a1 = [0, 1]

    Returns:
        bool: True if antenna pairs form a triad. 

        E.g. a1 = [0, 1], a2 = [1, 2], a3 = [0, 2] returns True
             a1 = [0, 1], a2 = [1, 2], a3 = [2, 0] returns True
             a1 = [0, 0], a2 = [1, 2], a3 = [2, 0] returns False
             a1 = [0, 1], a2 = [0, 1], a3 = [1, 2] returns False
    """
    
    # unique list of antenna numbers
    alist = numpy.unique(a1 + a2 + a3)

    # check if list contains 3 unique antennas
    return len(alist) == 3 and len(numpy.unique(a1, a2, a3)) == 3


def mktrlist(antpos, blns):
    """

    Get list of equilateral triads of a particular shape given by the variable antpos.
    
    Args:
        antpos (list): list of antenna position vectors, e.g. [[0, 0, 1], [0, 1, 0], [1, 0, 0], ...].
        blns (list): list of baseline vectors for a given triad shape, e.g. [[1, 0], [-0.5, 1], [-0.5, -1]]

    Returns:
        numpy array: unique list of triads

    """
    # make sure the specified baselines add to zero
    assert numpy.isclose(numpy.sum(blns), 0, 1e0) & numpy.shape(blns)[0] == 3, "Baseline do not form a triad!"

    # get antenna pairs that qualify for the given triad shape
    apairs = [antpairs(antpos, bln) for bln in blns]
    trlist = []
    
    # baseline 1
    for a1 in apairs[0]:
        # baseline 2
        for a2 in apairs[1]:
            # baseline 3
            for a3 in apairs[2]:
                # Do baselines 1, 2 and 3 form a triad?
                if istriad(a1, a2, a3):
                    tr = numpy.sort(numpy.unique(a1 + a2 + a3)).tolist()
                    trlist.append(tr)

    return numpy.unique(trlist, axis=0).tolist()

def mktrlistF(infile, outfile, blns, returntr=False):
    """

    Same as mktrlist, but read/write from/to file.
    
    Args:
        infile (str): antenna position file
        outfile (str): file to write triad list to
        blns (list): list of baseline vectors for a given triad shape, e.g. [[1, 0], [-0.5, 1], [-0.5, -1]]
        returntr (bool): if True, return triad list

    Returns:
        numpy array: unique list of triads if returntr == True

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

    Args:
        trlist (list): list of triads
        antlist (list): list of antennas to keep

    Returns:
        list: triad list containing only antennas in antlist
    """
    
    trkeep = [True] * len(trlist)
    for i,tr in enumerate(trlist):
        for ant in tr:
            if ant not in antlist:
                trkeep[i] = False
                break
    
    return trlist[trkeep]


def triads_to_baselines(triads):
    """
    Convert a list of triads to a list of unique baselines
    """
    
    bls_pairs = []
    
    triads = np.atleast_2d(triads)
    
    for triad in triads:
        bls_pairs.append([np.sort([triad[i], triad[(i+1)%3]]) for i in range(3)])

    bls_pairs = np.array(bls_pairs).reshape(-1, 2)
    idx = np.sort(np.unique(bls_pairs, return_index=True, axis=0)[1])
    return bls_pairs[idx]