# Bojan Nikolic <b.nikolic@mrao.cam.ac.uk> 2016,2017,2018 (as part of
# "heracasa" package), 2020 (as part of "closureps2")

"""
Closure quantities from visibility data
"""

import numpy

def rewrap(p):
    """
    Rewrap phase so that branch cut is along the negative real axis.
    """
    return numpy.arctan2(numpy.sin(p), numpy.cos(p))


def eitherWay(a1, a2, i, j):
    """
    Return rows where a1==i and a2==j OR a1==j and a2==i. Also return
    sign +1 if former or -1 if latter.
    """
    r1 = numpy.logical_and(a1 == i, a2 == j).nonzero()[0]
    if r1.shape[0]:
        return r1[0], +1.0
    else:
        r2 = numpy.logical_and(a1 == j, a2 == i).nonzero()[0]
        if r2.shape[0]:
            return r2[0], -1.0
        else:
            return 0, 0


def triadRows(a1, a2, tr):
    """
    Rows corresponding to single triad tr
    """
    i, j, k = tr
    p1, s1 = eitherWay(a1, a2, i, j)
    p2, s2 = eitherWay(a1, a2, j, k)
    p3, s3 = eitherWay(a1, a2, k, i)
    return ((p1, p2, p3), (s1, s2, s3))


def quadRows(a1, a2, qd):
    """
    Rows corresponding to a single quad qd
    """
    i, j, k, l = qd
    a12, s1 = eitherWay(a1, a2, i, j)
    a34, s2 = eitherWay(a1, a2, k, l)
    a13, s3 = eitherWay(a1, a2, i, k)
    a24, s4 = eitherWay(a1, a2, j, l)
    return (a12, a34, a13, a24)


def triads(a1, a2, alist):
    """
    List the rows corresponding to all triads in a list

    :param a1, a2: Arrays with antenna IDs for first and second antenna
    :param alist:  List of antenna IDs for which to generate the triads

    :returns: Tuple of (list of (tuple containing rows with data in the triad)),
              (list of tuples contianing antenna IDs in the triad),
              (list of signs to be used in computing a closure phase)

    """
    if len(alist) < 3:
        raise "Need at least three antennas to generate triads"
    nant = len(alist)
    rows = []
    tr = []
    signs = []
    for ni, i in enumerate(alist[:-2]):
        for nj, j in enumerate(alist[ni + 1 : -1]):
            for nk, k in enumerate(alist[ni + nj + 2 :]):
                ((p1, p2, p3), (s1, s2, s3)) = triadRows(a1, a2, (i, j, k))
                rows.append((p1, p2, p3))
                tr.append((i, j, k))
                signs.append((s1, s2, s3))
    return rows, tr, signs


def triadsList(a1, a2, trlist):
    """
    List the rows corresponding to specified triads

    :param a1, a2: Arrays with antenna IDs for first and second antenna

    :returns: Tuple of (list of (tuple containing rows with data in the triad)),
              (list of tuples contianing antenna IDs in the triad),
              (list of signs to be used in computing a closure phase)

    """
    rows = []
    trres = []
    signs = []
    for tr in trlist:
        ((p1, p2, p3), (s1, s2, s3)) = triadRows(a1, a2, tr)
        rows.append((p1, p2, p3))
        trres.append(tr)
        signs.append((s1, s2, s3))
    return rows, trres, signs


def closurePh(f, time, trlist=None):
    """The closure phase on a triad of antennas

    :param f: h5py file

    :param time: time slot to read

    :param trlist: Explicit triads to compute the closure; if given
    only these triads will be returned

    :returns: Dictionary: "phase": array containing phases; "tr": an
    array with the triad ids; "flags": array containing flags ,
    row-synchronous with phase; "JD": an array containing julian day of each integration

    """
    Nbls = f["Header/Nbls"][()]
    i1 = time * Nbls
    i2 = (time + 1) * Nbls

    a1 = f["Header/ant_1_array"][i1:i2]
    a2 = f["Header/ant_2_array"][i1:i2]

    ants = f["Header/antenna_numbers"]

    if trlist is not None:
        rows, tr, signs = triadsList(a1, a2, trlist)
    else:
        rows, tr, signs = triads(a1, a2, ants)
    ph = numpy.angle(f["Data/visdata"][i1:i2])
    fl = f["Data/flags"][i1:i2]

    jd = numpy.unique(f["Header/time_array"][i1:i2])
    last = numpy.unique(f["Header/lst_array"][i1:i2])
    if len(last) > 1:
        raise RuntimeError("Inonsistent LST in integration")

    clp, flags = [], []
    for (p1, p2, p3), (s1, s2, s3) in zip(rows, signs):
        if s1 == 0 or s2 == 0 or s3 == 0:
            phnan = numpy.nan * numpy.ones(numpy.shape(ph)[1:])
            flnan = numpy.ones(numpy.shape(flags)[1:]).astype(bool)
            clp.append(phnan)
            flags.append(flnan)
        else:
            clp.append(rewrap(ph[p1] * s1 + ph[p2] * s2 + ph[p3] * s3))
            flags.append(numpy.logical_or(fl[p1], fl[p2], fl[p3]))
        
    return {
        "closure phase": numpy.array(clp),
        "flags": numpy.array(flags),
        "tr": numpy.array(tr),
        "JD": jd,
        "LST": last,
    }


def bispectrum(vis1, vis2, vis3, s1, s2, s3):
    """
    Return bispectrum of three visibilities
    """
    
    v1 = vis1.copy()
    v2 = vis2.copy()
    v3 = vis3.copy()
    
    if s1 == -1:
        v1 = v1.conjugate()
    if s2 == -1:
        v2 = v2.conjugate()
    if s3 == -1:
        v3 = v3.conjugate()
        
    return v1 * v2 * v3


def bispec(f, time, trlist=None):
    """The bispectrum on a triad of antennas

    :param f: h5py file

    :param time: time slot to read

    :param trlist: Explicit triads to compute the closure; if given
    only these triads will be returned

    :returns: Dictionary: "bisp": array containing bispectrum; "tr": an
    array with the triad ids; "flags": array containing flags ,
    row-synchronous with phase; "JD": an array containing julian day of each integration

    """
    Nbls = f["Header/Nbls"][()]
    i1 = time * Nbls
    i2 = (time + 1) * Nbls

    a1 = f["Header/ant_1_array"][i1:i2]
    a2 = f["Header/ant_2_array"][i1:i2]
    
    ants = f["Header/antenna_numbers"]

    if trlist is not None:
        rows, tr, signs = triadsList(a1, a2, trlist)
    else:
        rows, tr, signs = triads(a1, a2, ants)
        
    vis = f["Data/visdata"][i1:i2]
    fl = f["Data/flags"][i1:i2]

    jd = numpy.unique(f["Header/time_array"][i1:i2])
    last = numpy.unique(f["Header/lst_array"][i1:i2])
    if len(last) > 1:
        raise Exception("Inonsistent LST in integration")


    bisp, flags = [], []
    for (p1, p2, p3), (s1, s2, s3) in zip(rows, signs):
        if s1 == 0 or s2 == 0 or s3 == 0:
            bsnan = numpy.nan * numpy.ones(numpy.shape(vis)[1:])
            flnan = numpy.ones(numpy.shape(flags)[1:]).astype(bool)
            bisp.append(bsnan)
            flags.append(flnan)
        else:
            bisp.append(bispectrum(vis[p1], vis[p2], vis[p3], s1, s2, s3))
            flags.append(numpy.logical_or(fl[p1], fl[p2], fl[p3]))
        
    return {
        "bispec": numpy.array(bisp),
        "flags": numpy.array(flags),
        "tr": numpy.array(tr),
        "JD": jd,
        "LST": last,
    }