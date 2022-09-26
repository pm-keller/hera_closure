# Bojan Nikolic <b.nikolic@mrao.cam.ac.uk> 2020
# Pascal M. Keller <pmk46@mrao.cam.ac.uk> 2021/22

"""
Handling of HERA datasets in uvh5 and similar format
"""

import numpy
import pandas
import h5py
import glob
import shutil
import os
import re

from astropy.time import Time
from pyuvdata import UVData

from . import clquants
from . import libtools as librarian


def jd_to_lst(jd, lat=-30.72138329631366, lon=21.428305555555557):
    """Convert julian day to apparent local sidereal time
    defaults are HERA coordinates

    Args:
        jd (float): julian day
        lat (float, optional): latitude. Defaults to -30.72138329631366.
        lon (float, optional): longitude. Defaults to 21.428305555555557.

    Returns:
        float: local sidereal time
    """

    t = Time(jd, format="jd", location=(lon, lat))

    return t.sidereal_time("apparent").to_value()


def LSTgrid(lstdata):
    """Generate an evenly spaced LST grid

    Args:
        lstdata (dict): {"start", "stop", "delta"} in hours

    Returns:
        array: LST grid
    """

    if lstdata["start"] < lstdata["stop"]:
        grid = numpy.arange(lstdata["start"], lstdata["stop"], lstdata["delta"])
    else:
        g1 = numpy.arange(lstdata["start"], 24, lstdata["delta"])
        g2 = numpy.arange(
            lstdata["delta"] - (24 - g1[-1]), lstdata["stop"], lstdata["delta"]
        )
        grid = numpy.hstack([g1, g2])
    return grid


def filesToJD(fnames, unique=True):
    """Extract julian days from a list of file names

    Args:
        fnames (list or str): file name or list of file names
        unique (bool): if True, return a unique and sorted list of julian days

    Returns:
        list: sorted list of julian days
    """

    if type(fnames) is str:
        jd = re.findall(r"\d+", fnames)[0]
    else:
        jd = [re.findall(r"\d+", fname)[0] for fname in fnames]

    if unique:
        return numpy.sort(numpy.unique(jd)).tolist()
    else:
        return jd
    

def filesToLST(fnames):
    """Extract local sideral times from a list of file names of LST-binned data

    Args:
        fnames (list or str): file name or list of file names

    Returns:
        list: sorted list of local sidereal times
    """

    if type(fnames) is str:
        lst = float(re.findall(r"\d+\.\d+", fnames)[0])
    else:
        lst = [float(re.findall(r"\d+\.\d+", fname)[0]) for fname in fnames]

    return numpy.array(lst) * 12 / numpy.pi


def frqFromFile(fpath):
    """Extract frequency array from a data file

    Args:
        fpath (str): path to data file

    Returns:
        array: frequencies
    """

    with h5py.File(fpath, "r") as fin:
        frq = fin["Header/frequency_array"][()]

    return frq


def getDataLST(root, ant1, ant2, lst, filename="*"):
    """Get LST-binned visibility data, given two antennas and an LST
    
    Args:
        root (str): path to LST-binned data
        ant1 (int): antenna number 1
        ant2 (int): antenna number 2
        lst (float): local sidereal time
        filename (str): filenames to search for
        
    Returns:
        numpy array: LST-binned visibilities of two antennas at the given LST.
    """
    
    # get file names
    fnames = numpy.array(glob.glob(root + "/" + filename))
    
    # find file containing LST
    lst_list = numpy.array(filesToLST(fnames))
    idx = numpy.where((lst_list - lst) <= 0)[0]
    fname = fnames[idx][numpy.argmax(lst_list[idx])]
        
    # read data 
    uv = UVData()
    uv.read(fname)
    uv_data = uv.get_data(ant1, ant2)
    
    
    lst_array = numpy.unique(uv.lst_array * 12 / numpy.pi)
    idx = numpy.argmin(numpy.abs(lst_array - lst))
    
    return uv_data[idx]


def scanTimeLib(jdrange, lstrange, outdir, namematch="zen.%.xx.HH.uv", jdex=[]):
    """Scan librarian for data files of a given JD and LST range and extract metadata

    Args:
        jdrange (tuple): (min, max) julian day
        lstrange (tuple): (min, max) local sidereal time
        outdir (str): directory to write temporary data to
        namematch (str, optional): file name must match this string. Defaults to "zen.*.sum.uvh5"
        jdex (list, optional): julian days to exclude

    Raises:
        Exception: Failed to stage file from librarian

    Returns:
        DataFrame: metadata
    """

    fname = []
    jd = []
    lsts = []

    fnames = librarian.getFileNames(jdrange, lstrange, namematch)
    stagedir = os.path.join(outdir, "tmp")
    if not os.path.exists(stagedir):
        os.mkdir(stagedir)

    for i, fn in enumerate(fnames):
        with open("./printfile.out", "a") as f:
            f.write(f"processing file {i} of {len(fnames)}\n")

        success, fpath = librarian.stageFile(fn, stagedir)
        if not success:
            raise Exception(f"Failed to stage file {fn} from librarian!")
        
        if os.path.exists(fpath):
            UV = UVData()
            UV.read(fpath)
            ll = numpy.unique(UV.lst_array)
            fname += ([str(fpath)] * len(ll))
            jd += ([numpy.floor(UV.time_array).astype(int)[0]] * len(ll))
            lsts += list(ll * 24 / (2 *numpy.pi))

            shutil.rmtree(fpath)

    return pandas.DataFrame({"fname": fname, "jd": jd, "lsts": lsts})


def nearestTime(d, lstrange, delta=10.7668166):
    """Find **nearest** observation for a grid of LSTs

    Args:
        d (DataFrame): metadata as returned by scanTimeLib()
        lstrange (dict): [LST min, LST max] in hours

    Returns:
        Grid with filename containing closest integration; grid of integration lst, jd axis and lst grid axis
    """
    lg = numpy.arange(lstrange[0], lstrange[1], delta/3600)
    jg = d["jd"].unique()
    df = d.loc[(d["lsts"] > lstrange[0]-delta/3600) & (d["lsts"] < lstrange[1]+delta/3600)]
    dsize = [len(df.loc[df["jd"] == jd]) for jd in jg]
    jg = jg[numpy.where(dsize >= numpy.max(dsize)-1)]
    dt = numpy.array(d["fname"], dtype=numpy.string_).dtype
    res = numpy.zeros(shape=(len(jg), len(lg)), dtype=dt)
    dt = numpy.zeros(shape=(len(jg), len(lg)), dtype=numpy.double)
    
    for i, jd in enumerate(jg):
        dd = d[d.jd == jd].reset_index(drop=True)
        
        for k, lst in enumerate(lg):            
            mm = (dd.lsts-lst).abs().idxmin()
            
            if (dd.lsts-lst).abs().loc[mm] <= delta / 2:
                res[i,k] = dd.loc[mm].fname
                dt[i,k] = dd.loc[mm].lsts
            else:
                raise("LST discontinuity!") 
            
    return res, dt, jg, lg


def nearestTimeF(fnameout, *args):
    """Like nearestTime but save output to a npz file

    Args:
        fnameout (str): path to npz file
    """

    fg, lstg, jdaxis, lstaxis = nearestTime(*args)
    numpy.savez(fnameout, fnameg=fg, lstg=lstg, jdaxis=jdaxis, lstaxis=lstaxis)


def prepClosureDS(timesnpz, trlist, fnameout, qname="phase"):
    """Prepare a HDF5 dataset for writing closure quantity
    information. Separate prepare stage allows easy incremental
    filling in

    Args:
        timesnpz (dict):  information on LST/JD grid and nearest dataset for each point (see nearestTimeF)
        trlist (list): triads
        fnameout (str): path to HDF5 file
        qname (str): name of closure quantity
    """

    f = numpy.load(timesnpz)

    with h5py.File(fnameout, "w") as fout:
        gridshape = f["fnameg"].shape + (len(trlist), 1024, 2)
        fout.create_dataset(qname, shape=gridshape, dtype=numpy.complex64)
        fout.create_dataset("flags", shape=gridshape, dtype=numpy.bool)
        fout.create_dataset("triads", data=trlist)
        fout.create_dataset("JDax", data=f["jdaxis"])
        fout.create_dataset("LSTAx", data=f["lstaxis"])

        
def mkh5(fname, stagedir):
    """Make UVH5 files
    
    Args:
        fname: generic file name
        stagedir: directory to stage files to
      
    Returns: 
        name of UVH5 file
    """
    
    fxx = fname[:-2]
    fyy = fname[:18] + "yy" + fname[20:-2]
    fpath = os.path.join(stagedir, fname[:18] + fname[21:])
    
    success, fpathxx = librarian.stageFile(fxx, stagedir)
    success, fpathyy = librarian.stageFile(fyy, stagedir)
    
    # set XX polarisation product to East-West
    uv = UVData()
    uv.read([fpathxx, fpathyy], axis="polarization")
    uv.x_orientation = 'east'
    uv.history += '\n\nx_orientation manually set to east\n\n'
    uv.write_uvh5(fpath, clobber=True)
    
    shutil.rmtree(fpathxx)
    shutil.rmtree(fpathyy)


def addClosureDS(timesnpz, outdir, fnameout, trlist, jdi, lsti):
    """Fill in closure quantity information for JD index jdi and lst index lsti

    Args:
        timesnpz (str): npz file containing information on LST/JD grid and nearest dataset for each point (see nearestTimeF)
        outdir (str): directory to write temporary data to
        fnameout (str): path to HDF5 file
        trlist (list): triads
        jdi (int): JD index
        lsti (int): LST index
    """
    f = numpy.load(timesnpz)

    stagedir = os.path.join(outdir, "tmp")
    if not os.path.exists(stagedir):
        os.mkdir(stagedir)

    with h5py.File(fnameout, "a") as fout:
        fname = f["fnameg"][jdi, lsti].decode('utf-8')
        fpath = os.path.join(stagedir, fname[:18] + fname[21:])
        if not os.path.exists(fpath):
            mkh5(fname, stagedir)
        ff=h5py.File(fpath, "r")
        fflst=numpy.unique(ff["Header/lst_array"]) * 24 / (2 * numpy.pi)
        ftime=numpy.abs(fflst-fout["LSTAx"][lsti]).argmin()
        r=clquants.closurePh(ff, ftime, trlist=trlist)
        fout["phase"][jdi, lsti]=r["phase"][:, 0, :, :]
        fout["flags"][jdi, lsti]=r["flags"][:, 0, :, :]    

        
def addBispecDS(timesnpz, outdir, fnameout, trlist, jdi, lsti):
    """Fill in bisprectrum information for JD index jdi and lst index lsti

    Args:
        timesnpz (dict): information on LST/JD grid and nearest dataset for each point (see nearestTimeF)
        outdir (str): directory to write temporary data to
        fnameout (str): path to HDF5 file
        trlist (list): triads
        jdi (int): JD index
        lsti (int): LST index
    """

    f = numpy.load(timesnpz)

    stagedir = os.path.join(outdir, "tmp")
    if not os.path.exists(stagedir):
        os.mkdir(stagedir)

    with h5py.File(fnameout, "a") as fout:
        fname = f["fnameg"][jdi, lsti].decode('utf-8')
        fpath = os.path.join(stagedir, fname[:18] + fname[21:])
        if not os.path.exists(fpath):
            mkh5(fname, stagedir)
        ff=h5py.File(fpath, "r")
        fflst=numpy.unique(ff["Header/lst_array"]) * 24 / (2 * numpy.pi)
        ftime=numpy.abs(fflst-fout["LSTAx"][lsti]).argmin()
        r=clquants.bispec(ff, ftime, trlist=trlist)
        fout["bispec"][jdi, lsti]=r["bispec"][:, 0, :, :]
        fout["flags"][jdi, lsti]=r["flags"][:, 0, :, :]        
        
        
def addTrList(fnameout, trlist):
    """
    Insert triad information
    """

    with h5py.File(fnameout, "a") as fout:
        fout["triads"]=numpy.array(trlist)
        
        
def prllSubset(fnamein, fnameout, qname="phase"):
    """Extract only parallel polarisation products

    Args:
        fnamein (str): path to closure data file
        fnameout (str): path to wirte new data to
        qname (str): name of closure quantity
    """

    fin = h5py.File(fnamein, "r")

    with h5py.File(fnameout, "w") as fout:
        fout.create_dataset(qname, data=fin[qname][..., 0:2])

        for ff in ["flags", "JD", "LST"]:
            fout.create_dataset(ff, data=fin[ff])


def visFormat(fnamein, fnameout, qname="phase"):
    """Re-orient for best visualisation using napari

    Args:
        fnamein (str): path to closure data file
        fnameout (str): path to wirte new data to
        qname (str): name of closure quantity
    """

    fin = h5py.File(fnamein, "r")

    with h5py.File(fnameout, "w") as fout:
        z = fin[qname].shape
        fout.create_dataset(qname, shape=(z[4], z[0], z[2], z[1], z[3]))

        for day in range(z[0]):
            d = numpy.moveaxis(fin[qname][day], 3, 0)
            d = numpy.moveaxis(d, 2, 1)
            fout[qname][:, day, :, :, :] = d + numpy.pi

        for ff in ["flags", "JD", "LST"]:
            fout.create_dataset(ff, data=fin[ff])
