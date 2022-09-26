""" 

Pascal M. Keller <pmk46@mrao.cam.ac.uk> 2021/22
Cavendish Astrophysics, University of Cambridge, UK

Tools for working with the HERA-librarian

"""

import os
import re
import time
import shutil
import hera_librarian as librarian


def getFileNames(jdrange, lstrange, namematch="zen.%.uvh5"):
    """query file names

    Args:
        jdrange (list): range of julian days to query [JD_MIN, JD_MAX]
        lstramge (list): LST range to query [LST_MIN, LST_MAX]
        namematch (str): additional constraints on file names. Default to "zen.%.sum.uvh5".

    Returns:
        list of file names (str)
    """

    cl = librarian.LibrarianClient(conn_name="local")

    query = """
        {{
            "name-matches": "{:s}",
            "start-time-jd-greater-than": {:d},
            "stop-time-jd-less-than": {:d},
            "start-lst-hr-in-range": [{:f}, {:f}]
        }}
    """.format(
        namematch, jdrange[0], jdrange[1], lstrange[0], lstrange[1]
    )

    r = cl.search_files(query)["results"]
    fnames = [f["name"] for f in r]

    return fnames


def stageFile(fname, stagedir):
    """stage file from librarian to lustre

    Args:
        fname (str): name of file to query
        stagedir (str): directory to stage files to

    Returns:
        bool, str: success status and path to staged file
    """

    jd = re.findall(r"\d+", fname)[0]
    fpath = os.path.join(stagedir, str(jd), fname)
    marker = os.path.join(stagedir, "STAGING-SUCCEEDED")

    if os.path.exists(fpath) and fpath.split(".")[-1] == "uv":
        shutil.rmtree(fpath)
    elif os.path.exists(fpath) and fpath.split(".")[-1] == "uvh5":
        os.remove(fpath)
    if os.path.exists(marker):
        os.remove(marker)

    cl = librarian.LibrarianClient(conn_name="local")
    stage = cl.launch_local_disk_stage_operation(
        user="pkeller",
        search='{{"name-matches": "{:s}"}}'.format(fname),
        dest_dir=stagedir,
    )

    while not os.path.exists(marker):
        time.sleep(1)

    return stage["success"], fpath
