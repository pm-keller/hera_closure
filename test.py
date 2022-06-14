""" 
Compute geometric median of bispectrum over triads
"""


import numpy as np
import concurrent.futures
import time

from closurelib import cptools as cp

eicp = np.ones((100, 100, 1000))
shape = eicp.shape[:2] + eicp.shape[3:]
Njd = 100

# geometric median over triads
def geomed(jd_index):
    return cp.geomed(eicp[:, jd_index], axis=1)


t0 = time.perf_counter()
with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
    results = [executor.submit(geomed, jd_index) for jd_index in range(Njd)]
    meicp = np.moveaxis(
        [f.result() for f in concurrent.futures.as_completed(results)], 0, 1
    )
t1 = time.perf_counter()


print(t1 - t0)
