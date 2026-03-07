import numpy as np
def lorentzian_histogram(n, bins=100, xmin=-10, xmax=10):
    """
    Sample n random points from the Lorentzian distribution
    using inverse transform sampling. Make a histogram with
    the specified bin count and range. Returns counts.
    """
    u = np.random.random(n)  # Uniform(0,1)
    x = 1. / np.tan(np.pi * u)  # x = 1/tan(pi*u)
    counts, _ = np.histogram(x, bins=bins, range=(xmin, xmax))
    return counts  # No need to return bin edges for uniform bins


# dask_lorentz.py
import dask
from dask import delayed

@delayed
def delayed_lorentzian_histogram(n, bins=100, xmin=-10, xmax=10):
    return lorentzian_histogram(n, bins=bins, xmin=xmin, xmax=xmax)

def run_dask(n, n_tasks=4, bins=100, xmin=-10, xmax=10):
    chunks = (n // n_tasks) * np.ones(n_tasks, dtype=int)
    chunks[:n % n_tasks] += 1

    tasks = [
        delayed_lorentzian_histogram(chunk, bins=bins, xmin=xmin, xmax=xmax)
        for chunk in chunks
    ]

    results = dask.compute(*tasks)
    return np.sum(results, axis=0)