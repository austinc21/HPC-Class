import numpy as np
from numba import njit, prange


def lorentzian_histogram(n, bins=100, xmin=-10, xmax=10):
    u = np.random.random(n)
    x = 1.0 / np.tan(np.pi * u)
    counts, _ = np.histogram(x, bins=bins, range=(xmin, xmax))
    return counts


@njit(parallel=True, nogil=True)
def lorentzian_histogram_numba(n, bins=100, xmin=-10, xmax=10, n_chunks=4):
    xfac = bins / (xmax - xmin)

    local_counts = np.zeros((n_chunks, bins), dtype=np.int64)

    base = n // n_chunks
    rem = n % n_chunks

    for c in prange(n_chunks):
        m = base
        if c < rem:
            m += 1

        for _ in range(m):
            u = np.random.random()
            x = 1.0 / np.tan(np.pi * u)
            ix = int((x - xmin) * xfac)

            if 0 <= ix < bins:
                local_counts[c, ix] += 1

    counts = np.zeros(bins, dtype=np.int64)

    for c in range(n_chunks):
        for b in range(bins):
            counts[b] += local_counts[c, b]

    return counts