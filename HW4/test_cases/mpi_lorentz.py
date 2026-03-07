import time
import numpy as np
from mpi4py import MPI


def lorentzian_histogram(n, bins=100, xmin=-10, xmax=10, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    u = rng.random(n)
    x = 1.0 / np.tan(np.pi * u)
    counts, _ = np.histogram(x, bins=bins, range=(xmin, xmax))
    return counts.astype(np.int64)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# settings
n_total = 10_000_000
bins = 100
xmin = -10
xmax = 10
seed = 42

# start timing
if rank == 0:
    start_time = time.time()

# independent RNG stream per rank
ss = np.random.SeedSequence(seed)
child = ss.spawn(size)[rank]
rng = np.random.default_rng(child)

# split work across ranks
chunks = np.full(size, n_total // size, dtype=int)
chunks[: n_total % size] += 1

local = lorentzian_histogram(
    int(chunks[rank]),
    bins=bins,
    xmin=xmin,
    xmax=xmax,
    rng=rng
)

global_counts = np.empty_like(local)
comm.Allreduce(local, global_counts, op=MPI.SUM)

if rank == 0:
    end_time = time.time()
    runtime = end_time - start_time

    print(f"Total samples: {n_total}")
    print(f"Ranks: {size}")
    print(f"Bins: {bins}")
    print(f"Runtime: {runtime:.3f} seconds")
    print(f"Samples per second: {n_total / runtime:.0f}")

    bin_edges = np.linspace(xmin, xmax, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    np.savetxt(
        f"lorentzian_histogram_bins{bins}_ranks{size}.txt",
        np.column_stack([bin_centers, global_counts]),
        fmt="%.6f %d"
    )

    print(f"Results saved to lorentzian_histogram_bins{bins}_ranks{size}.txt")