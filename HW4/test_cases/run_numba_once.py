import argparse
import time
from numba import set_num_threads
from numba_lorentz import lorentzian_histogram_numba

parser = argparse.ArgumentParser()
parser.add_argument("--workers", type=int, required=True)
parser.add_argument("--N", type=int, required=True)
parser.add_argument("--bins", type=int, default=100)
args = parser.parse_args()

set_num_threads(args.workers)

# warmup
lorentzian_histogram_numba(1000, bins=args.bins, n_chunks=1)

start = time.perf_counter()
lorentzian_histogram_numba(args.N, bins=args.bins, n_chunks=args.workers)
runtime = time.perf_counter() - start

print(f"Runtime: {runtime}")