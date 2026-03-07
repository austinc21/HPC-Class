import os
import time
import pandas as pd
from numba import set_num_threads

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from async_lorentz import run_async
from dask_lorentz import run_dask
from joblib_lorentz import run_joblib
from mp_lorentz import run_multiproc
from mpire_lorentz import run_mpire
from numba_lorentz import lorentzian_histogram, lorentzian_histogram_numba
from ppe_lorentz import run_ppe
from thread_lorentz import run_threaded


def run_numba(n, p, bins, xmin=-10, xmax=10):
    set_num_threads(p)
    return lorentzian_histogram_numba(n, bins=bins, xmin=xmin, xmax=xmax, n_chunks=p)


methods = {
    "baseline": lambda n, p, bins: lorentzian_histogram(n, bins=bins),
    "threading": lambda n, p, bins: run_threaded(n, p, bins=bins),
    "multiprocessing": lambda n, p, bins: run_multiproc(n, p, bins=bins),
    "processpool": lambda n, p, bins: run_ppe(n, p, bins=bins),
    "asyncio": lambda n, p, bins: run_async(n, p, bins=bins),
    "dask": lambda n, p, bins: run_dask(n, p, bins=bins),
    "numba": lambda n, p, bins: run_numba(n, p, bins=bins),
    "joblib": lambda n, p, bins: run_joblib(n, p, bins=bins),
    "mpire": lambda n, p, bins: run_mpire(n, p, bins=bins),
}


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    N = 10_000_000
    workers = [1, 2, 4, 8]
    bins_list = [10, 100, 1000]

    results = []

    # warm up numba once
    lorentzian_histogram_numba(1000, bins=100, n_chunks=1)

    print(f"Running benchmark with N = {N}")

    for bins in bins_list:
        print(f"\nTesting bins = {bins}")

        for name, func in methods.items():
            for p in workers:
                start = time.perf_counter()
                func(N, p, bins)
                runtime = time.perf_counter() - start

                results.append({
                    "method": name,
                    "workers": p,
                    "bins": bins,
                    "N": N,
                    "runtime": runtime
                })

                print(f"{name:15s} workers={p:<2d} bins={bins:<4d} runtime={runtime:.6f}")

    df = pd.DataFrame(results)
    df.to_csv("results/runtime.csv", index=False)
    print("\nSaved to results/runtime.csv")
    print(df)


    