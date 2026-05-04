import importlib.util
import time
from pathlib import Path

import numpy as np


# This file is inside final_project/parallelization/dask.
THIS_DIR = Path(__file__).resolve().parent

# final_project is two folders up from this file.
PROJECT_DIR = THIS_DIR.parents[1]

# tripleDiff_function.py lives in final_project/tripleDiffLC_optimize.
TRIPLE_DIFF_PATH = PROJECT_DIR / "tripleDiffLC_optimize" / "tripleDiff_function.py"

# Load tripleDiff_function.py directly from its file path.
spec = importlib.util.spec_from_file_location("tripleDiff_function", TRIPLE_DIFF_PATH)
tripleDiff_function = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tripleDiff_function)
getEQU_ISOchange = tripleDiff_function.getEQU_ISOchange


def compute_one_point(X, M_BH):
    # Compute one independent spin-mass grid point.
    dXdt, dMdt = getEQU_ISOchange(X, M_BH)
    return dXdt, dMdt


def compute_grid_dask(X_grid, M_grid):
    # Import Dask here so the file can still be imported if Dask is not installed.
    from dask import delayed, compute

    # Create one delayed task for each independent grid point.
    tasks = []
    for X in X_grid:
        row = []
        for M_BH in M_grid:
            row.append(delayed(compute_one_point)(X, M_BH))
        tasks.append(row)

    # Run all delayed tasks using Dask's process scheduler.
    start = time.perf_counter()
    results = compute(tasks, scheduler="processes")[0]
    elapsed = time.perf_counter() - start

    # Convert list results into two NumPy arrays.
    dXdt_grid = np.array([[value[0] for value in row] for row in results])
    dMdt_grid = np.array([[value[1] for value in row] for row in results])

    return elapsed, dXdt_grid, dMdt_grid


if __name__ == "__main__":
    # Small default grid for a quick smoke test.
    X_grid = np.linspace(0.1, 0.9, 3)
    M_grid = np.logspace(5, 8, 3)

    elapsed, dXdt_grid, dMdt_grid = compute_grid_dask(X_grid, M_grid)

    print("Dask grid run")
    print("-------------")
    print(f"shape: {dXdt_grid.shape}")
    print(f"time:  {elapsed:.6f} seconds")
    print("dXdt:")
    print(dXdt_grid)
    print("dMdt:")
    print(dMdt_grid)
