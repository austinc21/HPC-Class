import importlib.util
import os
import time
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent

# Each N is the number of points in a 2D physics grid.
# Override from Slurm/shell with, for example:
#   BENCH_PROBLEM_SIZES=10,100 python benchmark_cupy_sizes.py
PROBLEM_SIZES = [
    int(value)
    for value in os.environ.get("BENCH_PROBLEM_SIZES", "10").split(",")
]


def load_module(module_name, file_path):
    # Import a Python file from an exact path without requiring package setup.
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sync_cupy(module):
    # CuPy launches GPU work asynchronously, so synchronize before stopping
    # the timer. If CuPy is unavailable, this function does nothing.
    if getattr(module, "CUPY_AVAILABLE", False):
        module.cp.cuda.Stream.null.synchronize()


def make_parameter_grid(n_points):
    # Build a 2D grid with exactly n_points total points.
    # X ranges linearly from 0 to 1.
    # M_BH ranges logarithmically from 10^5 to 10^8.
    n_x = int(np.floor(np.sqrt(n_points)))
    n_m = int(np.ceil(n_points / n_x))
    X_values = np.linspace(0.0, 1.0, n_x)
    M_values = np.logspace(5, 8, n_m)
    points = [(X, M_BH) for X in X_values for M_BH in M_values]
    return points[:n_points], n_x, n_m


def run_grid(module, points):
    # Run getEQU_ISOchange once at every point in the 2D grid.
    results = []
    for X, M_BH in points:
        results.append(module.getEQU_ISOchange(X, M_BH))
    return np.asarray(results)


def time_grid(module, points):
    start = time.perf_counter()
    results = run_grid(module, points)
    sync_cupy(module)
    elapsed = time.perf_counter() - start
    return elapsed, results


def main():
    triple_diff = load_module(
        "tripleDiff_function",
        PROJECT_DIR / "tripleDiffLC_optimize" / "tripleDiff_function.py",
    )
    cupy_diff = load_module(
        "cupy_function",
        PROJECT_DIR / "CuPy_optimize" / "cupy_function.py",
    )

    print("getEQU_ISOchange 2D-grid benchmark")
    print("----------------------------------")
    print(f"CuPy available: {cupy_diff.CUPY_AVAILABLE}")
    print("Grid ranges: X in [0, 1], M_BH in [1e5, 1e8]")
    print()
    print(
        f"{'N':>8} "
        f"{'grid':>12} "
        f"{'tripleDiff total (s)':>22} "
        f"{'CuPy total (s)':>16} "
        f"{'speedup':>10} "
        f"{'CuPy/call (s)':>15} "
        f"{'max abs diff':>14}"
    )
    print("-" * 107)

    for n_points in PROBLEM_SIZES:
        points, n_x, n_m = make_parameter_grid(n_points)

        triple_time, triple_results = time_grid(triple_diff, points)
        cupy_time, cupy_results = time_grid(cupy_diff, points)

        speedup = triple_time / cupy_time if cupy_time > 0 else float("inf")
        cupy_per_call = cupy_time / n_points
        max_abs_diff = np.max(np.abs(triple_results - cupy_results))

        print(
            f"{n_points:8d} "
            f"{n_x}x{n_m:<9d} "
            f"{triple_time:22.6f} "
            f"{cupy_time:16.6f} "
            f"{speedup:10.2f} "
            f"{cupy_per_call:15.6f} "
            f"{max_abs_diff:14.6e}"
        )


if __name__ == "__main__":
    main()
