import argparse
import importlib.util
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


# This file is inside final_project/parallelization.
THIS_DIR = Path(__file__).resolve().parent

# final_project is one folder up.
PROJECT_DIR = THIS_DIR.parent

# Add the Dask example folder to Python's import path.
sys.path.insert(0, str(THIS_DIR / "dask"))

# Load tripleDiff_function.py directly from its file path.
TRIPLE_DIFF_PATH = PROJECT_DIR / "tripleDiffLC_optimize" / "tripleDiff_function.py"
spec = importlib.util.spec_from_file_location("tripleDiff_function", TRIPLE_DIFF_PATH)
tripleDiff_function = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tripleDiff_function)
getEQU_ISOchange = tripleDiff_function.getEQU_ISOchange


def compute_grid_serial(X_grid, M_grid):
    # Serial baseline: compute every grid point one at a time.
    dXdt_grid = np.empty((len(X_grid), len(M_grid)))
    dMdt_grid = np.empty((len(X_grid), len(M_grid)))

    start = time.perf_counter()
    for i, X in enumerate(X_grid):
        for j, M_BH in enumerate(M_grid):
            dXdt_grid[i, j], dMdt_grid[i, j] = getEQU_ISOchange(X, M_BH)
    elapsed = time.perf_counter() - start

    return elapsed, dXdt_grid, dMdt_grid


def try_dask(X_grid, M_grid):
    # Run the Dask implementation if Dask is installed.
    try:
        from dask_grid import compute_grid_dask
    except ImportError:
        return None

    return compute_grid_dask(X_grid, M_grid)


def try_mpi(x_points, m_points, ranks):
    # Launch the MPI script if mpiexec is available.
    mpiexec = shutil.which("mpiexec") or shutil.which("mpirun")
    if mpiexec is None:
        return None

    output_path = THIS_DIR / "mpi_result.npz"
    script_path = THIS_DIR / "mpi" / "mpi_grid.py"

    command = [
        mpiexec,
        "-n",
        str(ranks),
        sys.executable,
        str(script_path),
        "--x-points",
        str(x_points),
        "--m-points",
        str(m_points),
        "--output",
        str(output_path),
    ]

    subprocess.run(command, check=True)
    return np.load(output_path)


def print_comparison(name, serial_time, serial_dXdt, serial_dMdt, result):
    # Print speed and correctness checks against the serial baseline.
    if result is None:
        print(f"{name}: not available")
        return

    elapsed, dXdt_grid, dMdt_grid = result
    dXdt_error = np.max(np.abs(serial_dXdt - dXdt_grid))
    dMdt_error = np.max(np.abs(serial_dMdt - dMdt_grid))

    print(f"{name}:")
    print(f"  time:        {elapsed:.6f} seconds")
    print(f"  speedup:     {serial_time / elapsed:.2f}x")
    print(f"  max dXdt err:{dXdt_error:.6e}")
    print(f"  max dMdt err:{dMdt_error:.6e}")


def main():
    parser = argparse.ArgumentParser(description="Compare serial, Dask, and MPI grid runs.")
    parser.add_argument("--x-points", type=int, default=3)
    parser.add_argument("--m-points", type=int, default=3)
    parser.add_argument("--mpi-ranks", type=int, default=2)
    args = parser.parse_args()

    # Small default grid; increase points after confirming everything works.
    X_grid = np.linspace(0.1, 0.9, args.x_points)
    M_grid = np.logspace(5, 8, args.m_points)

    serial_time, serial_dXdt, serial_dMdt = compute_grid_serial(X_grid, M_grid)

    print("Parallelization comparison")
    print("--------------------------")
    print(f"grid shape: {serial_dXdt.shape}")
    print(f"serial time: {serial_time:.6f} seconds")
    print()

    dask_result = try_dask(X_grid, M_grid)
    print_comparison("Dask", serial_time, serial_dXdt, serial_dMdt, dask_result)
    print()

    mpi_result = try_mpi(args.x_points, args.m_points, args.mpi_ranks)
    if mpi_result is None:
        print("MPI: not available")
    else:
        print_comparison(
            "MPI",
            serial_time,
            serial_dXdt,
            serial_dMdt,
            (
                float(mpi_result["elapsed"]),
                mpi_result["dXdt_grid"],
                mpi_result["dMdt_grid"],
            ),
        )


if __name__ == "__main__":
    main()
