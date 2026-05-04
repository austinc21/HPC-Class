import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
from mpi4py import MPI


# This file is inside final_project/parallelization/mpi.
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


def compute_one_point(task):
    # Each task is one independent grid point: (i, j, X, M_BH).
    i, j, X, M_BH = task
    dXdt, dMdt = getEQU_ISOchange(X, M_BH)
    return i, j, dXdt, dMdt


def split_tasks(tasks, size):
    # Give each MPI rank a roughly equal slice of the task list.
    return [tasks[rank::size] for rank in range(size)]


def main():
    parser = argparse.ArgumentParser(description="MPI spin-mass grid example.")
    parser.add_argument("--x-points", type=int, default=3)
    parser.add_argument("--m-points", type=int, default=3)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        X_grid = np.linspace(0.1, 0.9, args.x_points)
        M_grid = np.logspace(5, 8, args.m_points)

        # Flatten the 2D grid into independent tasks.
        tasks = []
        for i, X in enumerate(X_grid):
            for j, M_BH in enumerate(M_grid):
                tasks.append((i, j, X, M_BH))

        task_chunks = split_tasks(tasks, size)
        start = time.perf_counter()
    else:
        X_grid = None
        M_grid = None
        task_chunks = None
        start = None

    # Scatter task chunks so each rank gets different grid points.
    local_tasks = comm.scatter(task_chunks, root=0)

    # Each rank computes its assigned points.
    local_results = [compute_one_point(task) for task in local_tasks]

    # Gather all results back to rank 0.
    gathered_results = comm.gather(local_results, root=0)

    if rank == 0:
        elapsed = time.perf_counter() - start
        dXdt_grid = np.empty((args.x_points, args.m_points))
        dMdt_grid = np.empty((args.x_points, args.m_points))

        for rank_results in gathered_results:
            for i, j, dXdt, dMdt in rank_results:
                dXdt_grid[i, j] = dXdt
                dMdt_grid[i, j] = dMdt

        print("MPI grid run")
        print("------------")
        print(f"ranks: {size}")
        print(f"shape: {dXdt_grid.shape}")
        print(f"time:  {elapsed:.6f} seconds")

        if args.output:
            np.savez(
                args.output,
                X_grid=X_grid,
                M_grid=M_grid,
                dXdt_grid=dXdt_grid,
                dMdt_grid=dMdt_grid,
                elapsed=elapsed,
                ranks=size,
            )


if __name__ == "__main__":
    main()
