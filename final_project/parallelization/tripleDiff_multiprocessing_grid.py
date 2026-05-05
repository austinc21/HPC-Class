import argparse
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np


sys.path.append(str(Path(__file__).resolve().parent.parent / "tripleDiffLC_optimize"))
import tripleDiff_function as tf


def make_points(n):
    n_x = max(1, int(math.floor(math.sqrt(n))))
    n_m = int(math.ceil(n / n_x))
    x_values = np.linspace(0.0, 1.0, n_x)
    m_values = np.logspace(5, 8, n_m)
    points = [(x, m_bh) for x in x_values for m_bh in m_values]
    return points[:n]


def run_one(point):
    x, m_bh = point
    return tf.getEQU_ISOchange(x, m_bh)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--processes", type=int, default=1)
    args = parser.parse_args()

    points = make_points(args.N)

    start = time.perf_counter()
    with mp.Pool(processes=args.processes) as pool:
        pool.map(run_one, points)
    runtime = time.perf_counter() - start

    print(f"N = {args.N}, processes = {args.processes}, runtime = {runtime:.6f}s")


if __name__ == "__main__":
    main()
