import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parent.parent / "CuPy_optimize"))
import cupy_function as cf


def run_one(_):
    return cf.getEQU_ISOchange(0.5, 1e6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--processes", type=int, default=1)
    args = parser.parse_args()

    start = time.perf_counter()
    with mp.Pool(processes=args.processes) as pool:
        pool.map(run_one, range(args.N))
    runtime = time.perf_counter() - start

    print(f"N = {args.N}, processes = {args.processes}, runtime = {runtime:.6f}s")


if __name__ == "__main__":
    main()
