import argparse

import cupy_function as cf


parser = argparse.ArgumentParser()
parser.add_argument("--N", type=int, required=True)
args = parser.parse_args()

for _ in range(args.N):
    cf.getEQU_ISOchange(0.5, 1e6)

if cf.CUPY_AVAILABLE:
    cf.cp.cuda.Stream.null.synchronize()
