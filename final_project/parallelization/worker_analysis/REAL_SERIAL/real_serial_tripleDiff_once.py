import argparse
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[3] / "tripleDiffLC_optimize"))
import tripleDiff_function as tf


parser = argparse.ArgumentParser()
parser.add_argument("--N", type=int, required=True)
args = parser.parse_args()

for _ in range(args.N):
    tf.getEQU_ISOchange(0.5, 1e6)
