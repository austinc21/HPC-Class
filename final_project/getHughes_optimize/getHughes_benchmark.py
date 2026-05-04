import sys
import time
from pathlib import Path

import numpy as np


# This file is in final_project/getHughes_optimize.
THIS_DIR = Path(__file__).resolve().parent

# edited_functions.py is one folder up, in final_project.
PROJECT_DIR = THIS_DIR.parent

# Add final_project to the import path so Python can find edited_functions.py.
sys.path.insert(0, str(PROJECT_DIR))

# Add this folder to the import path so Python can find newFunction.py.
sys.path.insert(0, str(THIS_DIR))

import edited_functions
import newFunction


# Pick one input pair to test.
X = 0.5
M_BH = 1e6

# Tolerances for deciding whether the two answers are close enough.
REL_TOL = 1e-8
ABS_TOL = 1e-12


# Time the original getEQU_ISOchange from edited_functions.py.
start = time.perf_counter()
original_dXdt, original_dMdt = edited_functions.getEQU_ISOchange(X, M_BH)
original_time = time.perf_counter() - start


# Time the new getEQU_ISOchange from newFunction.py.
start = time.perf_counter()
new_dXdt, new_dMdt = newFunction.getEQU_ISOchange(X, M_BH)
new_time = time.perf_counter() - start


# Measure how different the new version is from the original version.
dXdt_error = abs(original_dXdt - new_dXdt)
dMdt_error = abs(original_dMdt - new_dMdt)

# Check whether the new function returns the same values as the original.
dXdt_matches = np.isclose(original_dXdt, new_dXdt, rtol=REL_TOL, atol=ABS_TOL)
dMdt_matches = np.isclose(original_dMdt, new_dMdt, rtol=REL_TOL, atol=ABS_TOL)
outputs_match = dXdt_matches and dMdt_matches


# Print timing results.
print("getEQU_ISOchange benchmark")
print("--------------------------")
print(f"X:    {X}")
print(f"M_BH: {M_BH}")
print(f"original time: {original_time:.6f} seconds")
print(f"new time:      {new_time:.6f} seconds")
print(f"speedup:       {original_time / new_time:.2f}x")


# Print output and accuracy results.
print()
print("Output comparison")
print("-----------------")
print(f"original dXdt: {original_dXdt:.12e}")
print(f"new dXdt:      {new_dXdt:.12e}")
print(f"dXdt error:    {dXdt_error:.12e}")
print()
print(f"original dMdt: {original_dMdt:.12e}")
print(f"new dMdt:      {new_dMdt:.12e}")
print(f"dMdt error:    {dMdt_error:.12e}")

print()
print("Match check")
print("-----------")
print(f"relative tolerance: {REL_TOL:.1e}")
print(f"absolute tolerance: {ABS_TOL:.1e}")
print(f"dXdt matches: {dXdt_matches}")
print(f"dMdt matches: {dMdt_matches}")
print(f"overall match: {outputs_match}")
