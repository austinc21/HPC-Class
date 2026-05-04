import sys
import time
from pathlib import Path

import numpy as np


# This file lives in final_project/getHughes_optimize.
THIS_DIR = Path(__file__).resolve().parent

# The original edited_functions.py file lives one folder up, in final_project.
PROJECT_DIR = THIS_DIR.parent

# Add final_project first so Python can import the original edited_functions.py.
sys.path.insert(0, str(PROJECT_DIR))

# Add this folder so Python can import sanity_check.py.
sys.path.insert(0, str(THIS_DIR))

import edited_functions
import sanity_check


# Test input for getEQU_ISOchange.
X = 0.5
M_BH = 1e6

# Floating-point tolerances for deciding whether the answers match.
REL_TOL = 1e-8
ABS_TOL = 1e-12


# Run and time the original function from edited_functions.py.
start = time.perf_counter()
original_dXdt, original_dMdt = edited_functions.getEQU_ISOchange(X, M_BH)
original_time = time.perf_counter() - start


# Run and time the version from sanity_check.py.
start = time.perf_counter()
sanity_dXdt, sanity_dMdt = sanity_check.getEQU_ISOchange(X, M_BH)
sanity_time = time.perf_counter() - start


# Compute absolute differences between the two outputs.
dXdt_error = abs(original_dXdt - sanity_dXdt)
dMdt_error = abs(original_dMdt - sanity_dMdt)


# Check whether the outputs are close enough to count as matching.
dXdt_matches = np.isclose(original_dXdt, sanity_dXdt, rtol=REL_TOL, atol=ABS_TOL)
dMdt_matches = np.isclose(original_dMdt, sanity_dMdt, rtol=REL_TOL, atol=ABS_TOL)
overall_match = dXdt_matches and dMdt_matches


print("getEQU_ISOchange sanity check")
print("-----------------------------")
print(f"X:    {X}")
print(f"M_BH: {M_BH}")
print()

print("Runtime")
print("-------")
print(f"edited_functions.py time: {original_time:.6f} seconds")
print(f"sanity_check.py time: {sanity_time:.6f} seconds")
print(f"speedup:              {original_time / sanity_time:.2f}x")
print()

print("Output comparison")
print("-----------------")
print(f"edited_functions dXdt: {original_dXdt:.12e}")
print(f"sanity_check dXdt:    {sanity_dXdt:.12e}")
print(f"dXdt error:           {dXdt_error:.12e}")
print()
print(f"edited_functions dMdt: {original_dMdt:.12e}")
print(f"sanity_check dMdt:    {sanity_dMdt:.12e}")
print(f"dMdt error:           {dMdt_error:.12e}")
print()

print("Match check")
print("-----------")
print(f"relative tolerance:   {REL_TOL:.1e}")
print(f"absolute tolerance:   {ABS_TOL:.1e}")
print(f"dXdt matches:         {dXdt_matches}")
print(f"dMdt matches:         {dMdt_matches}")
print(f"overall match:        {overall_match}")
