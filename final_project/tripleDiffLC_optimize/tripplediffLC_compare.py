import importlib.util
import time
from pathlib import Path

import numpy as np


# This file is inside final_project/tripleDiffLC_optimize.
THIS_DIR = Path(__file__).resolve().parent

# final_project is one folder up from tripleDiffLC_optimize.
PROJECT_DIR = THIS_DIR.parent


def load_module(module_name, file_path):
    # Load a Python file as a module from an exact file path.
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def time_equ_iso(module, X, M_BH):
    # Run getEQU_ISOchange once and return runtime plus output values.
    start = time.perf_counter()
    dXdt, dMdt = module.getEQU_ISOchange(X, M_BH)
    elapsed = time.perf_counter() - start
    return elapsed, dXdt, dMdt


# Load the three versions we want to compare.
original = load_module("edited_functions", PROJECT_DIR / "edited_functions.py")
hughes = load_module("hughes_newFunction", PROJECT_DIR / "getHughes_optimize" / "newFunction.py")
triple_diff = load_module(
    "tripleDiff_function",
    PROJECT_DIR / "tripleDiffLC_optimize" / "tripleDiff_function.py",
)


# Pick one input pair to test.
X = 0.5
M_BH = 1e6

# Tolerances for deciding whether answers match the original.
REL_TOL = 1e-8
ABS_TOL = 1e-12


# Run all three versions.
original_time, original_dXdt, original_dMdt = time_equ_iso(original, X, M_BH)
hughes_time, hughes_dXdt, hughes_dMdt = time_equ_iso(hughes, X, M_BH)
triple_time, triple_dXdt, triple_dMdt = time_equ_iso(triple_diff, X, M_BH)


# Compare Hughes version to original.
hughes_dXdt_error = abs(original_dXdt - hughes_dXdt)
hughes_dMdt_error = abs(original_dMdt - hughes_dMdt)
hughes_dXdt_match = np.isclose(original_dXdt, hughes_dXdt, rtol=REL_TOL, atol=ABS_TOL)
hughes_dMdt_match = np.isclose(original_dMdt, hughes_dMdt, rtol=REL_TOL, atol=ABS_TOL)

# Compare triple-diff version to original.
triple_dXdt_error = abs(original_dXdt - triple_dXdt)
triple_dMdt_error = abs(original_dMdt - triple_dMdt)
triple_dXdt_match = np.isclose(original_dXdt, triple_dXdt, rtol=REL_TOL, atol=ABS_TOL)
triple_dMdt_match = np.isclose(original_dMdt, triple_dMdt, rtol=REL_TOL, atol=ABS_TOL)


print("getEQU_ISOchange benchmark")
print("--------------------------")
print(f"X:    {X}")
print(f"M_BH: {M_BH}")
print()

print("Runtime")
print("-------")
print(f"original:    {original_time:.6f} seconds")
print(f"Hughes:      {hughes_time:.6f} seconds ({original_time / hughes_time:.2f}x)")
print(f"tripleDiff:  {triple_time:.6f} seconds ({original_time / triple_time:.2f}x)")
print()

print("Outputs")
print("-------")
print(f"original dXdt:   {original_dXdt:.12e}")
print(f"Hughes dXdt:     {hughes_dXdt:.12e}")
print(f"tripleDiff dXdt: {triple_dXdt:.12e}")
print()
print(f"original dMdt:   {original_dMdt:.12e}")
print(f"Hughes dMdt:     {hughes_dMdt:.12e}")
print(f"tripleDiff dMdt: {triple_dMdt:.12e}")
print()

print("Difference from original")
print("------------------------")
print(f"Hughes dXdt error:     {hughes_dXdt_error:.12e} match={hughes_dXdt_match}")
print(f"Hughes dMdt error:     {hughes_dMdt_error:.12e} match={hughes_dMdt_match}")
print(f"tripleDiff dXdt error: {triple_dXdt_error:.12e} match={triple_dXdt_match}")
print(f"tripleDiff dMdt error: {triple_dMdt_error:.12e} match={triple_dMdt_match}")
