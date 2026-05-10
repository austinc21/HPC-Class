import importlib.util
import time
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent


def load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def time_equ_iso(label, module, x_value, m_bh):
    start = time.perf_counter()
    d_xdt, d_mdt = module.getEQU_ISOchange(x_value, m_bh)
    elapsed = time.perf_counter() - start
    return {
        "label": label,
        "time": elapsed,
        "dXdt": d_xdt,
        "dMdt": d_mdt,
    }


def main():
    original = load_module("original_functions", THIS_DIR / "original_functions.py")
    triple_diff = load_module(
        "tripleDiff_function",
        THIS_DIR / "tripleDiffLC_optimize" / "tripleDiff_function.py",
    )
    cupy_diff = load_module(
        "cupy_function",
        THIS_DIR / "CuPy_optimize" / "cupy_function.py",
    )

    x_value = 0.5
    m_bh = 1e6
    rel_tol = 1e-8
    abs_tol = 1e-12

    results = [
        time_equ_iso("original", original, x_value, m_bh),
        time_equ_iso("tripleDiff optimized", triple_diff, x_value, m_bh),
        time_equ_iso("CuPy attempt", cupy_diff, x_value, m_bh),
    ]

    baseline = results[0]

    print("getEQU_ISOchange comparison")
    print("---------------------------")
    print(f"X: {x_value}")
    print(f"M_BH: {m_bh}")
    print(f"CuPy available: {getattr(cupy_diff, 'CUPY_AVAILABLE', False)}")
    print()

    for result in results:
        d_xdt_match = np.isclose(
            baseline["dXdt"], result["dXdt"], rtol=rel_tol, atol=abs_tol
        )
        d_mdt_match = np.isclose(
            baseline["dMdt"], result["dMdt"], rtol=rel_tol, atol=abs_tol
        )
        print(result["label"])
        print(f"  runtime: {result['time']:.6f} seconds")
        print(f"  dXdt:    {result['dXdt']:.12e} match={d_xdt_match}")
        print(f"  dMdt:    {result['dMdt']:.12e} match={d_mdt_match}")
        print()


if __name__ == "__main__":
    main()
