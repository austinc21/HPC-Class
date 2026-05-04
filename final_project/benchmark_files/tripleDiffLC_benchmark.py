import importlib.util
import sys
from pathlib import Path

from line_profiler import LineProfiler


# This file is inside final_project/benchmark_files.
THIS_DIR = Path(__file__).resolve().parent

# final_project is one folder up from benchmark_files.
PROJECT_DIR = THIS_DIR.parent

# tripleDiff_function.py lives in final_project/tripleDiffLC_optimize.
TRIPLE_DIFF_PATH = PROJECT_DIR / "tripleDiffLC_optimize" / "tripleDiff_function.py"

# Load tripleDiff_function.py directly from its file path.
spec = importlib.util.spec_from_file_location("tripleDiff_function", TRIPLE_DIFF_PATH)
tripleDiff_function = importlib.util.module_from_spec(spec)
sys.modules["tripleDiff_function"] = tripleDiff_function
spec.loader.exec_module(tripleDiff_function)

getEQU_ISOchange = tripleDiff_function.getEQU_ISOchange


# Pick one test input for the benchmark.
X = 0.5
M_BH = 1e6

# Create the line profiler object.
profiler = LineProfiler()

# Tell the profiler which function we want line-by-line timings for.
profiler.add_function(getEQU_ISOchange)

# Make a profiled version of getEQU_ISOchange.
profiled_getEQU_ISOchange = profiler(getEQU_ISOchange)

# Run the function once while the profiler records line timings.
result = profiled_getEQU_ISOchange(X, M_BH)

# Print the function output so we know the run completed correctly.
print("Result:", result)

# Print the line-by-line timing report.
profiler.print_stats()
