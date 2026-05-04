import sys
from pathlib import Path

from line_profiler import LineProfiler


# This file is inside final_project/initial benchmark.
# edited_functions.py is one folder up, inside final_project.
PROJECT_DIR = Path(__file__).resolve().parents[1]

# Add final_project to Python's import path so we can import edited_functions.py.
sys.path.insert(0, str(PROJECT_DIR))

from edited_functions import getEQU_ISOchange


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
