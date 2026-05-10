# tripleDiff Multiprocessing Run Instructions

This folder contains the CPU multiprocessing driver for the optimized
`tripleDiff` calculation:

```bash
tripleDiff_multiprocessing_fixed.py
```

The script calls:

```bash
../tripleDiffLC_optimize/tripleDiff_function.py
```

## Tested Environment

This code was run on the HPC cluster in the conda environment `hpc-final`.
The saved SLURM output files show successful runs with:

- Python 3.11.9 / 3.11.15
- NumPy
- SciPy
- Matplotlib

`multiprocessing`, `argparse`, `sys`, `time`, `pathlib`, `math`, and `pickle`
are part of the Python standard library, so they do not need to be installed
separately.

CuPy and CUDA are not required for `tripleDiff_multiprocessing_fixed.py`.
They are only needed for the separate CuPy versions of the project.

## Recommended Conda Setup

If the `hpc-final` environment is not already available, create a compatible
environment with:

```bash
conda create -n hpc-final python=3.11 numpy scipy matplotlib
conda activate hpc-final
```

Equivalent `pip` setup:

```bash
python -m venv hpc-final
source hpc-final/bin/activate
pip install numpy scipy matplotlib
```

## Running Directly

From this folder:

```bash
python tripleDiff_multiprocessing_fixed.py --N 100 --processes 8
```

Arguments:

- `--N`: number of repeated `getEQU_ISOchange(0.5, 1e6)` evaluations
- `--processes`: number of CPU worker processes

Example output:

```text
N = 100, processes = 8, runtime = ...s
```

## Running with SLURM

The CPU SLURM script is:

```bash
run_tripleDiff_multiprocessing_cpu.slurm
```

Submit the fixed version with:

```bash
SCRIPT=tripleDiff_multiprocessing_fixed.py sbatch run_tripleDiff_multiprocessing_cpu.slurm
```

The SLURM script activates `hpc-final`, uses the current submit directory, and
runs several values of `N`.

## Notes

- Run the command from `final_project/parallelization` so the relative paths
  match the project layout.
- The CPU multiprocessing version does not require a GPU partition.
- The saved warnings about invalid scalar powers come from some parameter
  regions inside the mathematical functions. They appeared in previous runs
  but did not stop the jobs from completing.
