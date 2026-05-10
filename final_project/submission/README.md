# Final Project Submission

This folder contains the files needed to run the original and optimized versions
of `getEQU_ISOchange(X, M_BH)`.

## Files

```text
original_functions.py
tripleDiffLC_optimize/tripleDiff_function.py
CuPy_optimize/cupy_function.py
CuPy_optimize/test_cupy.py
parallelization/tripleDiff_multiprocessing_fixed.py
parallelization/run_tripleDiff_multiprocessing_cpu.slurm
compare_versions.py
```

## Versions

- `original_functions.py`: original baseline function.
- `tripleDiffLC_optimize/tripleDiff_function.py`: optimized CPU version.
- `CuPy_optimize/cupy_function.py`: attempted CuPy version. I tried to make this
  work on the GPU, but it didn't provide speedup for the problem size I needed, and I do not quite remember the exact installation process I used.

## Dependencies

For the original and optimized CPU versions:

```text
python=3.11
numpy
scipy
matplotlib
```

The CuPy attempt also needs `cupy`, but I am not sure of the exact working
install. I remember there may have been an issue with needing an older NumPy
version, but I could be misremembering. 

The CuPy is not attempt does not provide speedup for my problem size, but I included my attempt out of completeness.

## Setup

```bash
conda create -n hpc-final python=3.11 numpy scipy matplotlib
conda activate hpc-final
```

## Compare Versions

From this `submission` folder:

```bash
python compare_versions.py
```

This compares the original, optimized CPU, and CuPy-attempt versions for:

```text
X = 0.5
M_BH = 1e6
```

## Run CPU Multiprocessing

From `submission/parallelization`:

```bash
python tripleDiff_multiprocessing_fixed.py --N 100 --processes 8
```

On the cluster:

```bash
sbatch run_tripleDiff_multiprocessing_cpu.slurm
```
