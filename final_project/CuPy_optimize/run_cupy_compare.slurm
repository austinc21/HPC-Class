#!/bin/bash
#SBATCH --job-name=cupy_compare
#SBATCH --output=cupy_compare_%j.out
#SBATCH --error=cupy_compare_%j.err
#SBATCH --time=1:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1

set -euo pipefail

echo "Job started:"
date

echo "Running on node:"
hostname

echo "Working directory:"
pwd

echo "Activating conda environment:"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hpc-final

echo "Python being used:"
which python
python --version

echo "Checking GPU with nvidia-smi:"
nvidia-smi

echo "Checking CuPy GPU count:"
python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount())"

echo "Running cupy_compare.py:"
python cupy_compare.py

echo "Job finished:"
date
