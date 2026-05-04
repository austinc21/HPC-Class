#!/bin/bash
#SBATCH --job-name=cupy_test
#SBATCH --output=cupy_test_%j.out
#SBATCH --error=cupy_test_%j.err
#SBATCH --time=0:02:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1

set -euo pipefail

echo "Job started:"
date

echo "Running on node:"
hostname

echo "Activating conda environment:"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hpc-final

echo "Python being used:"
which python
python --version

echo "Testing CuPy import:"
python -c "import cupy; print('cupy import works')"

echo "Checking GPU with nvidia-smi:"
nvidia-smi

echo "Checking CuPy GPU count:"
python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount())"

echo "Running test_cupy.py:"
python test_cupy.py

echo "Job finished:"
date
EOF
