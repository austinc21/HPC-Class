#!/bin/bash
#SBATCH --job-name=cupy_call_bench
#SBATCH --output=cupy_call_bench_%j.out
#SBATCH --error=cupy_call_bench_%j.err
#SBATCH --time=0:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
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

# Number of points in the 2D X by M_BH grid.
# Override at submit time, for example:
#   BENCH_PROBLEM_SIZES=10,100 sbatch run_cupy_size_benchmark.slurm
export BENCH_PROBLEM_SIZES="${BENCH_PROBLEM_SIZES:-10}"

echo "Running CuPy 2D-grid benchmark:"
echo "BENCH_PROBLEM_SIZES=${BENCH_PROBLEM_SIZES}"
python benchmark_cupy_sizes.py

echo "Job finished:"
date
