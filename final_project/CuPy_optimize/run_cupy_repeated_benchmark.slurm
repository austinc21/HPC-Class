#!/bin/bash
#SBATCH --job-name=cupy_repeat_bench
#SBATCH --output=cupy_repeat_bench_%j.out
#SBATCH --error=cupy_repeat_bench_%j.err
#SBATCH --time=0:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate hpc-final
cd "$(dirname "$0")"

for N in 10 100 1000 10000 100000 1000000
do
    start=$(date +%s.%N)
    python run_cupy_once.py --N $N
    end=$(date +%s.%N)
    runtime=$(awk "BEGIN {print $end - $start}")
    echo "N = $N, ${runtime}s"
done
