#!/bin/bash
#SBATCH --job-name=triple_worker_cpu
#SBATCH --output=triple_worker_cpu_%j.out
#SBATCH --error=triple_worker_cpu_%j.err
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=8G
#SBATCH --partition=cpu-preempt

source ~/miniconda3/etc/profile.d/conda.sh
conda activate hpc-final

cd "$SLURM_SUBMIT_DIR/.."

N=100

for WORKERS in 1 2 4 8 16 32 64
do
    start=$(date +%s.%N)
    python tripleDiff_multiprocessing_fixed.py --N $N --processes $WORKERS
    end=$(date +%s.%N)
    runtime=$(awk "BEGIN {print $end - $start}")
    echo "N = $N, workers = $WORKERS, runtime = ${runtime}s"
done
