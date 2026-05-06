#!/bin/bash
#SBATCH --job-name=real_serial_N
#SBATCH --output=real_serial_N_%j.out
#SBATCH --error=real_serial_N_%j.err
#SBATCH --time=5:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=cpu-preempt

source ~/miniconda3/etc/profile.d/conda.sh
conda activate hpc-final
cd "$SLURM_SUBMIT_DIR"

for N in 10 100 1000
do
    start=$(date +%s.%N)
    python real_serial_tripleDiff_once.py --N $N
    end=$(date +%s.%N)
    runtime=$(awk "BEGIN {print $end - $start}")
    echo "N = $N, runtime = ${runtime}s"
done
