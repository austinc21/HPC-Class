#!/bin/bash
#SBATCH --job-name=cupy_mp_bench
#SBATCH --output=cupy_mp_bench_%j.out
#SBATCH --error=cupy_mp_bench_%j.err
#SBATCH --time=0:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate hpc-final
cd "$SLURM_SUBMIT_DIR"

SCRIPT=${SCRIPT:-cupy_multiprocessing_grid.py}
PROCESSES=${PROCESSES:-$SLURM_CPUS_PER_TASK}

echo "script = $SCRIPT"
echo "processes = $PROCESSES"

for N in 10 100 1000 10000 100000 1000000
do
    python $SCRIPT --N $N --processes $PROCESSES
done
