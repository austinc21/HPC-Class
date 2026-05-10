#!/bin/bash
#SBATCH --job-name=triple_mp_cpu
#SBATCH --output=triple_mp_cpu_%j.out
#SBATCH --error=triple_mp_cpu_%j.err
#SBATCH --time=0:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=8G
#SBATCH --partition=cpu-preempt

source ~/miniconda3/etc/profile.d/conda.sh
conda activate hpc-final
cd "$SLURM_SUBMIT_DIR"

SCRIPT=${SCRIPT:-tripleDiff_multiprocessing_fixed.py}
PROCESSES=${PROCESSES:-$SLURM_CPUS_PER_TASK}

echo "script = $SCRIPT"
echo "processes = $PROCESSES"

for N in 10 100 1000 10000 100000 1000000
do
    python $SCRIPT --N $N --processes $PROCESSES
done
