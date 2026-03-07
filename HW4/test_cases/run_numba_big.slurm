#!/bin/bash
#SBATCH -J numba_scaling
#SBATCH -p cpu-preempt
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 00:30:00
#SBATCH --exclusive

module purge

source /home/jac190017/miniconda3/etc/profile.d/conda.sh
conda activate HW4

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

N=1000000000
BINS=100

echo "method,workers,bins,N,runtime" > numba_scaling.csv

for p in 1 2 4 8 16
do
    output=$(python run_numba_once.py --workers $p --N $N --bins $BINS)
    runtime=$(echo "$output" | awk '/Runtime:/ {print $2}')
    echo "numba,$p,$BINS,$N,$runtime" >> numba_scaling.csv
done