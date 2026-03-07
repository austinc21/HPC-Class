#!/bin/bash
#SBATCH -J numba_scaling
#SBATCH -p cpu-preempt
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 00:10:00
#SBATCH --exclusive

module purge
module load python

source /home/jac190017/miniconda3/etc/profile.d/conda.sh
conda activate HW4

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

N=1000000000
BINS=100

echo "method,workers,bins,N,runtime" > numba_scaling.csv

for p in 1 2 4 8 16
do
    echo "Running numba with $p workers"

    output=$(python benchmark.py --method numba --workers $p --N $N --bins $BINS)

    echo "$output"

    runtime=$(echo "$output" | awk '/Runtime:/ {print $(NF-1); exit}')

    echo "numba,$p,$BINS,$N,$runtime" >> numba_scaling.csv
done
