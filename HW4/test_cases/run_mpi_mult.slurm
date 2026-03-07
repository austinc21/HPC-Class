#!/bin/bash
#SBATCH -J hw4-lorentz
#SBATCH -p cpu-preempt
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -c 1
#SBATCH -t 00:10:00
#SBATCH --exclusive

module purge
module load python mpi

source /home/jac190017/miniconda3/etc/profile.d/conda.sh
conda activate HW4

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

OUTCSV=mpi_runtime.csv
echo "method,workers,bins,N,runtime" > $OUTCSV

N=10000000

for bins in 10 100 1000
do
    for workers in 1 2 4 8
    do
        echo "Running bins=$bins workers=$workers"

        output=$(srun --exclusive -n $workers python mpi_lorentz.py --n $N --bins $bins)

        echo "$output"

        runtime=$(echo "$output" | awk '/Runtime:/ {print $(NF-1)}')

        echo "mpi,$workers,$bins,$N,$runtime" >> $OUTCSV
    done
done
