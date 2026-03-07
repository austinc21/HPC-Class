#!/bin/bash
#SBATCH -J hw4-lorentz
#SBATCH -p cpu-preempt
#SBATCH -N 2 # nodes
#SBATCH -n 16 # total ranks
#SBATCH -c 1 # CPUs per rank
#SBATCH -t 00:10:00 # walltime
#SBATCH --exclusive # avoid interference
module purge
module load python mpi

source /home/jac190017/miniconda3/etc/profile.d/conda.sh
conda activate HW4
# Avoid oversubscription by BLAS/NumPy threads
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
srun -n 1 python mpi_lorentz.py --n 10000000 --bins 100
