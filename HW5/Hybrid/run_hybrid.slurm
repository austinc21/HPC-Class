#!/bin/bash
#SBATCH --job-name=nbody-hybrid
#SBATCH --output=hybrid_out.txt
#SBATCH --time=00:15:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu-preempt


echo "Compiling hybrid code..."

mpicxx -x c++ -fopenmp -O3 -std=c++17 NBODY_MPI_OMP.CC -o nbody_mpi_omp

# Set OpenMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Set your problem size here
N=20000

echo "Running hybrid with:"
echo "MPI ranks = $SLURM_NTASKS"
echo "OMP threads per rank = $OMP_NUM_THREADS"
echo "N = $N"

# Run and time
runtime=$(/usr/bin/time -f "%e" mpiexec -n $SLURM_NTASKS ./nbody_mpi_omp $N 2>&1)

echo "Runtime (seconds): $runtime"
