#!/bin/bash
#SBATCH --job-name=nbody-mpi-shared
#SBATCH --output=mpi_shared_out.txt
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --partition=cpu-preempt

echo "Compiling shared-memory MPI code..."

mpicxx -x c++ -O3 -std=c++17 NBODY_MPI_SHARED.CC -o nbody_mpi_shared

echo "Ranks N Runtime" > mpi_shared_timing.txt

N=10000

for RANKS in 1 2 4 8 16 32 64 128
do
    echo "Running ranks=$RANKS N=$N"
    runtime=$(/usr/bin/time -f "%e" mpiexec -n $RANKS ./nbody_mpi_shared $N 2>&1 > /dev/null)
    echo "$RANKS $N $runtime" >> mpi_shared_timing.txt
done

echo "Done!"
