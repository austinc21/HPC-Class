#!/bin/bash
#SBATCH --job-name=nbody-mpi
#SBATCH --output=mpi_out.txt
#SBATCH --time=01:00:00
#SBATCH --ntasks=128
#SBATCH --partition=cpu-preempt

mpicxx -x c++ -O3 -std=c++17 NBODY_MPI.CC -o nbody_mpi

echo "Ranks N Runtime" > mpi_timing.txt

N=10000

for RANKS in 1 2 4 8 16 32 64 128
do
    echo "Running ranks=$RANKS N=$N"
    runtime=$(/usr/bin/time -f "%e" mpiexec -n $RANKS ./nbody_mpi $N 2>&1 > /dev/null)
    echo "$RANKS $N $runtime" >> mpi_timing.txt
done

echo "Done!"
