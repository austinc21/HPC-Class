#!/bin/bash
#SBATCH --job-name=nbody-omp
#SBATCH --output=omp_out.txt
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --partition=cpu-preempt


g++ -x c++ -fopenmp -O3 -std=c++17 NBODY_OMP.CC -o nbody_omp

echo "Threads N Runtime" > omp_timing.txt

N=10000

for THREADS in 1 2 4 8 16 32 64 128 
do
    echo "Running threads=$THREADS N=$N"
    export OMP_NUM_THREADS=$THREADS
    runtime=$(/usr/bin/time -f "%e" ./nbody_omp $N 2>&1 > /dev/null)
    echo "$THREADS $N $runtime" >> omp_timing.txt
done

echo "Done!"
