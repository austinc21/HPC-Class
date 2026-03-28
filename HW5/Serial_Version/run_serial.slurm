#!/bin/bash
#SBATCH --job-name=nbody-serial
#SBATCH --output=serial_out.txt
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --partition=cpu-preempt

g++ -x c++ -O3 -std=c++17 NBODY.CC -o nbody

echo "N Runtime" > timing.txt

for N in 128 256 512 1024 2048 4096 8192 16384 32768
do
    echo "Running N=$N"
    runtime=$(/usr/bin/time -f "%e" ./nbody $N 2>&1 > /dev/null)
    echo "$N $runtime" >> timing.txt
done

echo "Done!"
