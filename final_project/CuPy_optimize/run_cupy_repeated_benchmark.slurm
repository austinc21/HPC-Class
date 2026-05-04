#!/bin/bash
#SBATCH --job-name=cupy_repeat_bench
#SBATCH --output=cupy_repeat_bench_%j.out
#SBATCH --error=cupy_repeat_bench_%j.err
#SBATCH --time=0:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate hpc-final

cd "$(dirname "$0")"

echo "node: $(hostname)"
echo "python: $(which python)"
nvidia-smi

N_VALUES=(10 100 1000 10000 100000 1000000)

for N in "${N_VALUES[@]}"; do
    start=$(date +%s.%N)

    python - "$N" <<'PY'
import sys

import cupy_function as cf

n = int(sys.argv[1])

for _ in range(n):
    cf.getEQU_ISOchange(0.5, 1e6)

if cf.CUPY_AVAILABLE:
    cf.cp.cuda.Stream.null.synchronize()
PY

    end=$(date +%s.%N)
    runtime=$(awk "BEGIN {print $end - $start}")
    echo "N = ${N}, ${runtime}s"
done
