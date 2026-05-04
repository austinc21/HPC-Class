#!/bin/bash
#SBATCH --job-name=triple_repeat_bench
#SBATCH --output=triple_repeat_bench_%j.out
#SBATCH --error=triple_repeat_bench_%j.err
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
import importlib.util
import sys
from pathlib import Path

this_dir = Path.cwd()
project_dir = this_dir.parent
module_path = project_dir / "tripleDiffLC_optimize" / "tripleDiff_function.py"

spec = importlib.util.spec_from_file_location("tripleDiff_function", module_path)
tf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tf)

n = int(sys.argv[1])

for _ in range(n):
    tf.getEQU_ISOchange(0.5, 1e6)
PY

    end=$(date +%s.%N)
    runtime=$(awk "BEGIN {print $end - $start}")
    echo "N = ${N}, ${runtime}s"
done
