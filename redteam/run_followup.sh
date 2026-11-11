#!/bin/bash
set -euo pipefail
cd /home/user
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
for seed in 1 2 3; do
  bash "redteam/better_amplitude_seed${seed}/run.sh"
done
bash redteam/shortcut_score_write/run.sh
python3 -m redteam.summarize | tee redteam/rank_summary.txt
