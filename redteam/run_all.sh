#!/bin/bash
set -euo pipefail
cd /home/user
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python3 -m redteam.audit all
python3 -m redteam.summarize | tee redteam/rank_summary.txt
