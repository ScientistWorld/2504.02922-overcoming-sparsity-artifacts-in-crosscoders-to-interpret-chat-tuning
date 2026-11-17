#!/bin/bash
# Single consolidated final hardening audit. Restored to reference reproduction
# after the audit has completed and its evidence has been reviewed.
set -euo pipefail
cd /home/user
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python3 hardening/previous_probe.py
bash scripts/download.sh
python3 hardening/final_audit.py
