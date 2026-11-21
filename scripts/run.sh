#!/bin/bash
# Reproduce the reference end to end from the committed data and source.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
bash scripts/reproduce.sh
