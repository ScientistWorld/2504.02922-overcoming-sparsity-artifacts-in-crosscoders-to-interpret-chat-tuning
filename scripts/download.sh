#!/bin/bash
# Download all data and models needed for this environment.
#
# Run on a node with internet access BEFORE building the container
# or submitting compute jobs.
#
# This script should be idempotent — safe to run multiple times.
# It should skip files that already exist.
#
# Update the storage estimate below as you add downloads.
#
# Estimated total download size: 0 GB
# Estimated disk usage after extraction: < 10 MB
#
# This benchmark uses generated paired activations and has no external
# dataset or model downloads. The compute scripts regenerate the small
# activation files inside the container.

set -e

cd /home/user

if python3 -c "import numpy" >/dev/null 2>&1; then
  python3 data/synthetic_activations.py --out-dir data --seed 0
else
  echo "No external downloads required; NumPy is only needed in the compute container."
fi
