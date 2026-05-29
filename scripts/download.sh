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
# Estimated disk usage after generation: < 10 MB
#
# This benchmark uses generated paired activations and has no external dataset
# or model downloads. The commands below regenerate every ignored activation
# bundle that the method and suite evaluators read.

set -e

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if ! python3 -c "import numpy" >/dev/null 2>&1; then
  echo "No external downloads required."
  echo "Install dependencies with environment/setup.sh or run inside the compute container to regenerate synthetic activations."
  exit 0
fi

generate_if_missing() {
  local out_dir="$1"
  local scenario="$2"

  if [ ! -f "$out_dir/synthetic_activations.npz" ]; then
    python3 data/synthetic_activations.py \
      --out-dir "$out_dir" \
      --seed 0 \
      --scenario "$scenario"
  fi
}

generate_if_missing data decoupling
generate_if_missing data/benchmarks/decoupling decoupling
generate_if_missing data/benchmarks/complete_shrinkage complete_shrinkage
