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
# dataset or model downloads.

set -e

cd /home/user

python data/synthetic_activations.py --out-dir data --seed 0
