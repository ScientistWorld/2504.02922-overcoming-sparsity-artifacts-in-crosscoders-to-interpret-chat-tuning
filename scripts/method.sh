#!/bin/bash
# Run the paper's proposed method.
#
# This script trains/runs the paper's method using code in method/.
#
# Usage:
#   bash scripts/method.sh

set -e

cd /home/user

python data/synthetic_activations.py --out-dir data --seed 0
python -m method.train \
  --data data/synthetic_activations.npz \
  --out-dir checkpoints/crosscoder_synthetic \
  --epochs "${EPOCHS:-80}" \
  --batch-size "${BATCH_SIZE:-256}" \
  --n-latents "${N_LATENTS:-88}" \
  --k "${K:-4}" \
  --mu "${MU:-0.002}"

bash scripts/evaluate.sh checkpoints/crosscoder_synthetic
