#!/bin/bash
# Run a baseline method from the paper.
#
# Baselines are simple reference methods the paper compares against. 
# Implement this only if applicable, i.e. the paper has baselines.
# You do not need to implement all baselines to begin — 
# just enough to validate that the evaluation pipeline works 
# and provide a reference point. You can decide to expand more baselines later.
# Each baseline must be actually run and scored through evaluate.sh
# so the numbers are verified, not self-reported.
# You can decide what this baseline script should do yourself.
#
# Usage:
#   bash scripts/baseline.sh <name>

set -e

cd /home/user

python3 data/synthetic_activations.py --out-dir data --seed 0
python3 -m method.train \
  --data data/synthetic_activations.npz \
  --out-dir checkpoints/crosscoder_synthetic \
  --epochs "${EPOCHS:-80}" \
  --batch-size "${BATCH_SIZE:-256}" \
  --n-latents "${N_LATENTS:-88}" \
  --k "${K:-4}" \
  --mu "${MU:-0.005}"

bash scripts/evaluate.sh checkpoints/crosscoder_synthetic
