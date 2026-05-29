#!/bin/bash
# Run the paper's proposed method.
#
# This script trains/runs the paper's method using code in method/.
#
# Usage:
#   bash scripts/method.sh          # single default benchmark
#   bash scripts/method.sh --suite  # both benchmark cases

set -e

cd /home/user

if [ "${1:-}" = "--suite" ]; then
  EPOCHS="${EPOCHS:-80}"
  BATCH_SIZE="${BATCH_SIZE:-256}"
  N_LATENTS="${N_LATENTS:-88}"
  K="${K:-14}"
  MU="${MU:-0.005}"

  python3 data/synthetic_activations.py \
    --out-dir data/benchmarks/decoupling \
    --seed 0 \
    --scenario decoupling
  python3 -m method.train \
    --data data/benchmarks/decoupling/synthetic_activations.npz \
    --out-dir checkpoints/crosscoder_suite/decoupling \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --n-latents "$N_LATENTS" \
    --k "$K" \
    --mu "$MU"

  python3 data/synthetic_activations.py \
    --out-dir data/benchmarks/complete_shrinkage \
    --seed 0 \
    --scenario complete_shrinkage
  python3 -m method.train \
    --data data/benchmarks/complete_shrinkage/synthetic_activations.npz \
    --out-dir checkpoints/crosscoder_suite/complete_shrinkage \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --n-latents "$N_LATENTS" \
    --k "$K" \
    --mu "$MU"

  bash scripts/evaluate.sh --suite checkpoints/crosscoder_suite data/benchmarks
  exit 0
fi

python3 data/synthetic_activations.py --out-dir data --seed 0
python3 -m method.train \
  --data data/synthetic_activations.npz \
  --out-dir checkpoints/crosscoder_synthetic \
  --epochs "${EPOCHS:-80}" \
  --batch-size "${BATCH_SIZE:-256}" \
  --n-latents "${N_LATENTS:-88}" \
  --k "${K:-14}" \
  --mu "${MU:-0.005}"

bash scripts/evaluate.sh checkpoints/crosscoder_synthetic
