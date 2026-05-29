#!/bin/bash
# Compute job: regenerate the scaled benchmark, train both crosscoder variants
# on each case, then score train/test/full slices.

set -e

cd /home/user

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

bash scripts/evaluate_train.sh --suite checkpoints/crosscoder_suite data/benchmarks
bash scripts/evaluate_test.sh --suite checkpoints/crosscoder_suite data/benchmarks
bash scripts/evaluate.sh --suite checkpoints/crosscoder_suite data/benchmarks
python3 validate.py --compare
