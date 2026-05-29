#!/bin/bash
# Compute job: reproduce the crosscoder artifact-control claim on two
# scaled paired-activation regimes: latent decoupling and complete shrinkage.

set -e

cd /home/user

SUITE_ROOT="checkpoints/crosscoder_suite"
LEGACY_DIR="checkpoints/crosscoder_synthetic"
DATA_ROOT="data/benchmarks"
EPOCHS="${EPOCHS:-140}"
BATCH_SIZE="${BATCH_SIZE:-256}"
N_LATENTS="${N_LATENTS:-88}"
MU="${MU:-0.005}"
K="${K:-12}"

rm -rf "$SUITE_ROOT" "$LEGACY_DIR" "$DATA_ROOT"
mkdir -p "$SUITE_ROOT" "$DATA_ROOT"

run_case() {
  local case_name="$1"
  local scenario="$2"
  local data_dir="$DATA_ROOT/$case_name"
  local artifact_dir="$SUITE_ROOT/$case_name"

  mkdir -p "$data_dir" "$artifact_dir"
  echo "[run] Generating ${scenario} benchmark"
  if [ "$scenario" = "complete_shrinkage" ]; then
    python3 data/synthetic_activations.py \
      --out-dir "$data_dir" \
      --scenario "$scenario" \
      --base-weak-scale 0.25 \
      --seed 0
  else
    python3 data/synthetic_activations.py \
      --out-dir "$data_dir" \
      --scenario "$scenario" \
      --seed 0
  fi

  echo "[run] Training L1 crosscoder for ${case_name}"
  python3 -m method.train \
    --data "$data_dir/synthetic_activations.npz" \
    --out-dir "$artifact_dir" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --n-latents "$N_LATENTS" \
    --k "$K" \
    --mu "$MU" \
    --seed 0 \
    --methods l1

  echo "[run] Training BatchTopK crosscoder for ${case_name}"
  python3 -m method.train \
    --data "$data_dir/synthetic_activations.npz" \
    --out-dir "$artifact_dir" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --n-latents "$N_LATENTS" \
    --k "$K" \
    --mu "$MU" \
    --seed 1 \
    --methods batchtopk
}

run_case "decoupling" "decoupling"
run_case "complete_shrinkage" "complete_shrinkage"

cp -a "$DATA_ROOT/decoupling/synthetic_activations.npz" data/synthetic_activations.npz
cp -a "$DATA_ROOT/decoupling/synthetic_metadata.json" data/synthetic_metadata.json
cp -a "$SUITE_ROOT/decoupling" "$LEGACY_DIR"

echo "[run] Evaluating full suite"
bash scripts/evaluate.sh --suite "$SUITE_ROOT" "$DATA_ROOT"
python3 validate.py --compare
