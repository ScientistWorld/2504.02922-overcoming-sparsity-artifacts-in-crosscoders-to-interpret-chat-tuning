#!/bin/bash
# Visible train-slice evaluation. Mirrors scripts/evaluate.sh but writes
# scoring/scores_train.json and enumerates all saved method artifacts.

set -e

cd /home/user

mkdir -p scoring

if [ "${1:-}" = "--suite" ]; then
  ARTIFACTS_ROOT="${2:-checkpoints/crosscoder_suite}"
  DATA_ROOT="${3:-data/benchmarks}"
  python3 -m eval.train.evaluate_suite \
    --artifacts-root "$ARTIFACTS_ROOT" \
    --data-root "$DATA_ROOT" \
    --out scoring/scores_train.json
else
  ARTIFACTS="${1:-checkpoints/crosscoder_synthetic}"
  EXPERIMENT_NAME="${EXPERIMENT_NAME:-synthetic_chat_specificity}"
  python3 -m eval.train.evaluate_crosscoders \
    --artifacts "$ARTIFACTS" \
    --data data/synthetic_activations.npz \
    --metadata data/synthetic_metadata.json \
    --experiment-name "$EXPERIMENT_NAME" \
    --out scoring/scores_train.json
fi
