#!/bin/bash
# Held-out test-slice evaluation. Mirrors scripts/evaluate.sh but writes
# scoring/scores_test.json and, when available, scores every method listed in
# scoring/scores_train.json.

set -e

cd /home/user

mkdir -p scoring

TRAIN_SCORES="${TRAIN_SCORES:-scoring/scores_train.json}"

if [ "${1:-}" = "--suite" ]; then
  ARTIFACTS_ROOT="${2:-checkpoints/crosscoder_suite}"
  DATA_ROOT="${3:-data/benchmarks}"
  python3 -m eval.test.evaluate_suite \
    --artifacts-root "$ARTIFACTS_ROOT" \
    --data-root "$DATA_ROOT" \
    --train-scores "$TRAIN_SCORES" \
    --out scoring/scores_test.json
else
  ARTIFACTS="${1:-checkpoints/crosscoder_synthetic}"
  EXPERIMENT_NAME="${EXPERIMENT_NAME:-synthetic_chat_specificity}"
  python3 -m eval.test.evaluate_crosscoders \
    --artifacts "$ARTIFACTS" \
    --data data/synthetic_activations.npz \
    --metadata data/synthetic_metadata.json \
    --experiment-name "$EXPERIMENT_NAME" \
    --train-scores "$TRAIN_SCORES" \
    --out scoring/scores_test.json
fi
