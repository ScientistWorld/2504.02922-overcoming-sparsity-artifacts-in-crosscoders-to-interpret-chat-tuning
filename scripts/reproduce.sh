#!/bin/bash
# Reproduce the reference end to end: data, proposed method, baseline, and the
# train-side and hidden-test scores (scoring/scores_{train,test}.json).

set -e

cd "$(dirname "${BASH_SOURCE[0]}")/.."

bash scripts/download.sh
bash scripts/method.sh
bash scripts/baseline.sh
bash scripts/evaluate.sh
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
python3 validate.py
