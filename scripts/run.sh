#!/bin/bash
# Compute job: score the frozen final crosscoder suite on the visible train
# slice and held-out test slice. This does not retrain or modify method code.

set -e

cd /home/user

bash scripts/evaluate_train.sh --suite /home/user/checkpoints/crosscoder_suite /home/user/data/benchmarks
bash scripts/evaluate_test.sh --suite /home/user/checkpoints/crosscoder_suite /home/user/data/benchmarks
