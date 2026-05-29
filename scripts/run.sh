#!/bin/bash
# Example job script submitted via the end-of-turn YAML block.
#
# This runs inside the compute container with:
#   - Your workspace mounted at /home/user
#   - GPU(s) available
#   - No internet access
#
# Make sure all data, code, and dependencies are already in the
# workspace or baked into the container before submitting.

set -e

cd /home/user

if [ ! -f checkpoints/crosscoder_synthetic/l1/val_outputs.npz ] || \
   [ ! -f checkpoints/crosscoder_synthetic/batchtopk/val_outputs.npz ]; then
  bash scripts/method.sh
fi
bash scripts/evaluate.sh checkpoints/crosscoder_synthetic
python3 validate.py --compare
