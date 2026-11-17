#!/bin/bash
# Train the paper's baseline: the L1 crosscoder, with its sparsity weight mu
# calibrated on each world's training activations so that its L0 matches the
# BatchTopK crosscoder (k = 6), as in the paper's L0-matched comparison.
# Writes checkpoints/methods/l1/<experiment>/{dev,heldout}/crosscoder.npz.

set -e

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python3 -m method.train --method l1 --out checkpoints/methods/l1 \
  --seed 0 --workers "${WORKERS:-1}"
