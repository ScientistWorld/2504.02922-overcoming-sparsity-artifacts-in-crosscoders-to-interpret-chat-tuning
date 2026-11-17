#!/bin/bash
# Train the paper's proposed method (BatchTopK crosscoder with the auxiliary
# dead-latent loss and an inference threshold estimated from training data)
# on the dev and held-out training activations of every experiment.
# Writes checkpoints/methods/batchtopk/<experiment>/{dev,heldout}/crosscoder.npz.

set -e

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python3 -m method.train --method batchtopk --out checkpoints/methods/batchtopk \
  --seed 0 --workers "${WORKERS:-1}"
