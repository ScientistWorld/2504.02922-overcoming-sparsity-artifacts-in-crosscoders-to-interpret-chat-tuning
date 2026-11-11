#!/bin/bash
# Prepare all data for this environment. Idempotent.
#
# Estimated total download size: 0 GB (no external datasets or model weights).
# Estimated disk usage: ~35 MB.
#
# The benchmark uses synthetic paired activations:
#   * data/worlds/<experiment>/dev/      public development worlds; regenerated
#                                        here from data/synthetic_activations.py
#                                        if missing (they are also committed).
#   * data/worlds/<experiment>/heldout/  unlabeled training activations of the
#                                        held-out worlds; committed to git.

set -e

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python3 data/synthetic_activations.py

for exp in synthetic_chat_specificity synthetic_faint_chat_concepts; do
  if [ ! -f "data/worlds/$exp/heldout/train.npz" ]; then
    if [ -f eval/test/build_heldout.py ]; then
      python3 -m eval.test.build_heldout
    else
      echo "missing data/worlds/$exp/heldout/train.npz (committed in git; restore it from the repository)" >&2
      exit 1
    fi
  fi
done
