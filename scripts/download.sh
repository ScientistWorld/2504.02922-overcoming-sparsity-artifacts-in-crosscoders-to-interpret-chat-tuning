#!/bin/bash
# Prepare all data for this environment. Idempotent.
#
# Estimated total download size: 0 GB (no external datasets or model weights).
# Estimated disk usage: ~65 MB.
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

# The private builder is present only in the evaluator/reproducer environment.
# Its own version check makes this idempotent. Scientist clones receive the
# committed unlabeled held-out training files, without the private builder.
if [ -f eval/test/build_heldout.py ]; then
  python3 -m eval.test.build_heldout
fi
for exp in synthetic_chat_specificity synthetic_faint_chat_concepts; do
  test -f "data/worlds/$exp/heldout/train.npz" || {
    echo "missing committed training data: data/worlds/$exp/heldout/train.npz" >&2
    exit 1
  }
done
