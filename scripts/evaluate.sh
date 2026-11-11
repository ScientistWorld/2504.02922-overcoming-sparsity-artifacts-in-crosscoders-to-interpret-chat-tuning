#!/bin/bash
# Standard (public) evaluation: scores every method under checkpoints/methods/
# on the public dev worlds of every experiment and writes scoring/scores.json.
# Uses only eval/train/ code, so it works without the hidden test.
#
# Output contract per method and experiment:
#   checkpoints/methods/<method>/<experiment>/dev/crosscoder.npz
# (see briefing/evaluation.md for the arrays it must contain).

set -e

cd "$(dirname "${BASH_SOURCE[0]}")/.."

mkdir -p scoring
python3 -m eval.train.evaluate \
  --methods-root checkpoints/methods \
  --out scoring/scores.json
