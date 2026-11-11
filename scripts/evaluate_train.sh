#!/bin/bash
# Train-side (development) evaluation. Takes no arguments.
#
# Scores every method directory under checkpoints/methods/ on every experiment
# using the public dev worlds: <method>/<experiment>/dev/crosscoder.npz is
# scored on fresh inputs sampled from data/worlds/<experiment>/dev/world.npz.
# Writes scoring/scores_train.json (plus unscored diagnostics). Same metrics
# and constraints as the hidden test.

set -e

cd "$(dirname "${BASH_SOURCE[0]}")/.."

mkdir -p scoring
python3 -m eval.train.evaluate \
  --methods-root checkpoints/methods \
  --out scoring/scores_train.json \
  --diagnostics scoring/diagnostics_train.json
