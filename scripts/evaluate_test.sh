#!/bin/bash
# Hidden-test evaluation. Takes no arguments.
#
# Scores every method directory under checkpoints/methods/ on every experiment:
# for each <method>/<experiment>/heldout/crosscoder.npz it samples fresh paired
# inputs from the hidden held-out world in eval/test/worlds/, recomputes all
# activations, reconstructions and metrics from the submitted parameters, and
# writes scoring/scores_test.json (plus unscored diagnostics). A missing or
# invalid artifact scores the complete suite with the worst value of every metric.

set -e

cd "$(dirname "${BASH_SOURCE[0]}")/.."

mkdir -p scoring
python3 -m eval.test.evaluate \
  --methods-root checkpoints/methods \
  --out scoring/scores_test.json \
  --diagnostics scoring/diagnostics_test.json
