#!/bin/bash
# Reproduce the reference end to end: build/verify data, train the proposed
# BatchTopK crosscoder and the L0-matched L1 baseline on every experiment and
# world, and write scoring/scores{,_train,_test}.json.
#
# The hidden test scores the frozen reference/baseline copies kept in
# eval/test/reference_methods/; retraining here does not change them.
# The hardening audit (seeds, variants, shortcuts) is separate:
#   python3 hardening/audit.py

set -e

cd /home/user

bash scripts/reproduce.sh
