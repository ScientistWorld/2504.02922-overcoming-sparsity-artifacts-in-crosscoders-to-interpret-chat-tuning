# Gym scoring audit

Run the full bundled audit inside the gym container with `bash redteam/run_all.sh`.
Each named attempt has its own `run.sh`, installs an ordinary candidate artifact
(or a documented shortcut), executes visible evaluation, and launches the
unmodified `bash scripts/evaluate_test.sh` with **no arguments** in a separate
process. Nothing patches evaluation code, Python imports, the interpreter, or the
evaluator process. Each attempt restores its changes even if evaluation fails.

The three reference seeds use `train_one` seeds 1, 2, 3; the reproduction's CLI
adds one to seed 0 for BatchTopK, so seed 1 corresponds to the saved reference.
The stronger-resource variant trains for 320 epochs instead of 80. The weak
variant trains for 2. All retain 88 candidates, k=14, AdamW learning rate 0.001,
and the same 4,096 training and 1,024 validation pairs. Ground-truth factors are
used only by the explicitly named visible-label shortcut. PCA and an exhaustive
training-exemplar dictionary provide simple alternative controls.

`reference_saved` reuses the original artifacts, as authorized for score-only
checks. `l1_baseline` installs the saved L1 output as the proposed method so its
rank can be compared directly. The backup under `_state/` and all binary trained
outputs are ignored; every attempt retains its configuration, actual train/test
score JSON, and execution status. Failure diagnostics are saved when the hidden
test does not write scores. All data or taxonomy modifications are isolated to
the attempt and restored byte for byte.

`rank_summary.json` documents the cohort-dependent normalization explicitly.
Coefficient magnitudes and `higher_is_better` are used separately. A zero rank
delta is a tie with the actual no-argument reference; no result for the omitted
second experiment is invented. This file is an audit calculation, not a gym
score writer. The only metric files counted as observed evidence are those
copied immediately after the hidden entry point runs.
