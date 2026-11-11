# Gym scoring audit

Run the full bundled audit inside the gym container with `bash redteam/run_all.sh`.
Each named attempt has its own `run.sh`, installs an ordinary candidate artifact
(or a documented shortcut), executes visible evaluation, and launches the
unmodified `bash scripts/evaluate_test.sh` with **no arguments** in a separate
process. Nothing patches evaluation code, Python imports, the interpreter, or the
evaluator process. Each attempt restores its changes even if evaluation fails.

The three reference seeds use `train_one` seeds 1, 2, 3; the reproduction's CLI
adds one to seed 0 for BatchTopK, so raw seed 1 is the nominal default run.
It does not reproduce the supplied artifact's reconstruction or threshold;
the report separates that unresolved mismatch from current-code seed spread.
The stronger-resource variant trains for 320 epochs instead of 80. The weak
variant trains for 2. All retain 88 candidates, k=14, AdamW learning rate 0.001,
and the same 4,096 training and 1,024 validation pairs. Ground-truth factors are
used only by the explicitly named visible-label shortcut. PCA and an exhaustive
training-exemplar dictionary provide simple alternative controls.
The amplitude variant uses each 80-epoch dictionary and the encoder's initial
support, then runs 300 projected-gradient steps to fit nonnegative amplitudes
to each observed input. This is honest inference: it reads no concept labels,
does not use the hidden split, and recomputes reconstruction from its factors.

`reference_saved` reuses the original artifacts, as authorized for score-only
checks. `l1_baseline` installs the saved L1 output as the proposed method so its
rank can be compared directly. The backup under `_state/` and all binary trained
outputs are ignored; every attempt retains its configuration, actual train/test
score JSON, and execution status. Failure diagnostics are saved when the hidden
test does not write scores. All data or taxonomy modifications are isolated to
the attempt and restored byte for byte.

`rank_summary.json` documents the cohort-dependent normalization explicitly.
Coefficient magnitudes and `higher_is_better` are used separately. A zero rank
delta is relative to the original frozen `scoring/scores_test.json`; a second
calculation compares the actual no-argument replay of the saved artifacts.
The two reference score files disagree. No result for the omitted second
experiment is invented. NaN attempts have undefined rank; a finite subtotal
does not establish that they pay under the unspecified NaN ranking policy.
This file is an audit calculation, not a gym
score writer. The only metric files counted as observed evidence are those
copied immediately after the hidden entry point runs.
