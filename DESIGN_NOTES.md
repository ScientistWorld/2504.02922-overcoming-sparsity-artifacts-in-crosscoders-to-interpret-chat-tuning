# Train/Test Evaluation Design Notes

## Checked

- Read `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh` side by side. The train script calls `eval.train.*` and writes `scoring/scores_train.json`; the test script calls `eval.test.*` and writes `scoring/scores_test.json`.
- Checked that the train and test evaluators read saved `val_outputs.npz` method artifacts rather than regenerating method outputs in-place.
- Checked train/test import direction with grep. There are no code imports from `eval/train` into `eval/test` or from `eval/test` into `eval/train`; the only matches are explanatory comments and the test-side `scores_train.json` method-enumeration input.
- Checked for paper-agreement metric names in the split/scoring files. No nonzero scored metric is shaped as agreement with published values.
- Confirmed `scoring/scores_train.json` and `scoring/scores_test.json` do not exist.
- Ran `bash -n scripts/evaluate_train.sh` and `bash -n scripts/evaluate_test.sh`.

## Changed

- Added `base_reconstruction_fve` and `chat_reconstruction_fve` to both `eval/train/evaluate_crosscoders.py` and `eval/test/evaluate_crosscoders.py`.
- Added matching metric entries to both experiments in `scoring/reference.json`.
- Added required `description` fields to every metric in `scoring/reference.json`.
- Reduced the aggregate `reconstruction_fve` coefficient from `0.2` to `0.15` and gave the two per-stream reconstruction metrics `0.1` each. This keeps specificity and false-positive metrics dominant while making one-sided reconstruction shortcuts visible.

## Runtime Concerns For Next Iteration

- The new per-stream FVE metrics should be inspected after the next evaluator run to ensure neither stream saturates near identical values for all methods.
- The held-out split currently uses complementary validation samples and complementary concept-family members. That is a reasonable anti-shortcut split for this synthetic crosscoder setup, but it still depends on the synthetic concept families being large enough for stable half splits.
- The paper markdown path provided in the prompt was not present on disk during this pass, so metric-design review was based on the existing workspace implementation, targets, constraints, and scoring metadata.
