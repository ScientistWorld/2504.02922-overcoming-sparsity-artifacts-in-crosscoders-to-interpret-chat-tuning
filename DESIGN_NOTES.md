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

## Runner Pass 2026-05-29

- Submitted CPU job `3c5ee5a4-858`; it exited 0 and produced `scoring/scores_train.json` and `scoring/scores_test.json` from the frozen `checkpoints/crosscoder_suite` artifacts.
- Ran `python3 validate.py --post-design`; all hard validation checks passed. The only warnings were the known heuristic false positives from using `np.random.default_rng` to create deterministic split indices while still reading `val_outputs.npz` method artifacts.
- Train and test method sets match exactly for both experiments (`batchtopk`, `l1`), and every metric in `scoring/reference.json` is populated with numeric values on both slices.
- Train/test are not byte-identical. Sample- and concept-dependent metrics move across slices; decoder-level metrics such as `chat_only_latent_count` can remain identical because they are computed from fixed saved decoders rather than sample partitions.
- Per-stream reconstruction FVE is not saturated across methods: L1 stays near 0.98-0.99 and BatchTopK near 0.95-0.97 on both streams. The values are close between train and test, which is expected for a reconstruction constraint over random half-splits of the same validation distribution.
- Caveat: the held-out concept-family split is coarse, so alignment fractions move in increments of 1/13 top latents. This is still useful as an anti-shortcut signal, but future improvements should be judged across the full metric set rather than a single alignment fraction.
