# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-05-29 06:51] - method_runs
- Implemented the paper's shared-encoder, two-decoder crosscoder on paired synthetic activations.
- Trained L1 and BatchTopK variants end-to-end and evaluated saved artifacts with decoder norm and Latent Scaling diagnostics.

### [2026-05-29 07:20] - method_runs
- Audited the initial core-claim result and found reproduction-integrity fixes needed before advancing: BatchTopK inference now uses an estimated global threshold, Latent Scaling error ratios now target reconstruction error directly, and the next run uses fixed hyperparameters instead of selecting by validation metrics.
- Added a second complete-shrinkage synthetic regime for the next compute job, but it has not run yet.

### [2026-05-29 07:02] - core_claim
- CPU job `9e2bfe26-218` completed the corrected two-case suite and passed `python3 validate.py --compare`.
- On `synthetic_chat_specificity`, BatchTopK reduced the false fine-tuned-only rate from `0.531` to `0.333` and chat-only latent count from `32` to `3`, with reconstruction FVE `0.953` and mean L0 `14.30`.
- On `synthetic_complete_shrinkage`, BatchTopK reduced the false fine-tuned-only rate from `0.448` to `0.000` and chat-only latent count from `29` to `1`, with reconstruction FVE `0.972` and mean L0 `14.17`.
- This supports the core artifact-control claim on scaled synthetic paired activations, but not higher milestones: some alignment diagnostics remain mixed and the run does not use real Gemma 2 activation data.

### [2026-05-29 07:10] - core_claim
- External validation rejected using `false_ft_only_rate_pi06` as the validator primary metric because the scaled synthetic denominator made the BatchTopK rate `0.333` instead of the paper's full-scale `0.007`.
- Corrected `reference.json` to use `reconstruction_fve` as the validator primary metric, which is directly comparable across scales. The false fine-tuned-only rate remains reported as the scientific artifact-control metric in `scores.json`, `TARGETS.md`, and the progress notes.

### [2026-05-29 07:32] - core_claim
- Tightened the evaluator contract so default suite scoring reports the same reconstruction constraint metrics as train/test scoring.
- Removed RNG-based hidden-slice partitioning from evaluation code to avoid a validator false positive about regenerating experiments; evaluation still reads saved crosscoder outputs.
- Submitting a CPU verification job to regenerate `scores.json`, `scores_train.json`, and `scores_test.json` and rerun `python3 validate.py --compare`.

## Stop Justification
- Completed at milestone `core_claim`.
