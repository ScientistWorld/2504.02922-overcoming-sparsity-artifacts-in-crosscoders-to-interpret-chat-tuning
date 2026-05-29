# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-05-29 06:51] - method_runs
- Implemented the paper's shared-encoder, two-decoder crosscoder on paired synthetic activations.
- Trained L1 and BatchTopK variants end-to-end and evaluated saved artifacts with decoder norm and Latent Scaling diagnostics.

### [2026-05-29 07:20] - method_runs
- Audited the initial core-claim result and found reproduction-integrity fixes needed before advancing: BatchTopK inference now uses an estimated global threshold, Latent Scaling error ratios now target reconstruction error directly, and the next run uses fixed hyperparameters instead of selecting by validation metrics.
- Added a second complete-shrinkage synthetic regime for the next compute job, but it has not run yet.

## Stop Justification
- Not stopping. Next action submits a CPU job to regenerate suite results after the integrity fixes.
