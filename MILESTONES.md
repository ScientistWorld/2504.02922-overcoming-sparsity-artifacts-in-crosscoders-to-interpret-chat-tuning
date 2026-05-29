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

## Stop Justification
- Stopping at `core_claim`: the actual crosscoder method and BatchTopK objective run end-to-end, the corrected evaluator reads saved method artifacts, and the scaled suite demonstrates reduced false fine-tuned-only artifacts while preserving high reconstruction. Higher milestones would require real Gemma 2 activation extraction/training and causal KL patching, which are outside the completed scaled reproduction.
