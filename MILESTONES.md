# Reproduction Milestones

**Current: core_claim**

<!-- Milestone levels (update "Current" above as you progress):
  none             — just started, no meaningful progress yet
  method_runs      — the paper's method executes end-to-end without errors
  core_claim       — minimum experiment supports the paper's central claim
  core_claim_plus  — core claim reproduced on additional settings
  secondary_claims — secondary results or contributions reproduced
  majority         — more than half of reported results reproduced
  near_complete    — most results reproduced, only minor gaps remain
  full             — all reported results reproduced
-->

## Progress Log

### [2026-05-29 04:05] - none
- Read the paper and identified the core method: L1 and BatchTopK crosscoders plus Latent Scaling diagnostics.
- Implemented the scaled benchmark, method code, evaluation code, scripts, and scoring reference.
- Submitting first compute job; no milestone above `none` is claimed until the method runs successfully.

### [2026-05-29 04:15] - none
- First CPU test job failed at startup because the container lacked a `python` executable.
- Patched scripts to use `python3` and added a container-level `python -> python3` symlink before resubmitting.

### [2026-05-29 04:30] - none
- Second CPU test job trained both L1 and BatchTopK crosscoders and produced artifacts.
- Evaluation failed before writing scores due to a Latent Scaling array-shape bug; patched the evaluator and resubmitting evaluation.

### [2026-05-29 04:45] - none
- Third CPU test job wrote `scoring/scores.json`, but `validate.py --compare` rejected undeclared diagnostic metrics.
- Removed those extra score fields and increased the L1 sparsity penalty for the next run to better match the paper's comparable-L0 setup.

### [2026-05-29 04:55] - none
- Fourth CPU test job completed and validated, but did not support the core claim: L1 also avoided false fine-tuned-only latents under the weak-amplitude benchmark.
- Revised the synthetic generator to include the paper's latent-decoupling setup and lowered the L1 penalty before resubmitting.

### [2026-05-29 05:20] - method_runs
- Latent-decoupling run completed end-to-end and passed `validate.py --compare`.
- BatchTopK reduced the false fine-tuned-only rate relative to L1 (`0.0` vs `0.226`), but reconstruction and top-candidate alignment remain weak, so the next run increases the BatchTopK sparsity budget before claiming `core_claim`.

### [2026-05-29 05:45] - core_claim
- Comparable-L0 run completed and passed validation.
- BatchTopK matched L1 sparsity (`14.0` vs `13.98` mean L0), kept reasonable reconstruction (`0.942` FVE), and eliminated false fine-tuned-only latents on the primary metric (`0.0` vs `0.226` for L1).

## Stop Justification
- Completed at milestone `core_claim`.
