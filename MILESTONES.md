# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-05-29 05:45] - method_runs
- Verified the existing implementation runs structurally and passes `python3 validate.py --reference-only`.
- Prepared a CPU compute job to sweep BatchTopK settings and select a stronger artifact set before promoting the milestone.

### [2026-05-29 07:10] - core_claim
- CPU job `7ebb562b-51d` completed successfully and passed `python3 validate.py --compare`.
- The selected real training artifacts reproduce the core artifact-control claim on the scaled latent-decoupling benchmark: L1 false fine-tuned-only rate is `0.194`, while BatchTopK reduces it to `0.0` with sparse activations and high reconstruction FVE.

## Stop Justification
- Completed at milestone `core_claim`. Higher milestones would require additional paper axes such as more synthetic regimes, causal patching experiments, or full Gemma activation extraction.
