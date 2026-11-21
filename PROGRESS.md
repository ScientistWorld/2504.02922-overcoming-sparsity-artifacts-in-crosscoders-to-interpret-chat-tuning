# Current status — 2026-11-21

Final hardening is complete. Job `caf0ab6e-583` exited 0 after rescoring the saved final training outputs. All 48 fits came from the final training code and data; reference and baseline fits were run sequentially. Evaluator-only follow-ups reused these fits.

- The paper's mechanism reproduces on both hidden worlds at all three training seeds: BatchTopK couples 16/16 partially separated concepts with zero spurious twins. L1 retains 14–16 twins in the main world and 15–16 in the faint world.
- Support-constrained amplitude fitting beats the reference on all five common input batches and all three matched training seeds. Mean paired rank gain 0.007865, SD 0.001472, 95% lower bound 0.006038. Longer training and the other iterative variants do not establish headroom.
- Decoder norm edits and the closed-form decoder refit no longer improve the hidden score. Invalid, missing, zero, dense, PCA and exemplar controls receive complete worst-valued results where they fail the task. No-argument discovery includes unlisted methods and propagates a missing experiment's failure to both experiments.
- Both independent scorers agree on their shared-input consistency check; streamed sufficient statistics agree with direct computation. Fresh uncached reference-only entry points exactly match the audit. Canonical train/test score files contain only frozen reference and baseline results.
- `python3 validate.py` passed. `scripts/run.sh` again reproduces data, reference, baseline and both score files end to end. Paper-reported metadata and experiment weights are unchanged.

Current milestone: `core_claim`, restricted to the scaled synthetic setting. Real-model and causal results remain unimplemented. `HARDENING.md` records before/after results, shortcut coverage, uncertainty and remaining limitations. Final machine-readable evidence is in `hardening/final_results/`. Earlier progress is preserved in `hardening/PROGRESS_HISTORY.md`; intermediate measurement changes and rejected headroom claims are in `hardening/MEASUREMENT_HISTORY.md`.
