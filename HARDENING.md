# Final hardening (measurements pending)

The prior report is preserved in `hardening/PRIOR_HARDENING.md`; its original scores and seed/rank audit are in `hardening/before/`. `REVIEW.md` correctly identified that the previous setup retained every decoupled concept as model-only twins, despite the improvement in its aggregate false rate. The targeted Latent Scaling decoder correction had also never been tested.

All fixes were decided together before a single consolidated audit job. The final results and the measured limitations will replace this pending report after that job.

Changes submitted together:

- Increase shared occurrence of partially separated concepts from 0.04 to 0.12 and reduce each exclusive occurrence from 0.03 to 0.012, giving coupling the activity savings described by the paper. Increase dictionary capacity from 256 to 512, reduce k from 8 to 6, anneal from 24, double reference steps to 32,000 and training/evaluation pairs to 65,536. Keep the paper's weighted BatchTopK, auxiliary dead-latent loss, global inference threshold, and matched-sparsity L1 procedure.
- Use raw-activation Latent Scaling to select a fixed 32 discoveries and judge correctness against hidden ground truth. Submitted decoder norms no longer set discovery labels; measured contribution mass is invariant to positive feature and decoder rescaling. Duplicate claims cannot dilute false discoveries. The paper's norm-based and residual-scaling measurements remain diagnostics, including explicit shared-factor/twin counts.
- Require useful reconstruction in both streams and nonempty useful recovery. A failure anywhere assigns worst values in all experiments. All failed/invalid outputs have complete finite metrics.
- Allow optional ONNX tensor-program inference, receiving only input activations, to express honest procedures such as support-constrained amplitude optimisation. No arbitrary method Python executes with access to answers. Preserve independent train/test implementations and evaluator-frozen comparisons.
- Keep the 0.6/0.4 experiment weights and coefficient magnitudes unchanged. Correct audit normalization to use every ranked audit attempt, including failures and shortcuts. Preserve paper results unchanged under underscore-prefixed metadata, with fresh measured gym results as the actual reference.

The job first measures minimal and full targeted norm correction on the old saved artifacts, then builds final data and runs the final reference and L1 baseline sequentially, three paired training seeds, longer and shorter training, amplitude fitting, PCA/exemplar controls and affected shortcuts. Four additional paired held-out input batches measure sampling noise. It exercises both no-argument entry points, then removes audit-only methods from the published scores and validates the workspace.

No success, headroom, mechanism claim, or shortcut closure is inferred from the pending code alone. Real Gemma activations, causal KL patching and interpretability studies remain outside this scaled reproduction.
