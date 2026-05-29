# Evaluation Targets

## Primary: synthetic_chat_specificity

Tests whether candidates identified as fine-tuned-model-specific are genuinely specific rather than shared factors misclassified by the scoring procedure.

- **Metric**: `false_ft_only_rate_pi06` (lower is better)
- **Metric**: `fraction_top_norm_scaling_specific` (higher is better)
- **Metric**: `fraction_top_norm_true_chat` (higher is better)
- **Run**: `bash scripts/evaluate.sh --suite checkpoints/crosscoder_suite data/benchmarks`

## Primary: synthetic_complete_shrinkage

Tests the complete-shrinkage failure mode, where a lower-amplitude shared factor can be treated as fine-tuned-only by a naive specificity score.

- **Metric**: `false_ft_only_rate_pi06` (lower is better)
- **Metric**: `fraction_top_norm_scaling_specific` (higher is better)
- **Metric**: `fraction_top_norm_true_chat` (higher is better)
- **Run**: `bash scripts/evaluate.sh --suite checkpoints/crosscoder_suite data/benchmarks`

## Constraint: shared_concept_false_positives

The method should not improve apparent model-specificity by labeling decoupled shared factors as fine-tuned-only.

- **Metric**: `fraction_top_norm_weak_or_shared` (lower is better)
- **Run**: `bash scripts/evaluate.sh --suite checkpoints/crosscoder_suite data/benchmarks`

## Constraint: reconstruction_and_sparsity

The submitted representation should still reconstruct paired activations with sparse candidate activations, so improvements cannot come from ignoring reconstruction quality or by activating many more candidates.

- **Metric**: `reconstruction_fve` (higher is better)
- **Metric**: `mean_l0` (lower is better)
- **Metric**: `chat_only_latent_count` (lower is better as an artifact-control proxy)
- **Run**: `bash scripts/evaluate.sh --suite checkpoints/crosscoder_suite data/benchmarks`
