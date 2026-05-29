# Evaluation Targets

## Primary: synthetic_chat_specificity

Tests whether latents identified as fine-tuned-model-specific by decoder norm are genuinely specific rather than shared concepts misclassified by the sparsity mechanism.

- **Metric**: `false_ft_only_rate_pi06` (lower is better)
- **Metric**: `fraction_top_norm_scaling_specific` (higher is better)
- **Metric**: `fraction_top_norm_true_chat` (higher is better)
- **Run**: `bash scripts/evaluate.sh checkpoints/crosscoder_synthetic`

## Constraint: shared_concept_false_positives

The method should not improve apparent model-specificity by labeling decoupled shared concepts as fine-tuned-only.

- **Metric**: `fraction_top_norm_weak_or_shared` (lower is better)
- **Run**: `bash scripts/evaluate.sh checkpoints/crosscoder_synthetic`

## Constraint: reconstruction_and_sparsity

The dictionary should still reconstruct paired activations with sparse codes, so improvements cannot come from ignoring reconstruction quality or by activating many more latents.

- **Metric**: `reconstruction_fve` (higher is better)
- **Metric**: `mean_l0` (lower is better)
- **Metric**: `chat_only_latent_count` (lower is better as an artifact-control proxy)
- **Run**: `bash scripts/evaluate.sh checkpoints/crosscoder_synthetic`
