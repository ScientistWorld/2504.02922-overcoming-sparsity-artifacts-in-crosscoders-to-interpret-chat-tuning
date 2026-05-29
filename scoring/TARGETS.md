# Evaluation Targets

Future scientist agents are scored on the experiments present in `scoring/scores.json`.
Each experiment combines benefit metrics with constraint metrics so improvements must
separate model-specific factors while preserving reconstruction and sparsity.

## synthetic_chat_specificity

Tests whether submitted candidate factors distinguish genuinely fine-tuned-model-specific
activation structure from factors present in both paired activation streams.

Primary scientific targets:

- `false_ft_only_rate_pi06` lower is better: fewer reported fine-tuned-specific candidates fail the independent specificity check.
- `fraction_top_norm_scaling_specific` higher is better: more top-ranked candidates pass the independent specificity check.
- `fraction_top_norm_true_chat` higher is better: more top-ranked candidates align to held-out fine-tuned-specific ground truth.

Constraint targets:

- `fraction_top_norm_weak_or_shared` lower is better: top-ranked candidates should not align to shared-factor families.
- `reconstruction_fve`, `base_reconstruction_fve`, and `chat_reconstruction_fve` higher are better: the submitted artifacts should reconstruct both streams.
- `mean_l0` lower is better: gains should not come from activating many more candidates per sample.
- `chat_only_latent_count` lower is better: gains should not come from over-labeling candidates as fine-tuned-specific.

Run:

```bash
bash scripts/evaluate.sh --suite
```

## synthetic_complete_shrinkage

Tests whether submitted candidate factors avoid marking lower-amplitude shared
activation structure as fine-tuned-model-specific.

Primary scientific targets:

- `false_ft_only_rate_pi06` lower is better.
- `fraction_top_norm_scaling_specific` higher is better.
- `fraction_top_norm_true_chat` higher is better.

Constraint targets:

- `fraction_top_norm_weak_or_shared` lower is better.
- `reconstruction_fve`, `base_reconstruction_fve`, and `chat_reconstruction_fve` higher are better.
- `mean_l0` lower is better.
- `chat_only_latent_count` lower is better.

Run:

```bash
bash scripts/evaluate.sh --suite
```
