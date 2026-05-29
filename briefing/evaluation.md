# Evaluation

Evaluation reads saved method artifacts from `checkpoints/crosscoder_synthetic/<method>/val_outputs.npz` and writes `scoring/scores.json`.

The validator-facing `primary_metric` is `reconstruction_fve` because it is on the same absolute scale as the paper's reported reconstruction scores. The central artifact-control target remains `false_ft_only_rate_pi06`, which is the highest-weight false-positive metric in `reference.json` and `TARGETS.md`; its absolute value is benchmark-scale dependent in this synthetic reproduction.

The primary experiment is `synthetic_chat_specificity`. It ranks latents by relative decoder norm difference and measures whether the top chat-specific candidates align with ground-truth chat-only factors rather than decoupled shared or shared factors. It also computes Latent Scaling ratios to test whether top candidates behave like genuinely chat-specific latents.

Metrics include:

- `fraction_top_norm_scaling_specific`: fraction of top norm-ranked candidates passing Latent Scaling specificity thresholds.
- `fraction_top_norm_true_chat`: fraction of top candidates aligned to true chat-only factors.
- `fraction_top_norm_weak_or_shared`: false-positive fraction aligned to shared or decoupled shared factors.
- `chat_only_latent_count`: number of latents classified chat-only by decoder norm.
- `reconstruction_fve`: fraction of activation variance explained by reconstructions.
- `mean_l0`: average number of active latents per validation sample.
