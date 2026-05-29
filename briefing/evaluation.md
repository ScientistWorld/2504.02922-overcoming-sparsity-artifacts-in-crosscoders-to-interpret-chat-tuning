# Evaluation

Evaluation reads saved method artifacts from `checkpoints/crosscoder_suite/<case>/<method>/val_outputs.npz` and writes `scoring/scores.json`. The legacy single-case evaluator still accepts `checkpoints/crosscoder_synthetic/<method>/val_outputs.npz`.

The validator-facing `primary_metric` is `false_ft_only_rate_pi06`, matching the paper's central artifact-control claim. Reconstruction FVE and mean L0 are constraint metrics so a method cannot improve apparent specificity by failing to reconstruct or by using much denser activations.

The suite contains two experiments:

- `synthetic_chat_specificity`: latent-decoupling setting where a shared concept has shared and model-exclusive activation subsets.
- `synthetic_complete_shrinkage`: complete-shrinkage setting where a concept is present in both streams but lower-amplitude in the base stream.

Both experiments rank latents by relative decoder norm difference and measure whether the top chat-specific candidates align with ground-truth chat-only factors rather than weak/shared factors. They also compute Latent Scaling ratios to test whether top candidates behave like genuinely chat-specific latents.

Metrics include:

- `fraction_top_norm_scaling_specific`: fraction of top norm-ranked candidates passing Latent Scaling specificity thresholds.
- `fraction_top_norm_true_chat`: fraction of top candidates aligned to true chat-only factors.
- `fraction_top_norm_weak_or_shared`: false-positive fraction aligned to shared or decoupled shared factors.
- `chat_only_latent_count`: number of latents classified chat-only by decoder norm.
- `reconstruction_fve`: fraction of activation variance explained by reconstructions.
- `mean_l0`: average number of active latents per validation sample.
