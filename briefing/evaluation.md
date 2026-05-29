# Evaluation

Evaluation reads saved method artifacts and writes `scoring/scores.json`. The suite entry point is:

```bash
bash scripts/evaluate.sh --suite
```

For each benchmark case and method, the evaluator expects a `val_outputs.npz` artifact containing validation-set candidate activations, reconstructed base and fine-tuned activation streams, and one vector per candidate for each stream. This is an output contract only; it does not prescribe how the method must train or infer those quantities.

The suite contains two experiments:

- `synthetic_chat_specificity`: tests whether a method separates genuinely fine-tuned-specific factors from factors present in both activation streams.
- `synthetic_complete_shrinkage`: tests whether a method avoids marking lower-amplitude shared factors as fine-tuned-specific.

The primary validator-facing metric is `reconstruction_fve`, because it is directly comparable between the paper's full activation study and this scaled reproduction. The central scientific target is `false_ft_only_rate_pi06`: a lower value means fewer apparently fine-tuned-only candidates fail an independent specificity check. The score also includes candidate purity, shared-factor false-positive rates, reconstruction quality, and sparsity constraints so that a method cannot win by ignoring reconstruction or activating many more candidates.

Metrics include:

- `fraction_top_norm_scaling_specific`: fraction of top candidate-score items that pass the independent specificity check.
- `fraction_top_norm_true_chat`: fraction of top candidate-score items aligned to true fine-tuned-specific factors.
- `fraction_top_norm_weak_or_shared`: fraction of top candidate-score items aligned to shared-factor families.
- `chat_only_latent_count`: number of candidates classified as fine-tuned-only by the evaluator's candidate score.
- `false_ft_only_rate_pi06`: false-positive rate among candidates classified as fine-tuned-only.
- `reconstruction_fve`, `base_reconstruction_fve`, `chat_reconstruction_fve`: fraction of activation variance explained by the submitted reconstructions.
- `mean_l0`: average number of active candidates per validation sample.
