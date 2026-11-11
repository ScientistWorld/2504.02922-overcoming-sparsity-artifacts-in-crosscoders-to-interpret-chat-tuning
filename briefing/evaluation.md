# Evaluation

## Output contract

A method is a directory `checkpoints/methods/<method>/`. For every experiment and both worlds it contains one dictionary trained on that world's training activations:

```
checkpoints/methods/<method>/<experiment>/dev/crosscoder.npz      # trained on data/worlds/<experiment>/dev/train.npz
checkpoints/methods/<method>/<experiment>/heldout/crosscoder.npz  # trained on data/worlds/<experiment>/heldout/train.npz
```

`crosscoder.npz` holds float arrays (d = 64, D = number of factors):

| key | shape | meaning |
|---|---|---|
| `W_enc` | (2d, D) | encoder; rows 0..d-1 act on the base stream, rows d..2d-1 on the chat stream |
| `b_enc` | (D,) | encoder bias |
| `threshold` | (D,) | per-factor activation threshold |
| `W_dec_base`, `W_dec_chat` | (D, d) | each factor's direction in the base / chat stream |
| `b_dec_base`, `b_dec_chat` | (d,) | decoder biases |
| `topk` (optional) | scalar int | if > 0, keep only the `topk` largest `a_j * (‖W_dec_base_j‖ + ‖W_dec_chat_j‖)` per input |

The evaluator computes everything itself on inputs it samples:
`pre = h_base @ W_enc[:d] + h_chat @ W_enc[d:] + b_enc`, then `a_j = pre_j` if `pre_j > 0` and `pre_j > threshold_j`, otherwise 0. The reconstructions are `a @ W_dec_base + b_dec_base` and `a @ W_dec_chat + b_dec_chat`. Nothing a method writes besides these parameters is read.

## Fixed constraints (failure = worst value on every metric of that experiment)

- the artifact must exist, load without pickle, have the shapes above and contain only finite values;
- dictionary size D ≤ 256;
- mean number of active factors per input (L0) ≤ 8.8 on the scored inputs.

## Metrics (per experiment)

A factor is labelled **fine-tuned-only** when its relative decoder norm difference `0.5 * (1 + (‖d_chat‖ − ‖d_base‖) / max(‖d_chat‖, ‖d_base‖))` is ≥ 0.9. A factor is alive if it fires on at least 10 scored inputs. A factor is **genuinely fine-tuned-specific** when it meets two conditions:

- its chat direction has cosine ≥ 0.8 with a ground-truth concept;
- that concept's base-to-chat presence ratio, weighted by the factor's own activations, is below 0.2.

- `false_ft_only_rate` (lower is better): the activation-mass-weighted fraction of alive fine-tuned-only factors that are not genuinely fine-tuned-specific. A factor's mass is its mean activation times ‖d_chat‖. If no factor is labelled fine-tuned-only, the rate is 1.0.
- `chat_specific_recall` (higher is better): the fraction of the world's fine-tuned-specific targets recovered by a genuinely specific factor. The targets are its fine-tuned-only concepts and the fine-tuned-exclusive uses of its decoupled concepts. Only the top 1.5 × (number of targets) alive factors, ranked by relative decoder norm difference, are considered.
- `reconstruction_fve` (higher is better): fraction of the variance of both streams explained by the evaluator-computed reconstruction, clipped to [0, 1].

## Entry points (no arguments)

- `bash scripts/evaluate_train.sh` → `scoring/scores_train.json`: dev dictionaries scored on fresh inputs from the public dev worlds.
- `bash scripts/evaluate.sh` → `scoring/scores.json`: same as the train side.
- `bash scripts/evaluate_test.sh` → `scoring/scores_test.json` (hidden): held-out dictionaries scored on fresh inputs from the hidden held-out worlds.

Every directory under `checkpoints/methods/` is scored. A method missing an experiment's artifact gets the worst values there. The names `batchtopk` and `l1` are reserved for the frozen reference and baseline: the hidden test always scores the copies stored with the evaluator under those names. Unscored diagnostics (L0, alive factors, per-family recall, spurious-mass breakdown, per-stream FVE) are written next to the scores.
