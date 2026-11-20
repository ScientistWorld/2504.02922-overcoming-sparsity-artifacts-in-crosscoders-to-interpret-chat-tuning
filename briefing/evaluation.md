# Evaluation

A submission is a directory `checkpoints/methods/<name>/`. Each experiment needs `dev/crosscoder.npz` and `heldout/crosscoder.npz`, trained on the corresponding `data/worlds/<experiment>/<world>/train.npz`. Use a new name: `batchtopk` and `l1` identify evaluator-frozen comparison artifacts on both evaluation sides.

## Artifact interface

The NPZ contains finite numeric arrays; d = 64 and D is the number of factors:

| Array | Shape | Meaning |
|---|---|---|
| `W_enc` | (2d, D) | base rows followed by chat rows |
| `b_enc`, `threshold` | (D,) | bias and activation threshold |
| `W_dec_base`, `W_dec_chat` | (D, d) | reconstruction directions |
| `b_dec_base`, `b_dec_chat` | (d,) | reconstruction biases |
| optional `topk` | integer scalar | positive values retain at most this many factors per input, ordered by activation times the sum of decoder norms |

By default `p = base @ W_enc[:d] + chat @ W_enc[d:] + b_enc`; a feature equals p where p > 0 and p > threshold, and zero elsewhere.

Alternatively, place `encoder.onnx` beside the NPZ to supply a different inference procedure. It receives one float32 tensor `(batch, 128)` containing base then chat activations and returns one finite nonnegative `(batch, D)` feature tensor. The evaluator invokes it in batches of 1,024 with ONNX Runtime's standard CPU operators. Constants must be embedded, external tensor data is forbidden, and the file must be at most 64 MiB. The required NPZ arrays still define D and the decoders; `W_enc`, `b_enc` and `threshold` are ignored in this mode. Optional `topk` is applied after either encoder. This tensor program receives only input activations, with no files, concept labels or evaluation statistics.

For both interfaces, reconstructions are always recomputed as `features @ W_dec_stream + b_dec_stream`. Submitted feature arrays, reconstructions, scores, labels and flags are never read.

## Measurements

Each experiment uses 524,288 fresh paired inputs, fixed by the evaluator and common to every method. Public development and hidden worlds have the same difficulty and measurement but independent concepts and samples.

There are 32 targets: fine-tuned-only concepts and fine-tuned-exclusive occurrences of partially separated concepts. A factor is eligible after firing on at least 80 inputs with nonzero chat direction. Its measured model presence is `max(sum(feature * dot(base, unit_chat)), 0) / sum(feature * dot(chat, unit_chat))`; nonpositive chat denominators are ineligible. Select the 32 eligible factors with lowest presence, breaking ties by greater measured chat-contribution mass. This operating point is fixed; decoder magnitudes cannot set the selection threshold.

Correctness requires chat-direction cosine at least 0.8 with its best-matching concept and activity-weighted ground-truth base/chat concept presence below 0.2. Each target receives credit once. The measured chat-contribution mass is mean(feature) times `max(sum(feature * dot(chat, unit_chat)), 0) / sum(feature²)`. It is invariant to positive rescaling of the factor or its decoder magnitude.

- `false_ft_only_rate`: fraction of selected mass without a credited target, including duplicates; lower is better. Empty discovery has value 1. This historical key now measures fixed-budget false discovery rather than an editable norm label.
- `chat_specific_recall`: fraction of the 32 targets credited, higher is better.
- `reconstruction_fve`: fraction of joint input variance explained by recomputed reconstructions, higher is better.

Decoder-norm labels and per-family recovery remain unscored mechanism diagnostics. No metric measures agreement with a published number or elapsed time.

## Constraints and failures

D must be between 1 and 512, mean active factors must not exceed 6.6, joint and each stream's FVE must be at least 0.60, and recall must be at least 0.10. Every artifact, feature, reconstruction and scored number must be finite and valid. A failure in either experiment assigns **every metric of both experiments** its worst value: false rate 1, recall 0 and FVE 0. Missing artifacts cannot omit losses. Unscored diagnostics retain evaluator-computed raw measurements for scientific inspection even when a constraint fails; they never substitute for ranked results.

## Entry points and rank

All commands take no arguments and discover every method directory on disk, independently of existing score files:

- `bash scripts/evaluate_train.sh` writes `scoring/scores_train.json` on public dev worlds.
- `bash scripts/evaluate_test.sh` writes `scoring/scores_test.json` on hidden held-out worlds.
- `bash scripts/evaluate.sh` writes `scoring/scores.json`, equivalent to the train side.

Only numbers under each experiment's `results` are ranked. Each metric contributes experiment weight times the absolute coefficient in `scoring/reference.json`, times improvement in its stated better direction, divided by its population standard deviation across attempts being ranked. A zero standard deviation contributes no distinction (the audit uses denominator 1). Rank values in the hardening audit use its complete cohort, including failed attempts; future cohort normalization changes their magnitudes.
