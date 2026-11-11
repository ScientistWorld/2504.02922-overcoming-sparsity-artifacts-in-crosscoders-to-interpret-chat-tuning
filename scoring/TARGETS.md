# Evaluation Targets

Scored by `bash scripts/evaluate_test.sh` (hidden held-out worlds) and, for development, `bash scripts/evaluate_train.sh` (public dev worlds). Both scripts take no arguments and score every directory under `checkpoints/methods/`.

Fixed constraints in every experiment: at most 256 factors, mean L0 ≤ 8.8 on the scored inputs, and a valid finite artifact. Breaking a constraint, or a missing artifact, gives every metric of that experiment its worst value (false rate 1, recall 0, FVE 0).

## synthetic_chat_specificity

- Benefit: `chat_specific_recall`, higher is better (coefficient 1.0).
- Benefit / specificity constraint: `false_ft_only_rate`, lower is better (coefficient 1.0). The rate is 1.0 when nothing is labelled fine-tuned-only.
- Constraint: `reconstruction_fve`, higher is better (coefficient 0.5).

## synthetic_faint_chat_concepts

- Benefit: `chat_specific_recall`, higher is better (coefficient 1.0).
- Specificity constraint: `false_ft_only_rate`, lower is better (coefficient 0.5).
- Constraint: `reconstruction_fve`, higher is better (coefficient 0.5).

Reference (hidden test, BatchTopK crosscoder): 0.572 / 0.50 / 0.981 and 0.912 / 0.25 / 0.979 (false rate / recall / FVE). L1 baseline: 0.701 / 0.50 / 0.818 and 1.000 / 0.00 / 0.699.
