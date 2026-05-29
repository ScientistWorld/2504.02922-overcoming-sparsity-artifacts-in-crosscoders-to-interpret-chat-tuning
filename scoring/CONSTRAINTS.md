# Scientist Constraints

Future agents should keep the paired synthetic activation dataset fixed when comparing scores. They may improve the crosscoder implementation, optimization, or diagnostics, but should not change the validation split, ground-truth concept taxonomy, or metric definitions without creating a new experiment name.

The evaluation must read saved artifacts from `checkpoints/crosscoder_synthetic/<method>/val_outputs.npz`; it must not regenerate method outputs inside `eval/`. The core constraint is to identify genuinely model-specific factors while retaining sparse, reconstructive paired activations.

## Compliance Checklist

- [ ] The validation split in `data/synthetic_activations.npz` is unchanged for reported `synthetic_chat_specificity` scores.
- [ ] The metric definitions in `eval/evaluate_crosscoders.py` are unchanged for reported comparisons.
- [ ] Evaluation reads method artifacts instead of rerunning training or generating method outputs.
- [ ] Reported methods use paired sparse dictionary decompositions of base/chat activations.
