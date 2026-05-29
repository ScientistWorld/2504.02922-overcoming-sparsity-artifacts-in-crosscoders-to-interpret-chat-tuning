# Scientist Constraints

Future agents should keep the paired synthetic activation datasets fixed when comparing scores. They may improve the representation-learning method, optimization, or diagnostics, but should not change the validation splits, ground-truth concept taxonomy, or metric definitions without creating a new experiment name.

The suite evaluation must read saved artifacts from the submitted output directory, organized as `<case>/<method>/val_outputs.npz`; it must not regenerate method outputs inside `eval/`. The core constraint is to identify genuinely model-specific factors while retaining sparse, reconstructive paired activations.

## Compliance Checklist

- [ ] The validation splits in `data/benchmarks/*/synthetic_activations.npz` are unchanged for reported suite scores.
- [ ] The metric definitions in `eval/` are unchanged for reported comparisons.
- [ ] Evaluation reads method artifacts instead of rerunning training or generating method outputs.
- [ ] Reported methods submit paired sparse factor artifacts for base and fine-tuned activations.
