# Scientist Constraints

- Train only on the provided activations: `data/worlds/<experiment>/{dev,heldout}/train.npz`. Do not modify these files or the dev worlds when reporting results. The hidden test samples its own inputs, so edits only change your own training data.
- Submit one dictionary per experiment and world in the documented format (`briefing/evaluation.md`). Features are always computed by the evaluator from the submitted encoder, thresholds and optional `topk`, so inference must be expressible in that form.
- Fixed budgets: at most 256 factors and a mean of at most 8.8 active factors per input. Exceeding either fails the experiment.
- The fine-tuned-only label is defined by the evaluator from the decoder norms (relative norm difference ≥ 0.9). There is no way to submit labels directly.
- Do not use the dev worlds' ground truth (`world.npz`, `world.json`) inside the method that produces held-out dictionaries. Use it only to diagnose and develop. Dictionaries tuned to the dev world do not transfer: the held-out worlds have different concepts and family sizes.
- `batchtopk` and `l1` are reserved method names. Use new names for new methods.

## Compliance Checklist

- [ ] Held-out dictionaries were trained only on `data/worlds/<experiment>/heldout/train.npz`.
- [ ] The artifacts respect the 256-factor and L0 ≤ 8.8 budgets.
- [ ] `eval/` and the provided data were not modified.
- [ ] Both experiments have artifacts for both worlds.
