# Research Direction

## Research Problem

Model diffing: learn a sparse paired dictionary of base/fine-tuned activations whose fine-tuned-only factors are genuinely specific to the fine-tuned model.

## Core Direction

Improve sparse paired dictionary learning so that:

- shared concepts are not labelled fine-tuned-only, whether they are weaker in one model or used on partly different inputs;
- genuinely fine-tuned-specific structure, including faint structure, is recovered;
- reconstruction stays high at a fixed sparsity and dictionary size.

## Approach Scope

Anything that produces a sparse paired dictionary through the documented NPZ/optional tensor-program interface is in scope. Alternative inference procedures and learned encoders are allowed. This includes:

- sparsity mechanisms, selection rules, auxiliary losses, initialisation, schedules and training budgets;
- dictionary partitioning (shared vs model-specific factors);
- diagnostics that correct the dictionary using the training activations.

## Out of Bounds

- Supervision from ground-truth concept labels.
- Dictionaries derived from the public dev worlds' ground truth and submitted for the held-out worlds.
- Editing the provided data or the evaluator.
- Exceeding the fixed sparsity or dictionary-size budgets.
