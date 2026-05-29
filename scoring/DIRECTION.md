# Research Direction

## Research Problem

This benchmark studies model diffing: identifying which activation factors are genuinely specific to a fine-tuned model rather than shared with its base model.

## Core Direction

Improve paired sparse factor discovery so that shared activation factors are not mistaken for fine-tuned-specific factors.

## Approach Scope

Future work should remain within sparse dictionary learning over paired base/fine-tuned activations. Valid changes include better sparsity controls, initialization, optimization, candidate ranking, and artifact diagnostics.

## Out of Bounds

Do not replace the task with supervised classification over ground-truth concept labels. A valid method must produce candidate factors with per-stream reconstruction artifacts in the documented output format, so the evaluator can test both specificity and reconstruction. Do not change the data-generating process and claim direct improvement on the same experiment.
