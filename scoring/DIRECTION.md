# Research Direction

## Research Problem

This benchmark studies model diffing: identifying which activation factors are genuinely specific to a fine-tuned model rather than shared with its base model.

## Core Contribution

The paper's contribution is improving crosscoder-based model diffing by diagnosing sparsity artifacts and using a sparsity mechanism that better separates shared and model-specific latents.

## Approach Scope

Future work should remain within crosscoder-style sparse dictionary learning over paired base/fine-tuned activations. Valid changes include better sparsity mechanisms, initialization, optimization, latent ranking, and artifact diagnostics.

## Out of Bounds

Do not replace the task with supervised classification over ground-truth concept labels. Do not use a method that bypasses a shared sparse latent representation with separate model-specific reconstructions. Do not change the data-generating process and claim direct improvement on the same experiment.
