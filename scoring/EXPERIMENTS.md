# Experiment Overview

Both experiments are scored on hidden held-out worlds. You see only those worlds' unlabeled training activations. The public dev worlds of the same generator family are for development and for `scores_train.json`. All scores are computed by the evaluator from the submitted dictionary parameters, on inputs it samples itself.

## synthetic_chat_specificity (weight 0.6)

Establishes whether factors labelled fine-tuned-only are genuinely specific to the fine-tuned model. The world mixes genuine fine-tuned-only concepts with two kinds of shared concepts that are easy to mistake for them: concepts weaker in the base stream, and concepts both models use on partly different inputs. The experiment asks three things:

- whether the fine-tuned-only label is trustworthy (false fine-tuned-only rate);
- whether the top of the fine-tuned-only ranking recovers the genuinely fine-tuned-specific structure (recall);
- whether this comes without giving up reconstruction (FVE).

Sparsity and dictionary size are fixed budgets.

## synthetic_faint_chat_concepts (weight 0.4)

The same question in a denser world whose fine-tuned-only concepts are faint (0.15–0.45 of normal strength). It establishes whether a method can recover fine-tuned-specific structure that carries little variance, under the same sparsity budget, while keeping its fine-tuned-only labels trustworthy. Recall weighs most here; the false fine-tuned-only rate has half weight because its spurious mass mostly repeats what `synthetic_chat_specificity` already measures.
