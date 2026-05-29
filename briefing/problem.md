# Problem

The task is model-difference factor discovery. Given paired activation vectors from a base model and a fine-tuned model on the same inputs, a method should produce candidate factors that distinguish what is genuinely specific to the fine-tuned model from what is shared by both models.

The main failure mode is false specificity: a shared factor can look fine-tuned-only when its strength, sparsity, or occurrence pattern differs between the two activation streams. A useful method should surface true fine-tuned-specific factors while avoiding these shared-factor false positives.

This reproduction uses a scaled paired-activation benchmark generated in `data/`. The benchmark contains ground-truth factor families: shared, base-specific, fine-tuned-specific, and shared factors with unequal or partially separated behavior across the two streams. These cases preserve the paper's model-diffing difficulty while avoiding the cost of extracting full large-language-model activations.
