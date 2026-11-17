# Experiments

Both experiments measure discovery of fine-tuning-specific structure while preserving reconstruction under fixed capacity and activity budgets. Each has a public development world and an independent held-out world. Methods learn from unlabeled activations; the evaluator generates new inputs, computes outputs from submitted procedures, and judges against its own ground truth. The 32-factor discovery budget is fixed for every method.

## synthetic_chat_specificity (weight 0.6)

The world contains genuinely specific concepts alongside shared concepts that are weaker in one model or occur on partly different inputs. The experiment measures false discoveries, coverage of the truly specific structures, and explained input variance. It asks whether discoveries are useful and trustworthy without sacrificing reconstruction or sparsity.

## synthetic_faint_chat_concepts (weight 0.4)

The same measurements on denser inputs with fine-tuned-only signal strength 0.25–0.65 of ordinary signal strength. Useful changes carry little variance here. This checks whether a method finds subtle structure instead of spending its budget only on dominant shared signals. False discoveries have half the main experiment's coefficient because recovery is the additional challenge; reconstruction remains a constraint and a scored quantity.

Every experiment reports `false_ft_only_rate` (lower), `chat_specific_recall` (higher), and `reconstruction_fve` (higher). A failure in either experiment gives the worst values in both. The detailed measurement and portable submission interface are in `briefing/evaluation.md`.
