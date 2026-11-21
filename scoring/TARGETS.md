# Evaluation Targets

`bash scripts/evaluate_test.sh` scores all methods on hidden inputs; `bash scripts/evaluate_train.sh` applies the same measurements to development worlds. Both take no arguments. Measured reference values are in `scores_test.json` and `scores_train.json`; `reference.json` preserves published values separately as underscore-prefixed context.

## synthetic_chat_specificity

Weight 0.6. Benefit: `chat_specific_recall` (higher, coefficient magnitude 1). Specificity constraint and benefit: `false_ft_only_rate` (lower, magnitude 1). Reconstruction constraint and benefit: `reconstruction_fve` (higher, magnitude 0.5).

## synthetic_faint_chat_concepts

Weight 0.4. Benefit: `chat_specific_recall` (higher, magnitude 1). Specificity constraint and benefit: `false_ft_only_rate` (lower, magnitude 0.5). Reconstruction constraint and benefit: `reconstruction_fve` (higher, magnitude 0.5).

Discovery requires measured model presence below 0.2 and firing on at least 0.5% of scored inputs. Centered feature/input covariance defines presence and contribution mass independently of submitted decoder rotation or magnitude; decoder alignment still determines correctness. Each of the 32 target structures receives credit once; duplicate contribution counts as false discovery. Empty discovery has false rate 1. Both experiments require dictionary size <= 512, mean L0 <= 6.6, each stream's FVE >= 0.60, joint FVE >= 0.60, and recall >= 0.10. Invalid or missing outputs, or any failed constraint, give false rate 1, recall 0, FVE 0 in every experiment. Success therefore dominates failure on all scored quantities.
