# Scientist Constraints

- Learn held-out dictionaries from the provided unlabeled held-out training activations. Public dev ground truth may inform development, but does not provide supervision for held-out concepts.
- Do not edit supplied data, frozen comparison artifacts, evaluators, scoring definitions or hidden assets to improve reported results.
- Submit both experiments through the same documented NPZ/optional tensor-program interface. Use a new method name. The hidden test reserves `batchtopk` and `l1` for frozen comparisons.
- Every number is recomputed from raw outputs on evaluator-fixed inputs. Ground truth, operating points, thresholds, test size and budgets belong to the evaluator. Method-supplied metric files and labels are ignored.
- Fixed budgets: <=512 factors and mean L0 <=6.6. Joint and individual stream FVE must be >=0.60; recall must be >=0.10 in both experiments. Features must be finite and nonnegative. A failure gives worst values for all metrics in every experiment.
- An optional ONNX tensor program may implement alternative inference, with embedded constants and standard operators; it receives only the paired activations. External tensor files are forbidden. See `briefing/evaluation.md`.
- Do not access hidden concept answers, reverse engineer test seeds, or build dictionaries from hidden test inputs or labels. Test scoring never trains the method.
