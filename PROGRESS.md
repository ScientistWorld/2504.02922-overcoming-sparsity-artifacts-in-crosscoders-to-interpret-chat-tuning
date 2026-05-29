# Progress

## What Works

- Implemented a scaled paired-activation benchmark with shared, weak-shared, chat-only, and base-only latent factors.
- Implemented the paper's shared-encoder/two-decoder crosscoder architecture.
- Implemented both L1 and BatchTopK sparsity objectives.
- Implemented method-agnostic evaluation for decoder norm ranking, Latent Scaling specificity, reconstruction FVE, and L0.
- Added a Table-1-style false fine-tuned-only rate (`false_ft_only_rate_pi06`) so the scaled benchmark explicitly measures the paper's artifact-control constraint.
- `python validate.py --reference-only` passes after staging the implementation files.

## Results

- No compute-node results yet. A first job is being submitted to train both crosscoders and write `scoring/scores.json`.

## Remaining

- Run the method in the compute container.
- Inspect `scoring/scores.json` and iterate on hyperparameters if BatchTopK does not improve specificity over L1.
- Update milestone after job results are available.

## Issues

- Login-node Python lacks NumPy, so only syntax checks and structural validation were run locally. Full execution is deferred to the compute container where `environment/setup.sh` installs dependencies.

## Deviations from Paper

- Original paper: Gemma 2 2B base and Gemma 2 2B-it residual-stream activations at layer 13, trained on 100M tokens from web and chat data.
- Reproduction: synthetic paired activations generated from sparse latent factors with known shared and model-specific structure.
- Reason: full model activation extraction and 73,728-latent crosscoder training would exceed the practical first-pass budget. The implemented computation remains the paper's crosscoder algorithm and Latent Scaling diagnostic.
