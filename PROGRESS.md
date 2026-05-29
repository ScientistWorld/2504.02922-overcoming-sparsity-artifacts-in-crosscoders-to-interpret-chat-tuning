# Progress

## What Works

- Implemented a scaled paired-activation benchmark with shared, weak-shared, chat-only, and base-only latent factors.
- Implemented the paper's shared-encoder/two-decoder crosscoder architecture.
- Implemented both L1 and BatchTopK sparsity objectives.
- Implemented method-agnostic evaluation for decoder norm ranking, Latent Scaling specificity, reconstruction FVE, and L0.
- Added a Table-1-style false fine-tuned-only rate (`false_ft_only_rate_pi06`) so the scaled benchmark explicitly measures the paper's artifact-control constraint.
- `python3 validate.py --reference-only` passes after staging the implementation files.

## Results

- CPU job `26a2e825-7e8` completed training and wrote crosscoder artifacts under `checkpoints/crosscoder_synthetic/`, but failed while computing Latent Scaling metrics.
- No valid `scoring/scores.json` yet. The next job reuses the completed training artifacts if present and reruns evaluation after the shape bug fix.

## Remaining

- Run the method in the compute container.
- Inspect `scoring/scores.json` and iterate on hyperparameters if BatchTopK does not improve specificity over L1.
- Update milestone after job results are available.

## Issues

- First CPU test job failed before method execution because the base container exposed `python3` but not `python`. Scripts now call `python3` explicitly, and `environment/container.def` also creates a `python` symlink for compatibility.
- Second CPU test job trained both crosscoders, then failed in `eval/evaluate_crosscoders.py` because NumPy batched matmul did not express the intended per-latent dot product. This is fixed with explicit `einsum`.
- Login-node Python lacks NumPy, so only syntax checks and structural validation were run locally. Full execution is deferred to the compute container where `environment/setup.sh` installs dependencies.

## Deviations from Paper

- Original paper: Gemma 2 2B base and Gemma 2 2B-it residual-stream activations at layer 13, trained on 100M tokens from web and chat data.
- Reproduction: synthetic paired activations generated from sparse latent factors with known shared and model-specific structure.
- Reason: full model activation extraction and 73,728-latent crosscoder training would exceed the practical first-pass budget. The implemented computation remains the paper's crosscoder algorithm and Latent Scaling diagnostic.
