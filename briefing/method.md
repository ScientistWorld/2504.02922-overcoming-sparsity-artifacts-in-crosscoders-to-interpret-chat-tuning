# Method

The paper trains crosscoders on paired base/chat residual-stream activations. A shared encoder maps concatenated base and chat activations into nonnegative sparse latent activations:

`f_j(x) = ReLU(e_j_base h_base(x) + e_j_chat h_chat(x) + b_j)`.

The decoder reconstructs both models with separate decoder vectors per latent:

`h_base_recon = sum_j f_j d_j_base + b_base`

`h_chat_recon = sum_j f_j d_j_chat + b_chat`.

This reproduction implements both sparsity mechanisms from the paper:

- L1 crosscoder: reconstruction loss plus `mu * sum_j f_j (||d_j_base|| + ||d_j_chat||)`.
- BatchTopK crosscoder: select the top `batch_size * k` scaled activations `f_j (||d_j_base|| + ||d_j_chat||)` across each batch and train reconstruction loss on those sparse activations. At inference, estimate the paper's global threshold `theta` from training batches and keep activations whose scaled value exceeds that threshold.

The encoder and both model decoders are initialized from paired latent directions with the L1 norm scale set to `0.05` and the BatchTopK norm scale set to `1.0`, matching the paper's initialization pattern at reduced width.

The evaluator implements the paper's decoder norm difference and Latent Scaling diagnostics. Latent Scaling fits a scalar beta for each latent direction against base/chat reconstruction and reconstruction-error targets, then uses ratio thresholds to identify true model-specific latents.

Deviation: the original paper trains on Gemma 2 2B base/chat layer-13 activations from 100M tokens. This reproduction uses synthetic paired activations with known latent ground truth because full activation extraction and crosscoder training are outside the current budget. The algorithmic procedure is preserved at smaller scale.

Deviation: the paper's BatchTopK setup includes an auxiliary loss for inactive latent recycling. This scaled reproduction omits that auxiliary term because the small synthetic dictionary does not exhibit the same dead-latent regime; the core BatchTopK selection and thresholded inference computation are preserved.
