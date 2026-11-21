# Method

The paper trains crosscoders on paired base/chat residual-stream activations. A shared encoder maps the concatenated activations to nonnegative latent activations

`f_j(x) = ReLU(e_j^base h^base(x) + e_j^chat h^chat(x) + b_j)`,

and separate decoders reconstruct each model: `h~^M = sum_j f_j d_j^M + b^M` for M in {base, chat}.

Implemented in `method/` (`crosscoder.py`, `train.py`):

- **L1 crosscoder (baseline, Eq. 8):** loss `½‖ε_base‖² + ½‖ε_chat‖² + μ Σ_j f_j (‖d_j^base‖ + ‖d_j^chat‖)`. The paper chose hyperparameters so that both crosscoders reach the same L0. Here μ is chosen the same way: bisection on each world's training activations to a mean L0 of 6, selected from fully trained candidates using training activations alone.
- **BatchTopK crosscoder (proposed, Eqs. 9–14):**
  - Training keeps the top `batch·k` scaled activations `f_j (‖d_j^base‖ + ‖d_j^chat‖)` across each batch (k = 6, annealed from 24 over the first 12.5% of training using the recommendation in App. I.1).
  - The loss is reconstruction plus the auxiliary dead-latent loss with α = 1/32. That loss reconstructs the detached residual with the top-32 activations of latents that have not fired for 200k inputs, normalised by the residual variance, as in the paper's training code.
  - The inference threshold θ is the mean over training batches of the smallest selected scaled activation (Eq. 14). It is exported as the per-latent threshold `θ / (‖d_j^base‖ + ‖d_j^chat‖)` (Eq. 13).
- **Initialisation (App. K):** decoder = encoder transpose, with base and chat paired with identical weights. Initial latent norm is 1.0 for BatchTopK and 0.05 for L1.
- **Optimisation:** Adam, batch 256, 32,000 steps (125 passes over 65,536 training pairs), lr 1e-3 with linear decay over the final 20%, 512 latents.

Options used by the hardening audit, not by the reference: the alternative selection norms of App. B.3 (`selection_norm = concat | max`); disabling the auxiliary loss.

Scale changes and evaluation departures from the paper:

- Synthetic 64-dimensional paired worlds with known ground truth replace Gemma-2-2B layer-13 activations on 100M tokens.
- 512 latents and k = 6 (paper: 73,728 and k = 100).
- 32k steps at lr 1e-3 (paper: about 2 epochs at lr 1e-4 over 100M tokens). The learning rate is raised because the run is far shorter.
- The causal (KL patching), steering and auto-interpretability experiments are not reproduced.
- Training preserves the paper's computational procedure. Scored discovery uses centered feature/input covariance to estimate model presence, with presence below 0.2 and support on at least 0.5% of inputs; negative base estimates are clipped to zero. This evaluator-specific measurement is independent of submitted decoder norms and rotations. It differs from the paper's uncentered Latent Scaling along a learned decoder. Only the largest-mass correct factor per target gets credit; duplicate contribution counts as false discovery. Correctness uses decoder alignment and ground-truth specificity. Decoder-norm classification and residual/reconstruction Latent Scaling remain diagnostics; these synthetic rates are not paper-scale Table 1 estimates.
