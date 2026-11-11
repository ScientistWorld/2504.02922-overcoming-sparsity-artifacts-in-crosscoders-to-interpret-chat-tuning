# Overview

The paper shows that standard L1 crosscoders misidentify concepts shared by a base and a chat model as chat-only latents, through Complete Shrinkage and Latent Decoupling. It introduces Latent Scaling to detect these artifacts and finds that BatchTopK crosscoders have far fewer of them, at a better reconstruction.

This gym reproduces that comparison on synthetic paired-activation worlds with known ground truth, scored on hidden held-out worlds:

- In `synthetic_chat_specificity`, the BatchTopK crosscoder (proposed) has a lower false fine-tuned-only rate than the L0-matched L1 crosscoder (0.57 vs 0.70). It also reconstructs far better (FVE 0.981 vs 0.818). Both recover all genuine fine-tuned-only concepts.
- The paper's mechanism claim is not reproduced in this world. BatchTopK still represents every decoupled concept as a fine-tuned-only/base-only twin pair (Latent Decoupling), and these twins make up all of its false fine-tuned-only mass. The paper's own Latent Scaling estimate would accept most of them (ν < 0.6).
- In `synthetic_faint_chat_concepts`, BatchTopK recovers only about 40% of the faint fine-tuned-only concepts. This is the regime the paper flags as a limitation, and the L1 crosscoder recovers none.

The gap between the reference and a genuinely better variant (a 4x training budget with the paper's k-annealing recipe) is larger than seed noise. Details: `HARDENING.md`.
