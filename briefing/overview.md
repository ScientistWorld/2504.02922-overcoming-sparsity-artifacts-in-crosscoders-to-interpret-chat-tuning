# Overview

The paper identifies Complete Shrinkage and Latent Decoupling as reasons that L1 crosscoders can mislabel shared structure as fine-tuned-only. It introduces Latent Scaling to detect these artifacts and finds that BatchTopK crosscoders recover more genuinely model-specific structure while preserving sparse reconstruction.

This gym implements the paper's training procedure on scaled synthetic paired activations, with independent development and hidden worlds. In the final three-seed hidden audit, BatchTopK learns shared factors for all 16 partially separated concepts, with no spurious chat-only twins. L1 retains 14–16 such twins in the main world and 15–16 in the faint world. Raw norm-based false-discovery diagnostics and reconstruction also favor BatchTopK. This establishes the mechanism on these synthetic worlds; real Gemma activations, causal patching, steering and interpretation studies are not reproduced.

The frozen BatchTopK reference has hidden false discovery / recall / FVE of 0.0441 / 0.9375 / 0.9773 in `synthetic_chat_specificity`, and 0.0210 / 0.8438 / 0.9753 in `synthetic_faint_chat_concepts`. L1's raw joint FVE is about 0.586, below the fixed 0.60 reconstruction requirement; its ranked results therefore receive complete failure values. Raw diagnostics remain available in the hardening evidence.

Discovery is judged from feature/input evidence at a fixed specificity threshold, alongside ground-truth correctness, coverage, reconstruction and sparsity. This hardened measurement differs from the paper's decoder-norm labels, which remain diagnostic. See `evaluation.md` for the method-neutral contract.

An alternative inference procedure improves reconstruction within the same learned support. It beats the reference at all three matched training seeds and all five common input batches; the mean paired rank gain is 0.00787 with a 95% lower bound of 0.00604. Neither it nor the reference saturates the scored metrics. Simple decoder refitting, PCA and exemplar controls do not beat the reference. Full measurements and remaining limitations are in `HARDENING.md`.
