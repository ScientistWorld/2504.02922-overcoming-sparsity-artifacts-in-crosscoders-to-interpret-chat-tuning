# Overview

This paper shows that standard L1 crosscoders can misidentify shared concepts as chat-model-only latents when comparing base and chat-tuned language models. It introduces Latent Scaling to diagnose those artifacts and shows that BatchTopK sparsity substantially reduces them.

The reproduction implements the actual crosscoder architecture, L1 and BatchTopK training objectives, decoder norm ranking, and Latent Scaling diagnostics on scaled paired-activation benchmarks. A corrected two-case suite runs end-to-end, but external validation rejected the latest `synthetic_chat_specificity` BatchTopK false fine-tuned-only rate as too far from the paper anchor. The next run regenerates the suite with a fixed `k=12` BatchTopK setting.
