# Overview

This paper shows that standard L1 crosscoders can misidentify shared concepts as chat-model-only latents when comparing base and chat-tuned language models. It introduces Latent Scaling to diagnose those artifacts and shows that BatchTopK sparsity substantially reduces them.

The reproduction implements the actual crosscoder architecture, L1 and BatchTopK training objectives, decoder norm ranking, and Latent Scaling diagnostics on scaled paired-activation benchmarks. The corrected two-case suite supports the core claim: BatchTopK reduces false fine-tuned-only artifacts versus L1 in both synthetic regimes while preserving high reconstruction, though higher-fidelity reproduction on real Gemma 2 activations remains future work.
