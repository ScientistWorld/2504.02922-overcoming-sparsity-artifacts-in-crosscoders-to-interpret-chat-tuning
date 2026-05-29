# Problem

The paper studies model diffing: given paired activations from a base model and a chat-tuned model on the same inputs, identify which latent factors are genuinely specific to the chat-tuned model and which are shared with the base model.

A successful system should learn sparse latent factors that reconstruct both activation streams while allowing each latent to have separate base and chat decoder directions. The main risk is false positives: a latent can appear chat-only by decoder norm even though the corresponding concept is present in the base activations.

This reproduction uses a scaled paired-activation benchmark generated in `data/`. It preserves the paper's problem structure with shared, base-only, chat-only, and weak-shared activation factors, while avoiding the full Gemma activation extraction cost.
