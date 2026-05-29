# Experiment Overview

## synthetic_chat_specificity

Tests whether the reproduced system can separate genuinely fine-tuned-model-specific activation factors from factors that are present in both paired activation streams. The experiment also checks that this separation is not obtained by sacrificing reconstruction quality or by using substantially denser latent activations.

## synthetic_complete_shrinkage

Tests whether the reproduced system avoids marking lower-amplitude shared activation factors as fine-tuned-model-specific. The experiment keeps the same scoring constraints on reconstruction quality and latent sparsity.
