# Problem

The task is model diffing with sparse dictionaries. You are given paired activation vectors from a base model and a fine-tuned (chat) model on the same inputs. A method learns a dictionary of factors that reconstructs both activation streams. Each factor has one activation per input, shared by the two streams, and its own direction in each stream. The aim is to find the factors that are genuinely specific to the fine-tuned model and to keep them apart from factors both models share.

The main failure mode is false specificity. A factor that both models use can look fine-tuned-only, for example because it is weaker in the base stream, or because the two models use the same direction on partly different inputs. A useful method surfaces the genuinely fine-tuned-specific structure without labelling shared structure as fine-tuned-only, and still reconstructs both streams sparsely.

The benchmark replaces language-model activations with synthetic paired worlds that have known ground truth. Each world is a set of concept directions in a 64-dimensional space. Every concept is shared, weaker in the base stream, decoupled (used by both models, but partly on different inputs), fine-tuned-only, or base-only. Activations are sparse nonnegative sums of these concepts plus noise.

Each experiment has two worlds drawn from the same generator family:

- a **dev** world (`data/worlds/<experiment>/dev/`), whose generator and ground truth are public, for development;
- a **held-out** world, of which you see only unlabeled training activations (`data/worlds/<experiment>/heldout/train.npz`). Its concepts, family sizes and labels are hidden, and the hidden test scores your dictionary on fresh held-out inputs.

Experiments:

- `synthetic_chat_specificity`: genuine fine-tuned-only concepts next to strength-asymmetric and decoupled shared concepts.
- `synthetic_faint_chat_concepts`: a denser world in which the fine-tuned-only concepts are faint, so the genuinely fine-tuned-specific structure carries little variance.
