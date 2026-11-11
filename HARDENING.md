# Hardening report (iteration 11)

**Bottom line.** The red team found that the old test did not reward better research. Its evaluator trusted whatever the method wrote. It scored only one of the two experiments, read its method list from an editable file, exposed the ground truth, and rewarded a method for finding nothing. A zero-output method outranked the reference by +2.06. The reproduced method also lost to L1 in the paper's own comparison.

These were structural problems, so I rebuilt the evaluation contract instead of patching it, and reran everything through the real entry points in one GPU job (`163fab30-025`, exit 0). After the fix:

- every red-team shortcut scores far below the reference;
- the paper's comparison holds in the paper's direction, well outside seed noise;
- a genuinely better variant beats the reference by more than the reference's seed spread, on every seed;
- a clearly weaker variant scores clearly lower.

Evidence: `hardening/results/` (scores, diagnostics, `rank_summary.json`, `script_checks.json`), produced by `hardening/audit.py`; job log under `jobs/163fab30-025/`.

## What changed

| Problem (REDTEAM.md) | Fix |
|---|---|
| Evaluator trusted submitted features, reconstructions and decoder statistics | A submission is now only parameters: encoder, thresholds, two decoders, biases. The evaluator samples its own inputs, computes activations (ReLU plus thresholds, so they are nonnegative by construction), reconstructions and every metric. |
| Test inputs, concept answers and the generator were public; the taxonomy and targets were editable | Hidden **held-out worlds**: secret 126-bit seeds and the ground truth live only in `eval/test/` (`build_heldout.py`, `worlds/*.npz`). Methods get only that world's unlabeled training activations (`data/worlds/*/heldout/train.npz`). Test inputs are sampled inside `eval/test/evaluate.py` at test time. Family sizes differ from the public dev worlds. |
| No-argument hidden command scored one experiment only | `scripts/evaluate_test.sh` takes no arguments and scores **both** experiments for **every** directory under `checkpoints/methods/`. |
| Method list read from editable `scores_train.json`; unlisted methods skipped; missing artifacts crashed the job | Methods are found on disk. A missing, unreadable, non-finite or out-of-budget artifact gets the **worst value of every metric** of that experiment (false rate 1, recall 0, FVE 0) and the run continues. |
| Vacuous wins: no chat-only candidates gave false rate 0; L0 0 and zero features gave perfect specificity | False fine-tuned-only rate is 1.0 when nothing is labelled fine-tuned-only. Recall counts genuinely specific factors in a fixed-size top-N. The L0 reward was replaced by a hard budget (mean L0 ≤ 8.8, dictionary ≤ 256), so a method cannot win by being sparse and empty. |
| Coarse, noisy metrics (13 candidates, 4 concepts, 512 inputs) | 32,768 scored inputs and 32 fine-tuned-specific targets per world. Each latent's specificity is judged against ground truth: concept alignment ≥ 0.8 and the concept's base/chat presence ratio under the latent's own activations < 0.2. The false rate is mass-weighted, so duplicating good latents cannot dilute bad ones. |
| Reference/baseline on the other side of the comparison was editable | Frozen copies of the reference (`batchtopk`) and baseline (`l1`) are stored in `eval/test/reference_methods/` (held-out) and `eval/train/reference_methods/` (dev). The hidden test always scores the frozen copy under those names. |
| Paper procedure departures: BatchTopK aux loss missing, sparsity not matched, mean-over-dims loss, gradient clipping, 1,280 steps | Restored the paper's auxiliary dead-latent loss (Eq. 12, α = 1/32), the sum-of-squares reconstruction loss and no clipping. The L1 μ is chosen by bisection to the same L0 as BatchTopK (paper Sec. 3; L0 8.0–8.2). Training is steps-based (16k steps). |
| Reference at bound / headroom unclear | New experiment `synthetic_faint_chat_concepts`, from the paper's own limitations (App. B.3 bias of BatchTopK toward shared latents; Sec. 5: much fine-tuned information stays in the error). |

The old experiment `synthetic_complete_shrinkage` is folded into `synthetic_chat_specificity`. That world now contains strength-asymmetric shared concepts (Complete Shrinkage) and decoupled concepts (Latent Decoupling) together, as real model pairs do. The diagnostics break spurious mass down by cause.

## Reference before and after

| | False FT-only rate | Recall | FVE |
|---|---:|---:|---:|
| Before: saved BatchTopK replay (red team) | 0.333 | n/a | 0.952 |
| Before: L1 (red team) | **0.281** (better than BatchTopK) | n/a | **0.982** (better) |
| After: BatchTopK, `synthetic_chat_specificity` | **0.572** | 0.50 | **0.981** |
| After: L1, `synthetic_chat_specificity` | 0.701 | 0.50 | 0.818 |
| After: BatchTopK, `synthetic_faint_chat_concepts` | **0.912** | **0.25** | **0.979** |
| After: L1, `synthetic_faint_chat_concepts` | 1.000 | 0.00 | 0.699 |

The metrics were redefined, so the before and after numbers are not on the same scale. What matters is the direction of the comparison.

Before, the replay reversed the paper. After, BatchTopK beats L1 in the paper's direction:

- lower false fine-tuned-only rate (paper: 0.7% vs 67.1%);
- higher FVE at matched L0 (paper: 87.6% vs 84.4%);
- far fewer fine-tuned-only latents: 32 vs 60 (paper: 134 vs 3,176).

The paper's own Latent Scaling estimate, computed by the evaluator as a diagnostic, flags 9% of BatchTopK's fine-tuned-only latents vs 23% of L1's.

Seed noise is far smaller than these gaps (test, 3 seeds each):

- BatchTopK false rate 0.5718–0.5720 vs L1 0.689–0.701, a gap of 0.13 against a spread of 0.012;
- FVE 0.9811–0.9812 vs 0.817–0.818.

Train and test agree:

- BatchTopK, train: 0.571 / 0.50 / 0.979 and 0.887 / 0.34 / 0.978;
- L1, train: 0.665 / 0.50 / 0.824 and 1.0 / 0.0 / 0.709.

## Do better results now score higher?

Rank is a proxy: Σ weight × |coefficient| × directed improvement / cohort std, using the honest attempts of this audit (`hardening/rank.py`) and the final coefficients. Seed-paired deltas compare each variant with the reference trained at the same seed. Reference seed spread is **0.26** on test (0.15 on train). Before, it was 0.60.

| Attempt (3 seeds) | Before (red team) | After, test: paired vs reference | After, train |
|---|---|---|---|
| Better: 4× steps + paper's k-annealing recipe (`better_long_kanneal`) | (no analogue) | **+0.56, +0.66, +0.86** (mean +0.69) | +0.31, +0.40, +0.18 |
| Better: 4× steps (`better_long`) | 320 epochs: **−0.20** (scored *lower* despite FVE +0.11) | **+0.28, +0.40, +0.27** | +0.33, +0.17, +0.16 |
| Honest refinement (red team: inference amplitude fit) | +0.012 mean, below the 0.60 spread | Not expressible in the new contract (inference = submitted encoder). Nearest analogue, least-squares decoder refit: +0.00, +0.09, +0.00 | 0, 0, 0 |
| Weaker: 1/16 of the steps (`weaker_short`) | 2 epochs: −0.08 (seed 2: **+0.15**) | **−1.91, −4.80, −5.23** | −5.19, −1.90, −5.20 |
| L1 baseline (paper) | ranked above the reference | **−2.40, −2.10, −2.19** | −2.18, −2.23, −2.29 |
| App. B.3 concatenated-norm selection | n/a | +0.18, +0.91, +0.00 (not consistently better) | +0.17, +0.17, +0.16 |
| Max-norm selection | n/a | +0.02, +0.45, +0.13 | +0.20, +0.46, −0.08 |
| k-annealing at the reference budget | n/a | −0.30, +0.23, −0.23 (not better) | +0.34, −0.08, −0.13 |
| Ablation: no auxiliary loss | n/a | −0.11, −0.10, −0.00 | 0, 0, 0 |

Why the better variant wins, in raw metrics (test):

- It recovers 75–81% of the faint fine-tuned-only concepts vs 38–44% for the reference (`synthetic_faint_chat_concepts` recall 0.375–0.406 vs 0.219–0.25).
- It occasionally resolves a decoupled concept (`synthetic_chat_specificity` recall 0.50–0.56).
- FVE is equal or higher, and its false rate is no worse.

Its smallest paired gain (+0.56) is twice the reference seed spread. The same ordering holds on train.

## Shortcuts: fix and score before → after

Scores are rank deltas relative to the reference. "Before" is the red team's delta against the frozen gym. "After" is this audit's hidden-test delta. Constructions are in `hardening/audit.py`.

| Shortcut | Before | After | How it is closed |
|---|---:|---:|---|
| Visible labels / ground-truth decoders | +0.90 | **−8.27** (`shortcut_dev_oracle`; +1.98 on train only) | Held-out ground truth is hidden. A dictionary built from the public dev world's ground truth fails on the held-out world (L0 38 > 8.8). |
| Zero outputs | +2.06 | **−8.27** (`shortcut_zero_encoder`) | Empty fine-tuned-only set gives false rate 1; recall 0; FVE 0. |
| Empty features with kept reconstructions | +2.62 | not expressible | The evaluator computes features and reconstructions itself. |
| Copied reconstruction | +1.66 | **−8.27** (`shortcut_dense_identity`) | Reconstruction is recomputed. An identity-like dictionary needs L0 128 > 8.8 and fails. |
| Threshold mask (move base/chat norm ratio) | +1.37 | **−1.93** (`shortcut_threshold_mask`) | Pushing every latent below the 0.9 label empties the fine-tuned-only set (false rate 1) and costs FVE. |
| Negative features | +1.49 | not expressible | The evaluator applies ReLU and thresholds. |
| Taxonomy replacement | +1.56 | **0 change on test** | Hidden taxonomy. Editing the public dev taxonomy changes only `scores_train.json`; the test rows were byte-identical. |
| Target replacement | +1.66 | not expressible | Test inputs are sampled inside `eval/test/`. |
| Registry omission (no output) | 0 (tie) | **−8.27** (no artifacts → worst everywhere); a missing experiment gives **−2.59** (`shortcut_missing_experiment`) | Methods are enumerated from disk; missing artifacts get worst values. |
| Unlisted method | omitted | scored | Verified: a method added after train scoring and with `scores_train.json` deleted was scored on both experiments. |
| Score-file forgery | tie | overwritten | Verified: a forged `scores_test.json` was replaced by the exact clean scores. |
| NaN outputs | undefined | **−8.27** (`shortcut_nan_params`) | Non-finite parameters fail the run with worst values. |
| Missing artifact | job crashed | worst values, no crash | |
| Brute-force exemplar dictionary | +1.12 | **−1.07** (`bruteforce_exemplars`) | 256 stored training pairs (the size cap) reconstruct poorly (FVE 0.64) and find no faint fine-tuned-only concepts. |
| PCA control | +0.34 | **−7.22** (`baseline_pca`) | |
| New: oversized dictionary (512 duplicated latents) | n/a | **−8.27** | Dictionary cap. |
| New: over-labelling (zero base decoder where Δ ≥ 0.6) | n/a | **−0.67** | Spurious labels raise the false rate and cost FVE. |
| New: reuse the dev-world dictionary for held-out | n/a | **−8.27** (0 on train) | The held-out world differs. |
| New: sabotage the workspace reference (`checkpoints/methods/batchtopk`) | n/a | reference rows unchanged | The hidden test scores the frozen copy. |

## Headroom check

**Does the reference sit at a bound?** No metric is at its bound in either experiment:

- `synthetic_chat_specificity`: recall 0.50, false rate 0.57, FVE 0.981. It does recover all 16 genuine fine-tuned-only concepts; the missing recall is the 16 decoupled concepts. Noise caps FVE near 0.995.
- `synthetic_faint_chat_concepts`: recall 0.25, false rate 0.91, FVE 0.979.

**What I added:** `synthetic_faint_chat_concepts`, built from the limitations the paper states itself, with weight 0.4.

**Does any variant actually score above the reference?** Yes. `better_long_kanneal` beats it on every seed by more than the reference seed spread (above).

**Is it cheap headroom?** No simple variant reaches the bound. The best faint-concept recall is 0.41 of 1.0.

**Does brute force solve it?** No. The exhaustive-exemplar dictionary at the maximum size scores −1.07.

**Do the baselines make sense?** Yes. L1 and PCA sit clearly below the reference, at −2.2 and −7.2.

## Weights and coefficients

- Experiment weights: `synthetic_chat_specificity` 0.6, since it carries the paper's core claim; `synthetic_faint_chat_concepts` 0.4. Before: 0.5 / 0.5 over two experiments that were both variants of the core claim.
- Coefficients: recall 1.0, false fine-tuned-only rate 1.0, FVE 0.5.
- In `synthetic_faint_chat_concepts` the false-rate coefficient is 0.5. The diagnostics show 89–94% of that experiment's fine-tuned-only mass is spurious decoupled-concept twins, which `synthetic_chat_specificity` already scores. At full weight this re-measurement dominated seed noise: reference seed spread 0.43 vs 0.26 at half weight. Conclusions are unchanged either way: the best variant's paired gains are +0.49/+0.75/+0.86 at full weight.
- The paper's reported numbers are unchanged in `scoring/reference.json`: false rate 0.007 vs 0.671 and FVE 0.876 vs 0.844, with Table 1/Table 4 counts under `_paper_reported`. `reconstruction_fve` stays the validator-facing primary metric because it is the only paper metric on a comparable scale.

## Shortcuts and limits left open (stated plainly)

1. **Label correction by editing decoders.** The fine-tuned-only label comes from the submitted decoder norms, as the paper defines it. A method can raise the base decoder of latents it believes are spurious, for example using the paper's Latent Scaling, and move them below Δ = 0.9. That lowers the false rate at a reconstruction cost. I treat this as legitimate, because it is the paper's recommended use of Latent Scaling and it makes the dictionary's own claim correct. It is cheap, though, and **was not run**. The undiscriminating version (`shortcut_threshold_mask`) loses −1.93.
2. **The paper's mechanism claim does not reproduce here.** In this world BatchTopK still represents all 16 decoupled concepts as fine-tuned-only/base-only twins (Latent Decoupling), which is its entire false fine-tuned-only mass. The paper's Latent Scaling diagnostic (ν > 0.6) would accept most of them. In `synthetic_chat_specificity` recall ties with L1 (0.50 each). The paper's "more genuinely specific latents" claim shows only in the faint-concept world (0.25 vs 0). The relative claims (lower false rate, better FVE) hold.
3. **Public dev ground truth.** Train-side scores can be inflated with it (dev oracle +1.98 on train). This does not transfer to the hidden test. The generator family code is public by design, but the held-out concepts and family sizes are not.
4. **Contract restriction.** Inference must be the submitted encoder plus thresholds and optional `topk`. Per-input optimisation, such as the red team's amplitude fit, cannot be scored. The optional `topk` path was not exercised by the audit.
5. **Scale.** The worlds are synthetic, 64-dimensional with 112 concepts; dictionaries have at most 256 latents, k = 8, and training runs 16k steps at lr 1e-3. The paper uses Gemma-2-2B, 73,728 latents, k = 100, lr 1e-4 and 100M tokens. The causal KL-patching, steering and auto-interpretability experiments are not reproduced.
6. **Rank normalization is approximate.** These rank numbers use this audit's cohort std. The real ranker normalizes over future attempts, so magnitudes will differ; orderings relative to seed spread are the durable evidence.
7. `python validate.py` passes. Its three soft warnings flag `np.random.default_rng` in `eval/`; that is deliberate, because the evaluator samples its own test inputs from the hidden world.

## Reproducing

- `bash scripts/run.sh` (= `scripts/reproduce.sh`): data, reference, baseline, train/test scores and validate.
- `python3 hardening/audit.py`: this report's audit, about 50 minutes on 1 GPU.
- `python3 hardening/rank.py`: recompute the ranks from the saved scores.

The frozen reference/baseline artifacts in `eval/{train,test}/reference_methods/` come from this iteration's job, trained by the final code.
