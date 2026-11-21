# Hardening outcome

**Better results now score higher in the measured paired comparisons.** Support-constrained amplitude fitting improves the reference on all five common hidden-input batches and all three matched training seeds. Its mean paired rank gain is **+0.007865**, paired SD **0.001472**, with a Student-t (4 degrees of freedom) 95% lower confidence bound of **+0.006038**. A simple closed-form decoder refit loses on every paired comparison. Neither the reference nor the improvement reaches the bounds of the scored metrics.

Final job `caf0ab6e-583` exited 0. It completed the consolidated audit, checked both independent evaluators and direct versus streamed statistics, exercised the no-argument entry points, regenerated fresh reference-only scores, and passed `python3 validate.py`. The final evidence is in [hardening/final_results/](hardening/final_results/), particularly [headroom.json](hardening/final_results/headroom.json), [rank_summary.json](hardening/final_results/rank_summary.json), and [entrypoint_checks.json](hardening/final_results/entrypoint_checks.json).

This is a reproduction of the paper's mechanism on scaled synthetic activations, **not** a reproduction of its Gemma results. The milestone is `core_claim`. The stronger variant makes a measured tradeoff: better reconstruction, unchanged coverage, and slightly worse specificity on the original batch. It earns a higher total score under the fixed scientific weights; it does not dominate every individual metric.

## What changed after REVIEW.md

Both review findings were substantive. The incoming setup retained all 16 partially separated concepts as model-only twins, and the suggested cheap norm correction had never been measured.

1. **Restore the mechanism before claiming success.** Increase the shared occurrence probability of partially separated concepts from 0.04 to 0.12 and reduce each exclusive occurrence from 0.03 to 0.012. Increase the dictionary from 256 to 512 factors, reduce k from 8 to 6, anneal from 24, increase reference training to 32,000 steps, and use 65,536 training pairs. Coupling now produces the activity savings discussed in the paper. These are scale, frequency and training-budget changes; the paper's computational procedure is retained.
2. **Measure the cheap correction.** Fit the paper's raw-activation Latent Scaling estimate on unlabeled training inputs, then minimally raise suspected false labels' base-decoder norms across the labeling threshold, or set them to the full estimate. The incoming norm-label score rewarded this correction far more than the honest improvement. Discovery therefore had to stop trusting a decoder norm as evidence of model specificity.
3. **Remove two unstable measurement dependencies.** A fixed-count discovery list changed coverage when nearly tied factors exchanged places, even after enlarging evaluation eightfold. Fixed presence and support thresholds remove that boundary. A subsequent decoder-refit control still gained by rotating the direction used to estimate presence, despite worse reconstruction. Final discovery selection and contribution mass use centered feature/input covariance, independent of decoder edits for fixed features. Decoder alignment still determines correctness, and submitted decoders still determine reconstruction.

The implemented training remains recognizable as the paper's algorithm: a shared nonnegative encoder, paired decoders and initialization, decoder-norm-weighted L1 and batch-wide weighted BatchTopK, the auxiliary dead-latent residual loss (Eq. 12, alpha 1/32), training-estimated inference threshold (Eqs. 13–14), and an L0-calibrated L1 comparison. Earlier hardening restored the missing auxiliary loss and corrected the leave-one-out Latent Scaling diagnostic. Final scale/settings and departures are recorded in [briefing/method.md](briefing/method.md). Learning rate, dimensions, dictionary, data source and training exposure remain scaled relative to the paper; causal patching, steering and interpretation studies are absent.

Across three final hidden training seeds, raw mechanism diagnostics are:

| Quantity | Main: BatchTopK | Main: L1 | Faint: BatchTopK | Faint: L1 |
|---|---:|---:|---:|---:|
| Partially separated concepts with a shared factor, out of 16 | 16 at every seed | 1–3 | 16 at every seed | 4–9 |
| Concepts with a spurious chat-only twin | 0 at every seed | 14–16 | 0 at every seed | 15–16 |
| Norm-labeled false contribution fraction, unscored | 0.103–0.114 | 0.745–0.758 | 0.310–0.314 | 0.986–0.989 |
| Raw joint FVE | 0.9773–0.9777 | 0.5852–0.5863 | 0.9750–0.9753 | 0.5866–0.5870 |

Thus the paper's direction appears independently of failure penalties, at every tested seed. L1's zero or low raw covariance-based false rate is not evidence of useful discovery: it underselects and fails reconstruction. The paper-style norm diagnostic above is retained separately; we do not claim that every specificity diagnostic favors BatchTopK.

## Reference before and after

“Before” here means the incoming previously hardened gym in [hardening/before/](hardening/before/), not the older REDTEAM replay. The measurement and scale changed, so these are versioned raw results, not a claim of directly comparable error reductions.

| Hidden experiment / method | Before false / recall / FVE | Final false / recall / FVE |
|---|---|---|
| Main BatchTopK | 0.571919 / 0.500000 / 0.981093 | **0.044059 / 0.937500 / 0.977314** |
| Faint BatchTopK | 0.912382 / 0.250000 / 0.978567 | **0.021031 / 0.843750 / 0.975304** |
| Main L1 | 0.701033 / 0.500000 / 0.817612 | **1 / 0 / 0 (failed)** |
| Faint L1 | 1 / 0 / 0.698650 | **1 / 0 / 0 (failed)** |

Final L1 failure values are penalties, not measured zero FVE. Its seed-0 raw joint FVE is 0.586266 / 0.586989, below the evaluator-fixed 0.60 requirement; its stream FVEs also fail. Raw diagnostics are preserved. The baseline is clearly below the reference without concealing why.

Final development BatchTopK false / recall / FVE is **0.018154 / 0.968750 / 0.975440** (main) and **0.016408 / 0.937500 / 0.973193** (faint). Train and test have independent worlds and samples, the same interface and difficulty, and similar reference behavior. L1 fails on both sides. Reference rank is zero by definition in each version.

The earlier REDTEAM replay was worse still: BatchTopK false rate 0.333333 versus L1 0.281250, with FVE 0.951530 versus 0.982151, reversing the claimed comparison. The final raw mechanism results above correct that failure on the scaled task.

All 48 fits (four groups, three training seeds, two experiments, two worlds) were produced from the final training code and data. Reference and baseline fits ran sequentially without another experiment beside them. Node loss interrupted the first audit; saved complete fits were resumed without changing settings. Later changes were evaluator-only and rescored the saved outputs. No reference score depends on elapsed time: training uses fixed steps, deterministic seeds and fixed thread settings, and no wall-clock cutoff occurs in the method path. Recorded training seconds are unscored diagnostics.

## Headroom and ordering

Rank uses experiment weight times the **absolute** metric coefficient, times improvement in `higher_is_better`'s direction, divided by the metric's population standard deviation across the attempts being ranked. The final audit includes its complete 40-attempt cohort, including failed attempts and shortcuts, excluding only duplicate frozen copies of reference seed 0 and L1 seed 0. Future cohorts change magnitudes. The incoming audit normalized using an honest-only cohort; historical rank magnitudes below are explicitly versioned and cannot be compared as a common unit.

The 0.6 main / 0.4 faint weights and all coefficient magnitudes are unchanged from the incoming hardening. Main specificity and coverage each have magnitude 1; reconstruction 0.5. Faint coverage has magnitude 1, specificity 0.5, reconstruction 0.5. Reconstruction and useful recovery are also hard constraints, so a failure cannot buy rank through a different metric. All three final metrics vary across methods; none is a time or paper-number agreement score.

| Variant | Before | Final hidden rank improvement over matching reference seeds 0 / 1 / 2 |
|---|---|---|
| Honest support amplitude fitting | Original REDTEAM: +0.152242 / +0.152026 / −0.269338; mean +0.011643, below unpaired seed spread 0.601475 | **+0.005758 / +0.005103 / +0.006355** |
| Longer training | Incoming 4x k-annealed variant: +0.555929 / +0.657 / +0.858 in its archived normalization; original REDTEAM longer training mean −1.326092 versus fresh references | Final 3x steps: −0.031360 / −0.073111 / +0.041477; **not reliable headroom** |
| Weaker training | REDTEAM 2-epoch mean −0.080965 against its stored reference; one seed gained +0.153173 despite negative FVE | Final 2,000 steps: **−4.804774 / −4.825252 / −4.852823**, complete suite failure |
| Iterative inference | Added during this audit; no incoming result | −0.062376 / +0.004712 / −0.028073; not reliable headroom |
| Iterative inference plus training-only decoder refit | Added during this audit; no incoming result | −0.060561 / +0.006170 / −0.027639; not reliable headroom |

The successful alternative performs 24 projected-gradient updates of nonnegative amplitudes on the reference encoder's original support, with the learned dictionary fixed. It uses only the input activations and learned parameters, through the same optional ONNX interface available to future methods. It changes inference, not the reproduced reference training algorithm.

At seed 0, main FVE increases **0.977314 → 0.985116**, faint **0.975304 → 0.983474**; recall remains **0.9375 / 0.84375** and mean L0 slightly decreases. False mass increases slightly on the original batch: **0.044059 → 0.045416**, **0.021031 → 0.023376**. The final score rewards the reconstruction improvement while charging for that specificity cost. On development worlds the three paired rank gains are +0.008427 / +0.008711 / +0.008789.

Each hidden batch contains **524,288 pairs per experiment**, common to every compared method, with fixed seed/backbone within each comparison. Five paired gains are **+0.005758, +0.007774, +0.008529, +0.007492, +0.009772**. Their mean exceeds their paired SD; the confidence lower bound is positive. All five also exceed the closed-form decoder control. Training-seed robustness is a separate three-pair check above. Unpaired reference rank spread is **0.048049**, larger than the improvement; we do not claim improvement exceeds arbitrary retraining variation. The positive claim is the requested paired comparison with the seed/backbone held fixed.

The stored reference sits at **none** of the scored bounds: false rates are positive, recalls below 1, and FVE below 1 in both experiments. The successful simple variant also reaches none of them. The faint, denser experiment retains weight 0.4 and targets the paper's small-variance-information limitation (Section 5 and Appendix B.3); it was added by the prior hardening and is retained and rerun here. No extra scored experiment was needed once these two regimes showed unsaturated, measured headroom. Enlarging evaluation from 65,536 to 524,288 and removing unstable fixed-count selection was necessary; numerical distance from a bound alone did not suffice.

A 512-atom exemplar enumeration control and joint PCA both fail the task and score **−4.804774** on test (**−4.821385** on train). The dimension/concept count, fixed capacity, fresh inputs and reconstruction/coverage constraints prevent these measured brute-force/simple controls from occupying the headroom. This is empirical evidence against these controls, not a proof against every possible search procedure.

## Shortcuts: measured closure

Before values in this table come from `REDTEAM.md` unless labeled “incoming” or “intermediate.” Final values use the complete final cohort and compare against reference seed 0. Failure means **(false 1, recall 0, FVE 0) in both experiments**, not an empty result or a skipped value.

| Shortcut / control | Before | Fix and final hidden result |
|---|---|---|
| All-zero output | +2.063351 | Empty discovery is false rate 1; reconstruction and recovery must pass. Zero encoder: **−4.804774**, failed. |
| Empty features with unrelated reconstruction | +2.618449 | Submitted feature/reconstruction arrays are ignored. Both are recomputed from the procedure and decoders. Re-expressed zero encoder: **−4.804774**, failed. |
| Copy input as claimed reconstruction | +1.663532 | Reconstruction must equal features times the submitted decoders plus bias. Dense identity control violates sparsity/recovery: **−4.804774**, failed. |
| Move decoder-norm threshold | +1.373180 | Measured presence, operating point and contribution mass are evaluator-defined. Norm-mask edit: **−0.004358**. Forcing base decoders to zero: **−0.013366**. |
| Negative features | +1.493514 | Validate finite nonnegative outputs from both interfaces. Negative tensor program: **−4.804774**, failed. |
| NaN/malformed outputs | NaN serialized; rank undefined | Validate artifacts, outputs and all metrics; exceptions become complete finite failure. NaN parameters: **−4.804774**. |
| Omit method/experiment through registry | Missing output tied at 0; missing artifact could abort scoring | Discover directories from disk, union with frozen comparisons; emit every metric in every experiment. Missing experiment: **−4.804774**. A newly added unlisted method was found, scored, and its missing experiment propagated to both sides' complete worst values. |
| Exceed dictionary capacity | Earlier capacity was not consistently enforced | D > 512 fails before evaluation: **−4.804774**. Dense and oversize variants cannot trade capacity for score. |
| Exemplar brute force / PCA | +1.120171 / +0.341801 | Scale to 112 concepts in 64-d streams, 512-factor budget, L0 <= 6.6, fresh hidden inputs and useful coverage/reconstruction constraints. Both controls: **−4.804774**. |
| Targeted Latent Scaling norm correction (REVIEW) | Incoming minimal **+6.150079**, full **+6.042864**, versus honest longer seed 0 **+0.555929**, using archived incoming cohort scales | Labels no longer follow editable norms. Final minimal seed-0 **−0.000002928**, full **−0.000003145**; negative at all three matched hidden training seeds. |
| Closed-form training-only decoder refit | Intermediate gains on five paired batches: **+0.001905, +0.002104, +0.001597, +0.001948, +0.001648**, despite worse FVE | Derive presence direction and mass from centered feature/input covariance, holding features fixed. Final gains: **−0.0000850, −0.0000823, −0.0000818, −0.0000772, −0.0000826**. Also negative at all three matched training seeds and below the honest improvement on all five batches. |

The newly measured incoming norm correction reduces main false mass **0.571919 → 0**, faint **0.912382 → 0.203442**, with unchanged recall and only small FVE losses for the minimal edit. That was a real paying shortcut, not headroom. Raw measurements are in [latent_scaling_probe.json](hardening/before/latent_scaling_probe.json); [latent_scaling_probe_rank.json](hardening/before/latent_scaling_probe_rank.json) states the historical normalization explicitly.

The minimal/full norm correction is an **exact no-op on all three development seeds** in the final setup: no parameter satisfies its correction condition. Its train tie therefore duplicates the same method, rather than concealing a changed-output shortcut. On hidden artifacts where it edits a decoder, discovery evidence is unchanged and reconstruction gets slightly worse. We do not present a harmless duplicate's tie as successful blocking of a paying attack.

Inherited protections also remain: disclosed labels/generators no longer regenerate hidden worlds; taxonomy and target arrays are evaluator-owned; score-file forgery is overwritten and is never an input to scoring. REDTEAM values were +0.898999 for visible-label supervision, +1.559987 for taxonomy replacement, +1.663532 for target replacement, and 0 for score-file forgery that left the underlying method unchanged. Those exact old write attacks were already closed by the prior hardening and were not replayed here. A forged file with a valid unchanged method still receives that method's computed score, a legitimate tie; it cannot supply ranked numbers. The changed interfaces, raw-output controls, missing-method check and fresh uncached entry points were rerun in the final audit. No before/after number is invented for an unrerun attack.

## Evaluation boundary and packaging

- Hidden worlds, seeds, truth, previous replay assets and their regeneration live only in `eval/test/`. Public development generation is separate; held-out training data shipped to methods is unlabeled. Scoring samples unseen inputs. Public development knowledge does not regenerate hidden concepts or inputs.
- No method Python executes inside the test. Methods submit finite arrays or an ONNX tensor program using standard CPU operators with embedded constants (maximum 64 MiB); recursive external-data references are rejected. It receives only paired input tensors, not file access or hidden metadata. A better procedure can therefore be expressed without limiting submissions to reference settings. Hidden scoring performs inference/statistics, never method training.
- The evaluator recomputes features, reconstructions, selection, correctness and all metrics. Its own ground truth, comparison artifacts, thresholds, sample count and budgets are fixed. Centered covariance avoids background-mean co-occurrence; minimum firing frequency is 0.5% and presence threshold 0.2. Each target receives credit once; duplicate contribution is false mass. Norm and residual Latent Scaling remain unscored mechanism diagnostics. The scored covariance measurement deliberately differs from the paper's uncentered decoder-direction scaling and is documented as such.
- Capacity <=512, mean L0 <=6.6, joint and each stream FVE >=0.60, recall >=0.10, and validity must hold in both experiments. Any failure fills all six scored entries with their worst values. Successful methods are better in every scored quantity than failures, so coefficient signs or other metrics cannot rescue task failure.
- `eval/train/` and `eval/test/` implement the measurement independently and share no code. Neither imports `method/`; data loading also does not depend on the method. Train and test invoke the same artifact interface and discover all methods without a method-name whitelist. Reserved comparison names resolve to evaluator-frozen artifacts. Both no-argument entry points were checked with a fresh unlisted method and a missing experiment.
- Canonical `scoring/scores_test.json`, `scores_train.json`, and `scores.json` contain **only** the final reference and L1 comparison runs. All ranked values are under `experiments.*.results`. Raw diagnostics, ranks and shortcuts in `hardening/` are audit evidence and are not ranked. `reference.json` contains measured gym results on the scaled setup; original paper numbers remain unchanged under underscore-prefixed metadata. No setup was selected to match those numbers or satisfy a paper-number comparison in validation.
- `scripts/run.sh` again calls the full `scripts/reproduce.sh`: prepare data, train reference, train baseline, evaluate both sides, validate. It does not depend on ignored audit checkpoints. Small authored training data and frozen comparisons are tracked. There are no required pre-existing external datasets or weights in this synthetic setup, so `download.sh` has no fictitious external fetch: it reproducibly prepares the synthetic data and checks the committed unlabeled held-out training files. No committed code refers to `shared/`; the container uses `environment/` with Python dependencies in its setup script.
- Final validation passed, including import separation, independent train/test code, entry-point wiring, complete schema and differentiated method scores. Four heuristic RNG warnings remain because the evaluator generates fresh inputs and uses submitted encoders rather than loading prediction arrays; direct/streamed consistency and fresh uncached scoring establish that it reads method artifacts and does not rerun training. Audit-only resume checkpoints were introduced after scheduler preemptions, fingerprint inputs/code/artifacts, and are never read by the published evaluators.

## Remaining limits and intentionally open avenues

No measured changed-output major shortcut remains profitable. We did not add defenses against leftover processes or a modified Python environment: the hidden test runs in a fresh job. A separate attempt grader still needs to reject edits to protected files, hidden-answer access and explicit test-seed reverse engineering. Public development supervision cannot stand in for learning the hidden concepts.

The synthetic process remains a sparse-mixture problem. Learning a better dictionary or an input-only sparse inference procedure from unlabeled training samples is permitted research; writing down the hidden world is not. The PCA, exemplar, norm-edit and closed-form controls above do not pay, but the audit is not an exhaustive search over every analytic estimator or every possible finite tensor program. The ONNX interface has no elapsed-time score or method wall-clock budget, and does not claim universal operation-count enforcement. It hides the truth/generator from the submitted tensor procedure and permits honest inference improvements. A more effective honest procedure is intended headroom.

Only three matched training seeds and five paired input batches were measured. They establish the reported paired gain, not superiority after arbitrary retraining or on every possible world. Longer training and two iterative alternatives were explicitly **not** accepted as headroom. Earlier small-sample and fixed-count headroom checks failed; complete intermediate evidence and the sequence of evaluator-only corrections are retained in `hardening/intermediate_65536/`, `hardening/intermediate_top32/`, `hardening/intermediate_decoder_selection/`, and [MEASUREMENT_HISTORY.md](hardening/MEASUREMENT_HISTORY.md). The previous hardening report is [PRIOR_HARDENING.md](hardening/PRIOR_HARDENING.md).

The scientific limitation left plainly unresolved is transfer to real model activations and the paper's causal and interpretation claims. None of these synthetic scores establishes that transfer. The final gym rewards useful sparse model-difference discovery on its stated scaled problem, with measured headroom and explicit constraints.
