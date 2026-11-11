> **Superseded (iteration 11):** this report verified the earlier `val_outputs.npz` evaluation contract. The red-team audit (`REDTEAM.md`) found that contract unsound, and it was replaced. See `HARDENING.md` for the current verification evidence.

# Verification Report

## Initial Assessment

The workspace started at `core_claim` with a completed scaled synthetic paired-activation reproduction. The claimed milestone is justified: the implemented method follows the paper's paired sparse factor approach, the evaluator reads saved artifacts, and the results support the main artifact-control claim on two synthetic settings. The reproduction is not a higher milestone because it does not use the original full Gemma 2 activation dataset and some auxiliary alignment metrics remain mixed.

## Issues Found

- **Minor**: `scoring/scores.json` was stale relative to the current evaluator contract and omitted `base_reconstruction_fve` and `chat_reconstruction_fve`, even though those constraint metrics were already present in train/test scores and `reference.json`.
- **Minor**: Some scoring documentation used target headings that could be read as experiments not present in `scores.json`, and a few descriptions were more method-specific than necessary for from-scratch mode.
- **Cosmetic**: A root Python cache directory was generated during local syntax checks.

## Fixes Applied

- Recomputed the missing base/chat reconstruction FVE values from saved suite artifacts and updated `scoring/scores.json`.
- Tightened `scoring/TARGETS.md`, `scoring/EXPERIMENTS.md`, `scoring/DIRECTION.md`, and `scoring/reference.json` wording so method-agnostic documents describe the task, output contract, and metrics without exposing implementation details.
- Added notes to `reference.json` explaining that raw paper candidate counts are full-scale anchors while the reproduced benchmark uses fewer candidates.
- Removed generated root cache artifacts.

Validation run during final verification:

- `python3 validate.py --compare`: passed.
- `bash -n scripts/*.sh environment/setup.sh`: passed.
- `python3 -m py_compile` over tracked Python sources: passed.
- Shared-path scan across method, eval, baseline, scripts, data, environment, scoring, and briefing: passed.

## Final Judgment

The workspace is ready to serve as a research gym. From-scratch agents can use the problem, data, and evaluator without importing the paper method, and improve-mode agents can add new method artifact directories that the suite evaluator discovers automatically.

- Milestone: core_claim
- Ready for gym use: yes
- Confidence: medium
- Key limitation: the reproduction validates the core claim on scaled synthetic paired activations, not the original full Gemma 2 activation dataset.
