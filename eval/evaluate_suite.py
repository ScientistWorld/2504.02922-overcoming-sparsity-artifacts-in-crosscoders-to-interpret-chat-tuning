"""Evaluate a suite of paired activation-factor benchmark cases.

This module stays inside eval/ and only reads saved method artifacts. Each case
uses the same output contract:
  <artifacts-root>/<case-name>/<method>/val_outputs.npz
  <data-root>/<case-name>/synthetic_activations.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval.evaluate_crosscoders import discover_methods, evaluate_one, run_dir_for_method


CASES = {
    "synthetic_chat_specificity": "decoupling",
    "synthetic_complete_shrinkage": "complete_shrinkage",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts-root", type=Path, default=Path("checkpoints/crosscoder_suite")
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/benchmarks"))
    parser.add_argument("--out", type=Path, default=Path("scoring/scores.json"))
    args = parser.parse_args()

    experiments = {}
    for experiment_name, case_name in CASES.items():
        data_dir = args.data_root / case_name
        artifact_dir = args.artifacts_root / case_name
        data = np.load(data_dir / "synthetic_activations.npz")
        meta = json.loads(
            (data_dir / "synthetic_metadata.json").read_text(encoding="utf-8")
        )
        results = {}
        for method_name in discover_methods(artifact_dir):
            run_dir = run_dir_for_method(artifact_dir, method_name)
            results[method_name] = evaluate_one(run_dir, data, meta)
        experiments[experiment_name] = {"results": results}

    scores = {"experiments": experiments}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
