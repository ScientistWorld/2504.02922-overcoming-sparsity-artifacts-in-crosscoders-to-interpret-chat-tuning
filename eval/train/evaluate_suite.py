"""Evaluate the visible train slice for all paired activation-factor cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval.train.evaluate_crosscoders import (
    SLICE_NAME,
    SPLIT_SEED,
    discover_methods,
    evaluate_one,
)


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
    parser.add_argument("--out", type=Path, default=Path("scoring/scores_train.json"))
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
            results[method_name] = evaluate_one(artifact_dir / method_name, data, meta)
        experiments[experiment_name] = {"results": results}

    scores = {
        "slice": SLICE_NAME,
        "split": {
            "strategy": "fixed 50/50 validation-sample and concept-family partition",
            "seed": SPLIT_SEED,
        },
        "experiments": experiments,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
