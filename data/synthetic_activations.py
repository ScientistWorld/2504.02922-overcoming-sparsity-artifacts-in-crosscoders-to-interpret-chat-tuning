"""Synthetic paired activation benchmark for crosscoder model diffing.

The paper trains on residual-stream activations from base/chat Gemma models.
This generator is a scaled-down substitute for those activations: it creates
paired base/chat vectors from sparse ground-truth concepts with shared,
base-only, chat-only, and weak-shared factors. The weak-shared factors are
present in both models but smaller in base, making them vulnerable to the L1
crosscoder sparsity artifact discussed in the paper.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _unit_columns(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    x = rng.normal(size=(n, d)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return x


def _sparse_positive(
    rng: np.random.Generator, n_samples: int, n_features: int, rate: float
) -> np.ndarray:
    mask = rng.random((n_samples, n_features)) < rate
    values = rng.gamma(shape=2.0, scale=1.0, size=(n_samples, n_features))
    return (mask * values).astype(np.float32)


def generate_dataset(
    out_dir: Path,
    seed: int = 0,
    n_train: int = 4096,
    n_val: int = 1024,
    d_model: int = 32,
    n_shared: int = 16,
    n_weak_shared: int = 12,
    n_chat_only: int = 8,
    n_base_only: int = 8,
    activation_rate: float = 0.08,
    base_weak_scale: float = 0.28,
    noise_std: float = 0.03,
) -> None:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = {
        "shared": n_shared,
        "weak_shared": n_weak_shared,
        "chat_only": n_chat_only,
        "base_only": n_base_only,
    }
    n_total = sum(counts.values())
    directions = _unit_columns(rng, n_total, d_model)

    idx = {}
    start = 0
    for name, count in counts.items():
        idx[name] = list(range(start, start + count))
        start += count

    def make_split(n_samples: int) -> dict[str, np.ndarray]:
        coeff = _sparse_positive(rng, n_samples, n_total, activation_rate)
        base_scale = np.ones(n_total, dtype=np.float32)
        chat_scale = np.ones(n_total, dtype=np.float32)
        base_scale[idx["weak_shared"]] = base_weak_scale
        base_scale[idx["chat_only"]] = 0.0
        chat_scale[idx["base_only"]] = 0.0

        base = (coeff * base_scale) @ directions
        chat = (coeff * chat_scale) @ directions
        base += rng.normal(scale=noise_std, size=base.shape).astype(np.float32)
        chat += rng.normal(scale=noise_std, size=chat.shape).astype(np.float32)
        return {
            "base": base.astype(np.float32),
            "chat": chat.astype(np.float32),
            "coefficients": coeff.astype(np.float32),
        }

    train = make_split(n_train)
    val = make_split(n_val)
    np.savez_compressed(
        out_dir / "synthetic_activations.npz",
        train_base=train["base"],
        train_chat=train["chat"],
        train_coefficients=train["coefficients"],
        val_base=val["base"],
        val_chat=val["chat"],
        val_coefficients=val["coefficients"],
        directions=directions.astype(np.float32),
        base_scale=np.concatenate(
            [
                np.ones(n_shared, dtype=np.float32),
                np.full(n_weak_shared, base_weak_scale, dtype=np.float32),
                np.zeros(n_chat_only, dtype=np.float32),
                np.ones(n_base_only, dtype=np.float32),
            ]
        ),
        chat_scale=np.concatenate(
            [
                np.ones(n_shared, dtype=np.float32),
                np.ones(n_weak_shared, dtype=np.float32),
                np.ones(n_chat_only, dtype=np.float32),
                np.zeros(n_base_only, dtype=np.float32),
            ]
        ),
    )
    metadata = {
        "seed": seed,
        "n_train": n_train,
        "n_val": n_val,
        "d_model": d_model,
        "counts": counts,
        "concept_indices": idx,
        "activation_rate": activation_rate,
        "base_weak_scale": base_weak_scale,
        "noise_std": noise_std,
    }
    (out_dir / "synthetic_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=4096)
    parser.add_argument("--n-val", type=int, default=1024)
    args = parser.parse_args()
    generate_dataset(args.out_dir, args.seed, args.n_train, args.n_val)


if __name__ == "__main__":
    main()
