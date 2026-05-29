"""Synthetic paired activation benchmark for crosscoder model diffing.

The paper trains on residual-stream activations from base/chat Gemma models.
This generator is a scaled-down substitute for those activations: it creates
paired base/chat vectors from sparse ground-truth concepts with shared,
base-only, chat-only, and decoupled shared factors. The decoupled factors
follow the paper's Appendix D setup: one concept direction appears on shared,
base-exclusive, and chat-exclusive subsets. A norm-difference classifier can
therefore mistake part of a shared concept for a chat-only latent.
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
    scenario: str = "decoupling",
    n_train: int = 4096,
    n_val: int = 1024,
    d_model: int = 32,
    n_shared: int = 16,
    n_weak_shared: int = 12,
    n_chat_only: int = 8,
    n_base_only: int = 8,
    activation_rate: float = 0.08,
    base_weak_scale: float = 1.0,
    decoupled_shared_rate: float = 0.07,
    decoupled_exclusive_rate: float = 0.055,
    noise_std: float = 0.03,
) -> None:
    if scenario not in {"decoupling", "complete_shrinkage"}:
        raise ValueError(f"unknown scenario: {scenario}")
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
        base_coeff = np.zeros((n_samples, n_total), dtype=np.float32)
        chat_coeff = np.zeros((n_samples, n_total), dtype=np.float32)

        shared = idx["shared"]
        weak = idx["weak_shared"]
        chat_only = idx["chat_only"]
        base_only = idx["base_only"]

        shared_coeff = _sparse_positive(rng, n_samples, len(shared), activation_rate)
        base_coeff[:, shared] = shared_coeff
        chat_coeff[:, shared] = shared_coeff

        if scenario == "decoupling":
            # Appendix-D latent decoupling substitute: each direction is active
            # in both models on a shared subset, plus separately active on
            # base-only and chat-only subsets. The concept itself is not novel
            # to the chat model, but a crosscoder can represent the
            # chat-exclusive subset with a high norm-difference latent.
            dec_shared = _sparse_positive(
                rng, n_samples, len(weak), decoupled_shared_rate
            )
            dec_base = _sparse_positive(
                rng, n_samples, len(weak), decoupled_exclusive_rate
            )
            dec_chat = _sparse_positive(
                rng, n_samples, len(weak), decoupled_exclusive_rate
            )
            base_coeff[:, weak] = dec_shared + dec_base
            chat_coeff[:, weak] = dec_shared + dec_chat
        else:
            # Complete-shrinkage substitute: the same weak-shared directions
            # appear in both streams, but are lower-amplitude in the base
            # stream. L1 pressure can zero the base decoder for these latents
            # even though they still explain base residual error.
            weak_coeff = _sparse_positive(rng, n_samples, len(weak), activation_rate)
            base_coeff[:, weak] = base_weak_scale * weak_coeff
            chat_coeff[:, weak] = weak_coeff

        chat_coeff[:, chat_only] = _sparse_positive(
            rng, n_samples, len(chat_only), activation_rate
        )
        base_coeff[:, base_only] = _sparse_positive(
            rng, n_samples, len(base_only), activation_rate
        )

        base = base_coeff @ directions
        chat = chat_coeff @ directions
        base += rng.normal(scale=noise_std, size=base.shape).astype(np.float32)
        chat += rng.normal(scale=noise_std, size=chat.shape).astype(np.float32)
        return {
            "base": base.astype(np.float32),
            "chat": chat.astype(np.float32),
            "coefficients": np.maximum(base_coeff, chat_coeff).astype(np.float32),
            "base_coefficients": base_coeff.astype(np.float32),
            "chat_coefficients": chat_coeff.astype(np.float32),
        }

    train = make_split(n_train)
    val = make_split(n_val)
    np.savez_compressed(
        out_dir / "synthetic_activations.npz",
        train_base=train["base"],
        train_chat=train["chat"],
        train_coefficients=train["coefficients"],
        train_base_coefficients=train["base_coefficients"],
        train_chat_coefficients=train["chat_coefficients"],
        val_base=val["base"],
        val_chat=val["chat"],
        val_coefficients=val["coefficients"],
        val_base_coefficients=val["base_coefficients"],
        val_chat_coefficients=val["chat_coefficients"],
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
        "scenario": scenario,
        "n_train": n_train,
        "n_val": n_val,
        "d_model": d_model,
        "counts": counts,
        "concept_indices": idx,
        "activation_rate": activation_rate,
        "base_weak_scale": base_weak_scale,
        "decoupled_shared_rate": decoupled_shared_rate,
        "decoupled_exclusive_rate": decoupled_exclusive_rate,
        "noise_std": noise_std,
    }
    (out_dir / "synthetic_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--scenario",
        choices=("decoupling", "complete_shrinkage"),
        default="decoupling",
    )
    parser.add_argument("--n-train", type=int, default=4096)
    parser.add_argument("--n-val", type=int, default=1024)
    parser.add_argument("--base-weak-scale", type=float, default=None)
    args = parser.parse_args()
    base_weak_scale = (
        args.base_weak_scale
        if args.base_weak_scale is not None
        else (0.25 if args.scenario == "complete_shrinkage" else 1.0)
    )
    generate_dataset(
        args.out_dir,
        seed=args.seed,
        scenario=args.scenario,
        n_train=args.n_train,
        n_val=args.n_val,
        base_weak_scale=base_weak_scale,
    )


if __name__ == "__main__":
    main()
