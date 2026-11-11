"""Public generator for the development ("dev") paired-activation worlds.

The paper studies paired residual-stream activations of a base model and a
fine-tuned (chat) model. This benchmark replaces the language models with a
synthetic world: a dictionary of ground-truth concept directions in R^d whose
sparse, nonnegative activations are summed (plus noise) into a base stream and
a chat stream. Concept families:

* ``shared``       present with the same strength in both streams;
* ``weak_shared``  present in both streams, but weaker in the base stream
                   (base strength is a per-concept fraction of chat strength);
* ``decoupled``    one direction used in both streams, but on partially
                   different inputs: a subset where both streams use it, plus
                   a base-exclusive and a chat-exclusive subset;
* ``chat_only``    present only in the chat stream (genuine fine-tuning
                   change);
* ``base_only``    present only in the base stream.

This file generates the *dev* worlds whose ground truth is public; methods may
use them for development and they are what ``scripts/evaluate_train.sh``
scores. The held-out worlds used by the hidden test come from the same family
of generators but with different, undisclosed concepts; only their training
activations are shipped (``data/worlds/<experiment>/heldout/train.npz``).

Usage (idempotent):  python3 data/synthetic_activations.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FAMILIES = ("shared", "weak_shared", "decoupled", "chat_only", "base_only")

# Family parameters of each experiment's dev world.
EXPERIMENTS: dict[str, dict] = {
    # Both shared-factor failure modes (strength asymmetry and partially
    # separated occurrence) next to genuine chat-only concepts.
    "synthetic_chat_specificity": {
        "d_model": 64,
        "counts": {"shared": 48, "weak_shared": 16, "decoupled": 16, "chat_only": 16, "base_only": 16},
        "rate": {"shared": 0.05, "weak_shared": 0.05, "chat_only": 0.05, "base_only": 0.05},
        "decoupled_rates": [0.04, 0.03, 0.03],  # both-stream subset, base-exclusive, chat-exclusive
        "weak_base_scale": [0.3, 0.6],
        "chat_only_scale": [1.0, 1.0],
        "concept_gain": [0.7, 1.3],
        "noise_std": 0.05,
    },
    # Same families, but denser inputs and faint chat-only concepts: the
    # genuinely fine-tuned-specific structure carries little variance.
    "synthetic_faint_chat_concepts": {
        "d_model": 64,
        "counts": {"shared": 48, "weak_shared": 16, "decoupled": 16, "chat_only": 16, "base_only": 16},
        "rate": {"shared": 0.08, "weak_shared": 0.05, "chat_only": 0.05, "base_only": 0.05},
        "decoupled_rates": [0.04, 0.03, 0.03],
        "weak_base_scale": [0.3, 0.6],
        "chat_only_scale": [0.15, 0.45],
        "concept_gain": [0.7, 1.3],
        "noise_std": 0.05,
    },
}

DEV_SEEDS = {"synthetic_chat_specificity": 11_011, "synthetic_faint_chat_concepts": 22_022}
N_TRAIN = 32_768
WORLDS_DIR = Path(__file__).resolve().parent / "worlds"


def sample_world(params: dict, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Draw concept directions, family labels and per-concept strengths."""
    d = int(params["d_model"])
    counts = [int(params["counts"][f]) for f in FAMILIES]
    n = sum(counts)
    family = np.concatenate([np.full(c, i, dtype=np.int64) for i, c in enumerate(counts)])
    directions = rng.standard_normal((n, d))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    gain = rng.uniform(*params["concept_gain"], size=n)
    base_scale = np.ones(n)
    chat_scale = np.ones(n)
    weak = family == FAMILIES.index("weak_shared")
    base_scale[weak] = rng.uniform(*params["weak_base_scale"], size=int(weak.sum()))
    chat_only = family == FAMILIES.index("chat_only")
    chat_scale[chat_only] = rng.uniform(*params["chat_only_scale"], size=int(chat_only.sum()))
    base_scale[chat_only] = 0.0
    base_only = family == FAMILIES.index("base_only")
    chat_scale[base_only] = 0.0

    decoupled = family == FAMILIES.index("decoupled")
    rate = np.zeros(n)
    for i, name in enumerate(FAMILIES):
        if name != "decoupled":
            rate[family == i] = float(params["rate"][name])
    r_both, r_base, r_chat = (float(x) for x in params["decoupled_rates"])
    rate[decoupled] = r_both
    rate_base_excl = np.where(decoupled, r_base, 0.0)
    rate_chat_excl = np.where(decoupled, r_chat, 0.0)
    return {
        "directions": directions.astype(np.float32),
        "family": family,
        "gain": gain,
        "base_scale": base_scale,
        "chat_scale": chat_scale,
        "rate": rate,
        "rate_base_excl": rate_base_excl,
        "rate_chat_excl": rate_chat_excl,
        "noise_std": np.float64(params["noise_std"]),
    }


def _sparse_gamma(rng: np.random.Generator, n: int, rates: np.ndarray) -> np.ndarray:
    active = rng.random((n, rates.shape[0])) < rates[None, :]
    return active * rng.gamma(2.0, 1.0, size=(n, rates.shape[0]))


def sample_activations(world: dict, n: int, rng: np.random.Generator):
    """Return (base, chat, base_coeff, chat_coeff) for ``n`` paired inputs."""
    both = _sparse_gamma(rng, n, np.asarray(world["rate"], dtype=np.float64))
    base_excl = _sparse_gamma(rng, n, np.asarray(world["rate_base_excl"], dtype=np.float64))
    chat_excl = _sparse_gamma(rng, n, np.asarray(world["rate_chat_excl"], dtype=np.float64))
    gain = np.asarray(world["gain"], dtype=np.float64)
    base_coeff = gain * (both * world["base_scale"] + base_excl)
    chat_coeff = gain * (both * world["chat_scale"] + chat_excl)
    directions = np.asarray(world["directions"], dtype=np.float64)
    noise = float(world["noise_std"])
    base = base_coeff @ directions + rng.normal(scale=noise, size=(n, directions.shape[1]))
    chat = chat_coeff @ directions + rng.normal(scale=noise, size=(n, directions.shape[1]))
    return base, chat, base_coeff, chat_coeff


def load_world(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def write_dev_world(experiment: str, root: Path = WORLDS_DIR, force: bool = False) -> None:
    out = root / experiment / "dev"
    if not force and (out / "world.npz").exists() and (out / "train.npz").exists():
        print(f"[data] {experiment}/dev exists; skipping")
        return
    out.mkdir(parents=True, exist_ok=True)
    params = EXPERIMENTS[experiment]
    seed = DEV_SEEDS[experiment]
    world = sample_world(params, np.random.default_rng(seed))
    base, chat, _, _ = sample_activations(world, N_TRAIN, np.random.default_rng([seed, 1]))
    np.savez_compressed(out / "world.npz", **world)
    np.savez_compressed(out / "train.npz", base=base.astype(np.float16), chat=chat.astype(np.float16))
    taxonomy = {
        name: [int(i) for i in np.flatnonzero(world["family"] == idx)]
        for idx, name in enumerate(FAMILIES)
    }
    (out / "world.json").write_text(
        json.dumps({"experiment": experiment, "seed": seed, "n_train": N_TRAIN,
                    "params": params, "families": list(FAMILIES), "taxonomy": taxonomy}, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"[data] wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for experiment in EXPERIMENTS:
        write_dev_world(experiment, force=args.force)


if __name__ == "__main__":
    main()
