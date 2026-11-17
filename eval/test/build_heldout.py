"""Build the held-out worlds used by the hidden test (run once, idempotent).

Writes, per experiment:

* ``eval/test/worlds/<experiment>.npz``  -- the hidden ground truth (concept
  directions, families, per-concept strengths and rates). Never shipped to
  methods.
* ``data/worlds/<experiment>/heldout/train.npz`` -- unlabeled paired training
  activations sampled from that world. This is the only view of the held-out
  world a method gets; it contains no coefficients, directions or labels.

The test inputs are sampled separately, at evaluation time, by
``eval/test/evaluate.py``. This module is self-contained: it shares no code
with ``data/`` or ``eval/train/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WORLD_DIR = Path(__file__).resolve().parent / "worlds"
FAMILIES = ("shared", "weak_shared", "decoupled", "chat_only", "base_only")
N_TRAIN = 65_536
DATA_VERSION = 2

# Secret 126-bit world seeds. Training activations use stream [seed, 1];
# hidden test inputs use stream [seed, 2] (see evaluate.py).
WORLD_SEEDS = {
    "synthetic_chat_specificity": 48792347030620228613965586186185244631,
    "synthetic_faint_chat_concepts": 62284550774079601701689413857518662305,
}

# Same generator family as the public dev worlds, with perturbed family sizes
# (the number of fine-tuned-specific targets is kept at 32).
PARAMS = {
    "synthetic_chat_specificity": {
        "d_model": 64,
        "counts": {"shared": 46, "weak_shared": 18, "decoupled": 16, "chat_only": 16, "base_only": 16},
        "rate": {"shared": 0.05, "weak_shared": 0.05, "chat_only": 0.05, "base_only": 0.05},
        "decoupled_rates": [0.12, 0.012, 0.012],
        "weak_base_scale": [0.3, 0.6],
        "chat_only_scale": [1.0, 1.0],
        "concept_gain": [0.7, 1.3],
        "noise_std": 0.05,
    },
    "synthetic_faint_chat_concepts": {
        "d_model": 64,
        "counts": {"shared": 50, "weak_shared": 14, "decoupled": 16, "chat_only": 16, "base_only": 16},
        "rate": {"shared": 0.06, "weak_shared": 0.05, "chat_only": 0.05, "base_only": 0.05},
        "decoupled_rates": [0.12, 0.012, 0.012],
        "weak_base_scale": [0.3, 0.6],
        "chat_only_scale": [0.25, 0.65],
        "concept_gain": [0.7, 1.3],
        "noise_std": 0.05,
    },
}


def _sample_world(params: dict, rng: np.random.Generator) -> dict[str, np.ndarray]:
    d = int(params["d_model"])
    counts = [int(params["counts"][f]) for f in FAMILIES]
    n = sum(counts)
    family = np.concatenate([np.full(c, i, dtype=np.int64) for i, c in enumerate(counts)])
    order = rng.permutation(n)  # hide family structure in concept order
    family = family[order]
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
    return {
        "directions": directions.astype(np.float32),
        "family": family,
        "gain": gain,
        "base_scale": base_scale,
        "chat_scale": chat_scale,
        "rate": rate,
        "rate_base_excl": np.where(decoupled, r_base, 0.0),
        "rate_chat_excl": np.where(decoupled, r_chat, 0.0),
        "noise_std": np.float64(params["noise_std"]),
    }


def sample_heldout(world: dict, n: int, rng: np.random.Generator):
    """(base, chat, base_coeff, chat_coeff) for ``n`` paired held-out inputs."""
    def sparse(rates):
        rates = np.asarray(rates, dtype=np.float64)
        active = rng.random((n, rates.shape[0])) < rates[None, :]
        return active * rng.gamma(2.0, 1.0, size=(n, rates.shape[0]))

    both = sparse(world["rate"])
    base_excl = sparse(world["rate_base_excl"])
    chat_excl = sparse(world["rate_chat_excl"])
    gain = np.asarray(world["gain"], dtype=np.float64)
    base_coeff = gain * (both * world["base_scale"] + base_excl)
    chat_coeff = gain * (both * world["chat_scale"] + chat_excl)
    directions = np.asarray(world["directions"], dtype=np.float64)
    noise = float(world["noise_std"])
    base = base_coeff @ directions + rng.normal(scale=noise, size=(n, directions.shape[1]))
    chat = chat_coeff @ directions + rng.normal(scale=noise, size=(n, directions.shape[1]))
    return base, chat, base_coeff, chat_coeff


def build(experiment: str) -> None:
    world_path = WORLD_DIR / f"{experiment}.npz"
    train_dir = ROOT / "data" / "worlds" / experiment / "heldout"
    train_path = train_dir / "train.npz"
    seed = WORLD_SEEDS[experiment]
    info_path = train_dir / "info.json"
    if world_path.exists() and train_path.exists() and info_path.exists() and json.loads(info_path.read_text()).get("data_version") == DATA_VERSION:
        print(f"[heldout] {experiment}: present; skipping")
        return
    if world_path.exists() != train_path.exists():
        raise RuntimeError(
            f"{experiment}: held-out world and training activations must be built together; "
            f"found world={world_path.exists()} train={train_path.exists()}"
        )
    world = _sample_world(PARAMS[experiment], np.random.default_rng(seed))
    base, chat, _, _ = sample_heldout(world, N_TRAIN, np.random.default_rng([seed, 1]))
    WORLD_DIR.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(world_path, **world)
    np.savez_compressed(train_path, base=base.astype(np.float16), chat=chat.astype(np.float16))
    (train_dir / "info.json").write_text(
        json.dumps({"data_version": DATA_VERSION, "experiment": experiment, "n_train": N_TRAIN, "d_model": int(world["directions"].shape[1]),
                    "note": "Unlabeled paired activations from the held-out world scored by the hidden test."},
                   indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"[heldout] {experiment}: wrote {world_path.relative_to(ROOT)} and {train_path.relative_to(ROOT)}")


def main() -> None:
    for experiment in PARAMS:
        build(experiment)


if __name__ == "__main__":
    main()
