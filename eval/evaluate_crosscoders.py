"""Evaluate paired activation-factor artifacts without importing from method/."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _relative_norm_difference(base_decoder: np.ndarray, chat_decoder: np.ndarray) -> np.ndarray:
    base = np.linalg.norm(base_decoder, axis=1)
    chat = np.linalg.norm(chat_decoder, axis=1)
    denom = np.maximum(np.maximum(base, chat), 1e-8)
    return 0.5 * (1.0 + (chat - base) / denom)


def _beta(features: np.ndarray, decoder: np.ndarray, target: np.ndarray) -> np.ndarray:
    yd = target @ decoder.T
    numerator = (yd * features).sum(axis=0)
    denominator = (features * features).sum(axis=0) * (decoder * decoder).sum(axis=1)
    return numerator / np.maximum(denominator, 1e-12)


def _latent_scaling_metrics(
    features: np.ndarray,
    base_decoder: np.ndarray,
    chat_decoder: np.ndarray,
    base_recon: np.ndarray,
    chat_recon: np.ndarray,
    base: np.ndarray,
    chat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    beta_r_base = _beta(features, chat_decoder, base_recon)
    beta_r_chat = _beta(features, chat_decoder, chat_recon)

    beta_e_base = _beta(features, chat_decoder, base - base_recon)
    beta_e_chat = _beta(features, chat_decoder, chat - chat_recon)

    nu_r = beta_r_base / np.maximum(np.abs(beta_r_chat), 1e-8)
    nu_e = beta_e_base / np.maximum(np.abs(beta_e_chat), 1e-8)
    return nu_r, nu_e


def _best_alignment(decoder: np.ndarray, directions: np.ndarray, indices: list[int]) -> np.ndarray:
    if not indices:
        return np.zeros(decoder.shape[0], dtype=np.float32)
    dec = decoder / (np.linalg.norm(decoder, axis=1, keepdims=True) + 1e-8)
    dirs = directions[indices]
    dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8)
    return np.abs(dec @ dirs.T).max(axis=1)


def discover_methods(artifacts: Path) -> list[str]:
    if (artifacts / "val_outputs.npz").exists():
        return [artifacts.name]
    if not artifacts.exists():
        return []
    return sorted(
        path.name
        for path in artifacts.iterdir()
        if path.is_dir() and (path / "val_outputs.npz").exists()
    )


def run_dir_for_method(artifacts: Path, method_name: str) -> Path:
    if (artifacts / "val_outputs.npz").exists() and artifacts.name == method_name:
        return artifacts
    return artifacts / method_name


def evaluate_one(run_dir: Path, data: np.lib.npyio.NpzFile, meta: dict) -> dict[str, float]:
    out = np.load(run_dir / "val_outputs.npz")
    features = out["features"]
    base_recon = out["base_recon"]
    chat_recon = out["chat_recon"]
    base_decoder = out["base_decoder"]
    chat_decoder = out["chat_decoder"]
    base = data["val_base"]
    chat = data["val_chat"]
    directions = data["directions"]
    idx = meta["concept_indices"]

    delta = _relative_norm_difference(base_decoder, chat_decoder)
    nu_r, nu_e = _latent_scaling_metrics(
        features, base_decoder, chat_decoder, base_recon, chat_recon, base, chat
    )

    top_n = max(1, int(round(len(delta) * 0.15)))
    top_chat = np.argsort(-delta)[:top_n]
    ft_only = delta >= 0.9
    true_chat_alignment = _best_alignment(chat_decoder, directions, idx["chat_only"])
    weak_shared_alignment = _best_alignment(chat_decoder, directions, idx["weak_shared"])
    shared_alignment = _best_alignment(chat_decoder, directions, idx["shared"])

    aligned_true = true_chat_alignment[top_chat] >= 0.45
    aligned_weak = weak_shared_alignment[top_chat] >= 0.45
    aligned_shared = shared_alignment[top_chat] >= 0.45
    scaling_specific = (nu_r[top_chat] < 0.5) & (nu_e[top_chat] < 0.2)
    ft_only_specific_pi06 = (nu_r[ft_only] < 0.6) & (nu_e[ft_only] < 0.6)
    false_ft_only_rate_pi06 = (
        1.0 - float(ft_only_specific_pi06.mean()) if np.any(ft_only) else 0.0
    )

    mse_base = float(np.mean((base - base_recon) ** 2))
    mse_chat = float(np.mean((chat - chat_recon) ** 2))
    base_fve = 1.0 - (mse_base / max(float(np.var(base)), 1e-12))
    chat_fve = 1.0 - (mse_chat / max(float(np.var(chat)), 1e-12))
    variance = 0.5 * float(np.var(base) + np.var(chat))
    fve = 1.0 - (0.5 * (mse_base + mse_chat) / max(variance, 1e-12))

    return {
        "fraction_top_norm_true_chat": float(aligned_true.mean()),
        "fraction_top_norm_weak_or_shared": float((aligned_weak | aligned_shared).mean()),
        "fraction_top_norm_scaling_specific": float(scaling_specific.mean()),
        "chat_only_latent_count": int(np.sum(delta >= 0.9)),
        "false_ft_only_rate_pi06": false_ft_only_rate_pi06,
        "reconstruction_fve": float(fve),
        "base_reconstruction_fve": float(base_fve),
        "chat_reconstruction_fve": float(chat_fve),
        "mean_l0": float((features > 0).sum(axis=1).mean()),
    }


def evaluate_artifacts(
    artifacts: Path,
    data_path: Path,
    metadata_path: Path,
    experiment_name: str,
) -> dict:
    data = np.load(data_path)
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    results = {}
    for name in discover_methods(artifacts):
        results[name] = evaluate_one(run_dir_for_method(artifacts, name), data, meta)
    return {"experiments": {experiment_name: {"results": results}}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("checkpoints/crosscoder_synthetic"))
    parser.add_argument("--data", type=Path, default=Path("data/synthetic_activations.npz"))
    parser.add_argument("--metadata", type=Path, default=Path("data/synthetic_metadata.json"))
    parser.add_argument("--out", type=Path, default=Path("scoring/scores.json"))
    parser.add_argument("--experiment-name", default="synthetic_chat_specificity")
    args = parser.parse_args()

    scores = evaluate_artifacts(
        args.artifacts, args.data, args.metadata, args.experiment_name
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
