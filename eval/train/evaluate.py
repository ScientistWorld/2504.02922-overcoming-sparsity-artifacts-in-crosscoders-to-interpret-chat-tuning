"""Train-side (development) scorer for paired-activation model-diffing dictionaries.

For every method directory under ``--methods-root`` (default
``checkpoints/methods``) and every experiment, this loads
``<method>/<experiment>/dev/crosscoder.npz`` -- a dictionary trained on
``data/worlds/<experiment>/dev/train.npz`` -- and scores it on fresh inputs
sampled from the public dev world ``data/worlds/<experiment>/dev/world.npz``.

The metric definitions and fixed constraints are the same as on the hidden
test, which scores dictionaries trained on the held-out worlds' activations.
Everything is recomputed from the submitted encoder/decoder parameters; a
missing or invalid artifact, a dictionary larger than ``D_MAX`` or a mean L0
above ``L0_MAX`` gives every metric of that experiment its worst value.

This module does not import from method/ or eval/test/.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

SLICE = "train"
WORLD_KEY = "dev"
ARTIFACT = "crosscoder.npz"
EXPERIMENTS = ("synthetic_chat_specificity", "synthetic_faint_chat_concepts")
FAMILIES = ("shared", "weak_shared", "decoupled", "chat_only", "base_only")
WORLDS_ROOT = Path("data/worlds")
VAL_SEEDS = {"synthetic_chat_specificity": 77_001, "synthetic_faint_chat_concepts": 77_002}

N_EVAL = 32_768
D_MAX = 256
L0_MAX = 8.8
FT_ONLY_DELTA = 0.9
ALIGN_COS = 0.8
SPECIFICITY_PI = 0.2
MIN_FIRING = 10
TOP_N_FACTOR = 1.5
LS_FALSE_NU = 0.6

WORST = {"false_ft_only_rate": 1.0, "chat_specific_recall": 0.0, "reconstruction_fve": 0.0}
REQUIRED = ("W_enc", "b_enc", "threshold", "W_dec_base", "W_dec_chat", "b_dec_base", "b_dec_chat")
TARGET_FAMILIES = ("chat_only", "decoupled")


class RunFailure(Exception):
    pass


def load_world(experiment: str) -> dict:
    with np.load(WORLDS_ROOT / experiment / WORLD_KEY / "world.npz", allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def sample_dev(world: dict, n: int, rng: np.random.Generator) -> dict:
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
    return {"base": base, "chat": chat, "base_coeff": base_coeff, "chat_coeff": chat_coeff}


def load_artifact(path: Path, d: int) -> dict:
    if not path.is_file():
        raise RunFailure(f"missing artifact {path}")
    try:
        with np.load(path, allow_pickle=False) as z:
            missing = [k for k in REQUIRED if k not in z.files]
            if missing:
                raise RunFailure(f"artifact lacks arrays {missing}")
            art = {k: np.asarray(z[k], dtype=np.float64) for k in REQUIRED}
            topk = 0
            if "topk" in z.files:
                raw = np.asarray(z["topk"])
                if raw.size != 1 or not np.issubdtype(raw.dtype, np.number):
                    raise RunFailure("topk must be a numeric scalar")
                value = float(raw.reshape(()))
                if not math.isfinite(value) or value < 0 or value != int(value):
                    raise RunFailure("topk must be a nonnegative integer")
                topk = int(value)
    except RunFailure:
        raise
    except Exception as exc:
        raise RunFailure(f"unreadable artifact: {type(exc).__name__}: {exc}") from exc
    w = art["W_enc"]
    if w.ndim != 2 or w.shape[0] != 2 * d:
        raise RunFailure(f"W_enc must have shape (2*{d}, D); got {w.shape}")
    n_lat = w.shape[1]
    if not 1 <= n_lat <= D_MAX:
        raise RunFailure(f"dictionary size {n_lat} outside [1, {D_MAX}]")
    expected = {
        "b_enc": (n_lat,), "threshold": (n_lat,), "W_dec_base": (n_lat, d),
        "W_dec_chat": (n_lat, d), "b_dec_base": (d,), "b_dec_chat": (d,),
    }
    for key, shape in expected.items():
        if art[key].shape != shape:
            raise RunFailure(f"{key} must have shape {shape}; got {art[key].shape}")
    for key, arr in art.items():
        if not np.all(np.isfinite(arr)):
            raise RunFailure(f"{key} contains non-finite values")
    art["topk"] = topk
    return art


def encode(art: dict, base: np.ndarray, chat: np.ndarray) -> np.ndarray:
    d = base.shape[1]
    w = art["W_enc"]
    pre = base @ w[:d] + chat @ w[d:] + art["b_enc"]
    f = np.where((pre > 0.0) & (pre > art["threshold"]), pre, 0.0)
    topk = art["topk"]
    if 0 < topk < f.shape[1]:
        score = f * (np.linalg.norm(art["W_dec_base"], axis=1) + np.linalg.norm(art["W_dec_chat"], axis=1))
        keep = np.argpartition(-score, topk - 1, axis=1)[:, :topk]
        mask = np.zeros_like(f, dtype=bool)
        np.put_along_axis(mask, keep, True, axis=1)
        f = np.where(mask, f, 0.0)
    return f


def score_artifact(art: dict, sample: dict, world: dict) -> tuple[dict, dict]:
    hb, hc, ab, ac = sample["base"], sample["chat"], sample["base_coeff"], sample["chat_coeff"]
    f = encode(art, hb, hc)
    l0 = float((f > 0).sum(axis=1).mean())
    if not math.isfinite(l0) or l0 > L0_MAX:
        raise RunFailure(f"mean L0 {l0:.3f} exceeds the sparsity budget {L0_MAX}")
    wdb, wdc = art["W_dec_base"], art["W_dec_chat"]
    rb = f @ wdb + art["b_dec_base"]
    rc = f @ wdc + art["b_dec_chat"]
    if not (np.all(np.isfinite(rb)) and np.all(np.isfinite(rc))):
        raise RunFailure("non-finite reconstruction")
    sse_b = float(((hb - rb) ** 2).sum())
    sse_c = float(((hc - rc) ** 2).sum())
    sst_b = float(((hb - hb.mean(axis=0)) ** 2).sum())
    sst_c = float(((hc - hc.mean(axis=0)) ** 2).sum())
    fve = 1.0 - (sse_b + sse_c) / (sst_b + sst_c)

    nb = np.linalg.norm(wdb, axis=1)
    nc = np.linalg.norm(wdc, axis=1)
    mx = np.maximum(nb, nc)
    delta = np.where(mx > 0, 0.5 * (1.0 + (nc - nb) / np.where(mx > 0, mx, 1.0)), 0.5)
    alive = (f > 0).sum(axis=0) >= MIN_FIRING
    mass = f.mean(axis=0) * nc

    directions = np.asarray(world["directions"], dtype=np.float64)
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    cos = (wdc / np.where(nc > 0, nc, 1.0)[:, None]) @ directions.T
    concept = cos.argmax(axis=1)
    aligned = (cos[np.arange(cos.shape[0]), concept] >= ALIGN_COS) & (nc > 0)

    idx = np.arange(f.shape[1])
    fb = (f.T @ ab)[idx, concept]
    fc = (f.T @ ac)[idx, concept]
    rho = np.where(fc > 0, fb / np.where(fc > 0, fc, 1.0), np.inf)
    genuine = alive & aligned & (rho < SPECIFICITY_PI)

    ft_only = alive & (delta >= FT_ONLY_DELTA)
    ft_mass = float(mass[ft_only].sum())
    false_rate = 1.0 if ft_mass <= 0 else float(mass[ft_only & ~genuine].sum()) / ft_mass

    family = np.asarray(world["family"])
    target_codes = [FAMILIES.index(name) for name in TARGET_FAMILIES]
    targets = {int(c) for c in np.flatnonzero(np.isin(family, target_codes))}
    top_n = int(math.ceil(TOP_N_FACTOR * len(targets)))
    alive_idx = np.flatnonzero(alive)
    order = alive_idx[np.lexsort((-mass[alive_idx], -delta[alive_idx]))]
    top = order[:top_n]
    covered = {int(concept[j]) for j in top if genuine[j]} & targets
    recall = len(covered) / max(len(targets), 1)

    metrics = {
        "false_ft_only_rate": float(false_rate),
        "chat_specific_recall": float(recall),
        "reconstruction_fve": float(min(max(fve, 0.0), 1.0)),
    }
    if not all(math.isfinite(v) for v in metrics.values()):
        raise RunFailure(f"non-finite metric {metrics}")

    sq = (f ** 2).sum(axis=0)
    denom = sq * nc ** 2
    num_b_err = ((f.T @ (hb - rb)) * wdc).sum(axis=1) + sq * (wdb * wdc).sum(axis=1)
    num_c_err = ((f.T @ (hc - rc)) * wdc).sum(axis=1) + sq * nc ** 2
    num_b_rec = ((f.T @ rb) * wdc).sum(axis=1)
    num_c_rec = ((f.T @ rc) * wdc).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        nu_err = (num_b_err / denom) / (num_c_err / denom)
        nu_rec = (num_b_rec / denom) / (num_c_rec / denom)
    ls_false = (~np.isfinite(nu_err)) | (~np.isfinite(nu_rec)) | (nu_err > LS_FALSE_NU) | (nu_rec > LS_FALSE_NU)
    n_ft = int(ft_only.sum())
    spurious: dict[str, float] = {}
    for j in np.flatnonzero(ft_only & ~genuine):
        key = FAMILIES[int(family[concept[j]])] if aligned[j] else "unaligned"
        spurious[key] = spurious.get(key, 0.0) + float(mass[j])
    recall_by_family = {}
    for name in TARGET_FAMILIES:
        fam = {int(c) for c in np.flatnonzero(family == FAMILIES.index(name))}
        recall_by_family[name] = len(covered & fam) / max(len(fam), 1)
    diag = {
        "status": "ok",
        "n_latents": int(f.shape[1]),
        "n_alive": int(alive.sum()),
        "mean_l0": l0,
        "base_fve": 1.0 - sse_b / sst_b,
        "chat_fve": 1.0 - sse_c / sst_c,
        "n_ft_only": n_ft,
        "n_ft_only_genuine": int((ft_only & genuine).sum()),
        "ft_only_mass": ft_mass,
        "spurious_mass_fraction_by_family": {k: v / ft_mass for k, v in sorted(spurious.items())} if ft_mass > 0 else {},
        "recall_by_family": recall_by_family,
        "top_n": top_n,
        "n_genuine_in_top_n": int(sum(bool(genuine[j]) for j in top)),
        "latent_scaling_false_ft_only_fraction": (float(ls_false[ft_only].mean()) if n_ft else None),
    }
    return metrics, diag


# Frozen dev-world artifacts of the reference and baseline, used when a
# workspace has no artifacts of its own under these names.
FROZEN = Path(__file__).resolve().parent / "reference_methods"


def discover_methods(root: Path) -> list[str]:
    names: set[str] = set()
    for base in (root, FROZEN):
        if base.is_dir():
            names |= {p.name for p in base.iterdir() if p.is_dir() and not p.name.startswith(".")}
    return sorted(names)


def artifact_path(root: Path, method: str, experiment: str) -> Path:
    base = root if (root / method).is_dir() else FROZEN
    return base / method / experiment / WORLD_KEY / ARTIFACT


def evaluate(root: Path) -> tuple[dict, dict]:
    methods = discover_methods(root)
    scores: dict = {"slice": SLICE, "experiments": {}}
    diagnostics: dict = {"slice": SLICE, "experiments": {}}
    for experiment in EXPERIMENTS:
        world = load_world(experiment)
        sample = sample_dev(world, N_EVAL, np.random.default_rng(VAL_SEEDS[experiment]))
        d = sample["base"].shape[1]
        results, diags = {}, {}
        for method in methods:
            path = artifact_path(root, method, experiment)
            try:
                art = load_artifact(path, d)
                metrics, diag = score_artifact(art, sample, world)
            except RunFailure as exc:
                metrics, diag = dict(WORST), {"status": "failed", "reason": str(exc)}
            except Exception as exc:
                metrics, diag = dict(WORST), {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
            results[method] = metrics
            diags[method] = diag
            print(f"[{SLICE}] {experiment} {method}: {metrics} ({diag.get('status')})", flush=True)
        scores["experiments"][experiment] = {"results": results}
        diagnostics["experiments"][experiment] = diags
    return scores, diagnostics


def _clean(obj):
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods-root", type=Path, default=Path("checkpoints/methods"))
    parser.add_argument("--out", type=Path, default=Path("scoring/scores_train.json"))
    parser.add_argument("--diagnostics", type=Path, default=None)
    args = parser.parse_args()
    scores, diagnostics = evaluate(args.methods_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(scores, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.diagnostics is not None:
        args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
        args.diagnostics.write_text(json.dumps(_clean(diagnostics), indent=2, allow_nan=False) + "\n",
                                    encoding="utf-8")


if __name__ == "__main__":
    main()
