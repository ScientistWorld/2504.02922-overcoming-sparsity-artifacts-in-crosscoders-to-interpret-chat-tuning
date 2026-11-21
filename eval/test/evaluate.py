"""Hidden-test scorer for paired-activation model-diffing dictionaries.

For every method directory under ``--methods-root`` (default
``checkpoints/methods``) and every experiment, this loads
``<method>/<experiment>/heldout/crosscoder.npz`` -- a dictionary trained on the
unlabeled held-out training activations -- and scores it on fresh inputs that
are sampled here, at test time, from the hidden world in ``eval/test/worlds``.

Nothing the method wrote is trusted: activations, reconstructions, decoder
norm statistics and every metric are recomputed from the submitted encoder and
decoder parameters. A missing, malformed or non-finite artifact, a dictionary
larger than ``D_MAX`` or a mean L0 above ``L0_MAX`` scores the complete suite as a
failed run: every metric receives its worst value.

This module is self-contained (no imports from method/, data/ or eval/train/).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from eval.test.build_heldout import FAMILIES, WORLD_DIR, WORLD_SEEDS, sample_heldout

SLICE = "test"
WORLD_KEY = "heldout"
ARTIFACT = "crosscoder.npz"
EXPERIMENTS = ("synthetic_chat_specificity", "synthetic_faint_chat_concepts")

# Fixed comparison settings (identical on the train side).
N_EVAL = 524_288          # paired inputs scored per experiment
D_MAX = 512              # maximum dictionary size
L0_MAX = 6.6             # maximum mean number of active latents per input
FT_ONLY_DELTA = 0.9      # relative decoder norm difference labelling a latent fine-tuned-only
ALIGN_COS = 0.8          # min cosine between a latent's chat decoder and a concept direction
SPECIFICITY_PI = 0.2     # max base/chat presence ratio for a genuinely chat-specific latent
MIN_FIRING_RATE = 0.005 # require support on at least 0.5% of scored inputs
FVE_MIN = 0.60
RECALL_MIN = 0.10
LS_FALSE_NU = 0.6        # diagnostic only: Latent Scaling false fine-tuned-only threshold

WORST = {"false_ft_only_rate": 1.0, "chat_specific_recall": 0.0, "reconstruction_fve": 0.0}
REQUIRED = ("W_enc", "b_enc", "threshold", "W_dec_base", "W_dec_chat", "b_dec_base", "b_dec_chat")
TARGET_FAMILIES = ("chat_only", "decoupled")


class RunFailure(Exception):
    """Invalid run; optional recomputed diagnostics never enter ranked results."""

    def __init__(self, reason, diagnostics=None):
        super().__init__(reason)
        self.diagnostics = diagnostics or {}


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
    except Exception as exc:  # unreadable / pickled / wrong dtype
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
    encoder_path = path.with_name("encoder.onnx")
    if encoder_path.exists():
        # A portable tensor program lets an attempt supply its own inference
        # procedure without executing Python with access to the hidden world.
        if encoder_path.stat().st_size > 64 * 1024 * 1024:
            raise RunFailure("encoder.onnx exceeds 64 MiB")
        import onnx
        raw = encoder_path.read_bytes()
        model = onnx.load_model_from_string(raw)
        def check_message(message):
            if hasattr(message, "external_data") and message.external_data:
                raise RunFailure("ONNX external data is forbidden; embed tensor constants")
            for field, value in message.ListFields():
                if field.message_type is not None:
                    if getattr(field, "is_repeated", False) or getattr(field, "label", None) == 3:
                        for item in value:
                            check_message(item)
                    else:
                        check_message(value)
        check_message(model)
        art["encoder_model"] = raw
    return art


def encode(art: dict, base: np.ndarray, chat: np.ndarray) -> np.ndarray:
    d = base.shape[1]
    w = art["W_enc"]
    if "encoder_model" in art:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        if "encoder_session" not in art:
            art["encoder_session"] = ort.InferenceSession(art["encoder_model"], sess_options=opts,
                                                        providers=["CPUExecutionProvider"])
        session = art["encoder_session"]
        inputs = session.get_inputs()
        if len(inputs) != 1 or inputs[0].type != "tensor(float)":
            raise RunFailure("encoder.onnx needs one float32 paired-input tensor")
        chunks = []
        for start in range(0, len(base), 1024):
            x = np.hstack([base[start:start+1024], chat[start:start+1024]]).astype(np.float32)
            outputs = session.run(None, {inputs[0].name: x})
            if len(outputs) != 1 or outputs[0].shape != (len(x), w.shape[1]):
                raise RunFailure("encoder.onnx must return one (batch, D) feature tensor")
            chunks.append(np.asarray(outputs[0], dtype=np.float64))
        f = np.concatenate(chunks)
    else:
        pre = base @ w[:d] + chat @ w[d:] + art["b_enc"]
        f = np.where((pre > 0.0) & (pre > art["threshold"]), pre, 0.0)
    if not np.isfinite(f).all() or np.any(f < 0):
        raise RunFailure("features must be finite and nonnegative")
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
    # Stream sufficient statistics so a larger paired test does not retain an
    # N-by-D feature matrix. No method-written summary is accepted.
    n, D = len(hb), art["W_enc"].shape[1]
    wdb, wdc = art["W_dec_base"], art["W_dec_chat"]
    sf, sq_f, fires = np.zeros(D), np.zeros(D), np.zeros(D, dtype=np.int64)
    fhb, fhc = np.zeros((D,hb.shape[1])), np.zeros((D,hc.shape[1]))
    fab, fac = np.zeros((D,ab.shape[1])), np.zeros((D,ac.shape[1]))
    frb, frc = np.zeros_like(fhb), np.zeros_like(fhc)
    sse_b = sse_c = 0.0
    for start in range(0,n,4096):
        stop = start + 4096
        xb, xc = hb[start:stop], hc[start:stop]
        f = encode(art, xb, xc)
        rb, rc = f @ wdb + art["b_dec_base"], f @ wdc + art["b_dec_chat"]
        if not (np.isfinite(rb).all() and np.isfinite(rc).all()):
            raise RunFailure("non-finite reconstruction")
        sf += f.sum(axis=0)
        sq_f += (f*f).sum(axis=0)
        fires += (f>0).sum(axis=0)
        fhb += f.T @ xb
        fhc += f.T @ xc
        fab += f.T @ ab[start:stop]
        fac += f.T @ ac[start:stop]
        frb += f.T @ rb
        frc += f.T @ rc
        sse_b += float(((xb-rb)**2).sum())
        sse_c += float(((xc-rc)**2).sum())
    if not (math.isfinite(sse_b) and math.isfinite(sse_c)):
        raise RunFailure("overflow in reconstruction error")
    l0 = float(fires.sum()/n)
    mean_f = sf/n
    sst_b = float((hb*hb).sum() - n*(hb.mean(axis=0)**2).sum())
    sst_c = float((hc*hc).sum() - n*(hc.mean(axis=0)**2).sum())
    fve = 1.0 - (sse_b+sse_c)/(sst_b+sst_c)

    nb = np.linalg.norm(wdb, axis=1)
    nc = np.linalg.norm(wdc, axis=1)
    mx = np.maximum(nb, nc)
    delta = np.where(mx > 0, 0.5 * (1.0 + (nc - nb) / np.where(mx > 0, mx, 1.0)), 0.5)
    alive = fires >= max(1, math.ceil(MIN_FIRING_RATE * n))
    mass = mean_f * nc  # mean magnitude each latent contributes to the chat stream

    directions = np.asarray(world["directions"], dtype=np.float64)
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    unit_chat = wdc / np.where(nc > 0, nc, 1.0)[:, None]
    cos = unit_chat @ directions.T
    concept = cos.argmax(axis=1)
    best_cos = cos[np.arange(cos.shape[0]), concept]
    aligned = (best_cos >= ALIGN_COS) & (nc > 0)

    # Ground-truth presence ratio of the matched concept, weighted by the
    # latent's own activations: sum_i f_ij a^base_c(i) / sum_i f_ij a^chat_c(i).
    idx = np.arange(D)
    fb = fab[idx, concept]
    fc = fac[idx, concept]
    rho = np.where(fc > 0, fb / np.where(fc > 0, fc, 1.0), np.inf)
    genuine = alive & aligned & (rho < SPECIFICITY_PI)

    # Fixed specificity operating point from raw-activation Latent Scaling
    # (Eq. 4), independent of submitted decoder magnitudes. A top-N boundary
    # made distinct-target recovery depend on near-tied rare factors. Require
    # adequate support and measured presence below the evaluator's fixed pi.
    num_base = (fhb * unit_chat).sum(axis=1)
    num_chat = (fhc * unit_chat).sum(axis=1)
    presence = np.divide(np.maximum(num_base, 0), num_chat,
                         out=np.full_like(num_chat, np.inf), where=num_chat > 1e-12)
    measured_mass = mean_f * np.divide(np.maximum(num_chat, 0), sq_f,
                    out=np.zeros_like(sq_f), where=sq_f > 0)
    family = np.asarray(world["family"])
    target_ids = {FAMILIES.index(name) for name in TARGET_FAMILIES}
    targets = {int(c) for c in np.flatnonzero(np.isin(family, list(target_ids)))}
    candidates = np.flatnonzero(alive & (nc > 0) & np.isfinite(presence)
                                & (presence < SPECIFICITY_PI))
    # Credit the largest contribution once per target. Sorting by an almost
    # zero presence estimate used to credit arbitrary low-mass duplicates.
    top = candidates[np.argsort(-measured_mass[candidates], kind="stable")]
    # Only one factor can claim a given concept. All duplicate contribution
    # remains in the false-discovery denominator and receives no extra credit.
    credited = np.zeros(D, dtype=bool)
    seen = set()
    for j in top:
        c = int(concept[j])
        if genuine[j] and c in targets and c not in seen:
            credited[j] = True
            seen.add(c)
    selected_mass = float(measured_mass[top].sum())
    false_rate = 1.0 if selected_mass <= 0 else float(measured_mass[top[~credited[top]]].sum()) / selected_mass
    covered = seen
    recall = len(covered) / max(len(targets), 1)
    # The paper's decoder-norm classification remains a mechanism diagnostic.
    ft_only = alive & (delta >= FT_ONLY_DELTA)
    ft_mass = float(mass[ft_only].sum())
    norm_false = 1.0 if ft_mass <= 0 else float(mass[ft_only & ~genuine].sum()) / ft_mass

    metrics = {
        "false_ft_only_rate": float(false_rate),
        "chat_specific_recall": float(recall),
        "reconstruction_fve": float(min(max(fve, 0.0), 1.0)),
    }
    if not all(math.isfinite(v) for v in metrics.values()):
        raise RunFailure(f"non-finite metric {metrics}")

    # ---- diagnostics (not scored) ----
    sq = sq_f
    denom = sq * nc ** 2
    num_b_err = ((fhb-frb) * wdc).sum(axis=1) + sq * (wdb * wdc).sum(axis=1)
    num_c_err = ((fhc-frc) * wdc).sum(axis=1) + sq * nc ** 2
    num_b_rec = (frb * wdc).sum(axis=1)
    num_c_rec = (frc * wdc).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        nu_err = (num_b_err / denom) / (num_c_err / denom)
        nu_rec = (num_b_rec / denom) / (num_c_rec / denom)
    ls_false = (~np.isfinite(nu_err)) | (~np.isfinite(nu_rec)) | (nu_err > LS_FALSE_NU) | (nu_rec > LS_FALSE_NU)
    n_ft = int(ft_only.sum())
    spurious = {}
    for j in np.flatnonzero(ft_only & ~genuine):
        key = FAMILIES[int(family[concept[j]])] if aligned[j] else "unaligned"
        spurious[key] = spurious.get(key, 0.0) + float(mass[j])
    recall_by_family = {}
    for name in TARGET_FAMILIES:
        fam = {int(c) for c in np.flatnonzero(family == FAMILIES.index(name))}
        recall_by_family[name] = len(covered & fam) / max(len(fam), 1)
    dec_ids = set(int(c) for c in np.flatnonzero(family == FAMILIES.index("decoupled")))
    shared_factor = alive & aligned & (rho >= SPECIFICITY_PI) & (rho <= 1 / SPECIFICITY_PI) & (nb >= .2 * nc) & (nc >= .2 * nb)
    shared_dec = set(int(concept[j]) for j in np.flatnonzero(shared_factor)) & dec_ids
    twin_dec = set(int(concept[j]) for j in np.flatnonzero(ft_only & ~genuine & aligned)) & dec_ids
    diag = {
        "status": "ok",
        "n_decoupled_with_shared_factor": len(shared_dec),
        "n_decoupled_with_spurious_chat_twin": len(twin_dec),
        "n_latents": D,
        "n_alive": int(alive.sum()),
        "mean_l0": l0,
        "base_fve": 1.0 - sse_b / sst_b,
        "chat_fve": 1.0 - sse_c / sst_c,
        "n_ft_only": n_ft,
        "norm_false_ft_only_rate": norm_false,
        "selection": "raw-activation Latent Scaling below fixed presence threshold",
        "minimum_firing_rate": MIN_FIRING_RATE,
        "presence_threshold": SPECIFICITY_PI,
        "n_selected": len(top),
        "n_ft_only_genuine": int((ft_only & genuine).sum()),
        "ft_only_mass": ft_mass,
        "spurious_mass_fraction_by_family": {k: v / ft_mass for k, v in sorted(spurious.items())} if ft_mass > 0 else {},
        "recall_by_family": recall_by_family,
        "n_genuine_selected": int(sum(bool(genuine[j]) for j in top)),
        "latent_scaling_false_ft_only_fraction": (float(ls_false[ft_only].mean()) if n_ft else None),
    }
    # Keep evaluator-computed mechanism and raw performance diagnostics even
    # when task constraints fail; ranked metrics are still the complete WORST.
    diag["raw_metrics"] = dict(metrics, reconstruction_fve=float(fve))
    if not math.isfinite(l0) or l0 > L0_MAX:
        raise RunFailure(f"mean L0 {l0:.3f} exceeds the sparsity budget {L0_MAX}", diag)
    if not math.isfinite(fve) or fve < FVE_MIN or fve > 1 + 1e-9:
        raise RunFailure(f"FVE {fve} outside [{FVE_MIN}, 1]", diag)
    if min(1 - sse_b / sst_b, 1 - sse_c / sst_c) < FVE_MIN:
        raise RunFailure("a stream fails the reconstruction constraint", diag)
    if recall < RECALL_MIN:
        raise RunFailure(f"specific recovery {recall} below {RECALL_MIN}", diag)

    return metrics, diag


def load_world(experiment: str) -> dict:
    with np.load(WORLD_DIR / f"{experiment}.npz", allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def eval_sample(experiment: str, world: dict, replicate: int = 0) -> dict:
    rng = np.random.default_rng([WORLD_SEEDS[experiment], 2] if replicate == 0 else [WORLD_SEEDS[experiment], 2, replicate])
    base, chat, base_coeff, chat_coeff = sample_heldout(world, N_EVAL, rng)
    return {"base": base, "chat": chat, "base_coeff": base_coeff, "chat_coeff": chat_coeff}


# Frozen reference and baseline artifacts. On the hidden test these names always
# refer to the copies kept here, whatever a workspace holds under the same name.
FROZEN = Path(__file__).resolve().parent / "reference_methods"


def discover_methods(root: Path) -> list[str]:
    names: set[str] = set()
    for base in (root, FROZEN):
        if base.is_dir():
            names |= {p.name for p in base.iterdir() if p.is_dir() and not p.name.startswith(".")}
    return sorted(names)


def artifact_path(root: Path, method: str, experiment: str) -> Path:
    base = FROZEN if (FROZEN / method).is_dir() else root
    return base / method / experiment / WORLD_KEY / ARTIFACT


def evaluate(root: Path) -> tuple[dict, dict]:
    methods = discover_methods(root)
    scores: dict = {"slice": SLICE, "experiments": {}}
    diagnostics: dict = {"slice": SLICE, "experiments": {}}
    for experiment in EXPERIMENTS:
        world = load_world(experiment)
        sample = eval_sample(experiment, world)
        d = sample["base"].shape[1]
        results, diags = {}, {}
        for method in methods:
            path = artifact_path(root, method, experiment)
            try:
                art = load_artifact(path, d)
                metrics, diag = score_artifact(art, sample, world)
            except RunFailure as exc:
                metrics, diag = dict(WORST), {**exc.diagnostics, "status": "failed", "reason": str(exc)}
            except Exception as exc:  # never let one submission abort the run
                metrics, diag = dict(WORST), {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
            results[method] = metrics
            diags[method] = diag
            print(f"[{SLICE}] {experiment} {method}: {metrics} ({diag.get('status')})", flush=True)
        scores["experiments"][experiment] = {"results": results}
        diagnostics["experiments"][experiment] = diags
    for method in methods:
        failures = {e: diagnostics["experiments"][e][method] for e in EXPERIMENTS
                    if diagnostics["experiments"][e][method].get("status") != "ok"}
        if failures:
            for experiment in EXPERIMENTS:
                scores["experiments"][experiment]["results"][method] = dict(WORST)
                diagnostics["experiments"][experiment][method]["suite_status"] = "failed"
                diagnostics["experiments"][experiment][method]["failed_experiments"] = sorted(failures)
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
    parser.add_argument("--out", type=Path, default=Path("scoring/scores_test.json"))
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
