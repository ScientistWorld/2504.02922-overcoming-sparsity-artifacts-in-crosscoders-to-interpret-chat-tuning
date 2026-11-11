"""Train L1 and BatchTopK crosscoders (paper App. B, K) and export evaluator artifacts.

Artifacts follow the benchmark's output contract (see briefing/evaluation.md):
``<out>/<experiment>/<world>/crosscoder.npz`` with arrays ``W_enc`` (2d, D),
``b_enc`` (D,), ``threshold`` (D,), ``W_dec_base`` / ``W_dec_chat`` (D, d) and
``b_dec_base`` / ``b_dec_chat`` (d,). One dictionary is trained per experiment
and per world (``dev`` = public development world, ``heldout`` = the world the
hidden test scores; only its unlabeled training activations are used).

Examples
--------
python3 -m method.train --method batchtopk --out checkpoints/methods/batchtopk
python3 -m method.train --method l1 --out checkpoints/methods/l1
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch

from method.crosscoder import Crosscoder

EXPERIMENTS = ("synthetic_chat_specificity", "synthetic_faint_chat_concepts")
WORLDS = ("dev", "heldout")
DATA_ROOT = Path("data/worlds")


@dataclass
class Config:
    sparsity: str = "batchtopk"          # "batchtopk" or "l1"
    n_latents: int = 256
    k: int = 8                           # BatchTopK active latents per input
    k_start: int = 0                     # >0: anneal k from k_start to k (paper App. I.1 recipe)
    k_anneal_frac: float = 0.25
    mu: float = 0.0                      # L1 sparsity weight (calibrated to the target L0)
    steps: int = 16_000
    batch_size: int = 256
    lr: float = 1e-3
    lr_decay_frac: float = 0.2           # linear decay to 0 over the final fraction of steps
    init_norm: float = 0.0               # 0 -> paper default (1.0 BatchTopK, 0.05 L1)
    aux_alpha: float = 1.0 / 32.0        # BatchTopK auxiliary dead-latent loss weight (Eq. 12)
    aux_k: int = 32                      # dead latents used by the auxiliary loss per input
    dead_samples: int = 200_000          # a latent is dead after this many inputs without firing
    selection_norm: str = "sum"          # BatchTopK scaling: sum (paper) / concat / max (App. B.3)
    target_l0: float = 8.0               # L1 calibration target (paper: match L0 across crosscoders)
    max_l0: float = 8.4


def load_pair(path: Path, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    with np.load(path, allow_pickle=False) as z:
        base = torch.from_numpy(z["base"].astype(np.float32)).to(device)
        chat = torch.from_numpy(z["chat"].astype(np.float32)).to(device)
    return base, chat


def _lr_at(cfg: Config, step: int) -> float:
    decay_start = int(cfg.steps * (1.0 - cfg.lr_decay_frac))
    if step < decay_start or cfg.steps <= decay_start:
        return cfg.lr
    return cfg.lr * (cfg.steps - step) / (cfg.steps - decay_start)


def _k_at(cfg: Config, step: int) -> int:
    if cfg.k_start <= 0 or cfg.k_start <= cfg.k:
        return cfg.k
    span = max(1, int(cfg.steps * cfg.k_anneal_frac))
    frac = min(1.0, step / span)
    return int(round(cfg.k_start + (cfg.k - cfg.k_start) * frac))


def _recon_loss(xb, xc, rb, rc):
    return 0.5 * ((xb - rb) ** 2).sum(dim=1).mean() + 0.5 * ((xc - rc) ** 2).sum(dim=1).mean()


def train_crosscoder(cfg: Config, hb: torch.Tensor, hc: torch.Tensor, seed: int) -> tuple[Crosscoder, dict]:
    device = hb.device
    torch.manual_seed(seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    n, d = hb.shape
    btk = cfg.sparsity == "batchtopk"
    if cfg.sparsity not in ("batchtopk", "l1"):
        raise ValueError(cfg.sparsity)
    init_norm = cfg.init_norm if cfg.init_norm > 0 else (1.0 if btk else 0.05)
    model = Crosscoder(d, cfg.n_latents, init_norm).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    since_fired = torch.zeros(cfg.n_latents, device=device)
    bsz = cfg.batch_size
    perm = torch.randperm(n, generator=gen, device=device)
    ptr = 0
    t0 = time.time()
    last = {}
    for step in range(cfg.steps):
        for group in opt.param_groups:
            group["lr"] = _lr_at(cfg, step)
        if ptr + bsz > n:
            perm = torch.randperm(n, generator=gen, device=device)
            ptr = 0
        idx = perm[ptr:ptr + bsz]
        ptr += bsz
        xb, xc = hb[idx], hc[idx]
        pre = model.activations(xb, xc)
        aux = torch.zeros((), device=device)
        if btk:
            k_cur = _k_at(cfg, step)
            with torch.no_grad():
                v = pre * model.selection_norm(cfg.selection_norm)
                keep = min(bsz * k_cur, v.numel())
                theta_b = torch.topk(v.flatten(), keep, sorted=False).values.min()
                mask = (v >= theta_b) & (pre > 0)
            f = pre * mask
            rb, rc = model.decode(f)
            recon = _recon_loss(xb, xc, rb, rc)
            loss = recon
            if cfg.aux_alpha > 0:
                dead = since_fired >= cfg.dead_samples
                n_dead = int(dead.sum())
                if n_dead > 0:
                    k_aux = min(cfg.aux_k, n_dead)
                    dead_acts = pre.masked_fill(~dead[None, :], float("-inf"))
                    vals, ind = dead_acts.topk(k_aux, dim=1)
                    buf = torch.zeros_like(pre).scatter(1, ind, vals)
                    eb = (xb - rb).detach()
                    ec = (xc - rc).detach()
                    l2 = ((eb - buf @ model.W_dec_base) ** 2).sum(1).mean() + \
                         ((ec - buf @ model.W_dec_chat) ** 2).sum(1).mean()
                    denom = ((eb - eb.mean(0)) ** 2).sum(1).mean() + ((ec - ec.mean(0)) ** 2).sum(1).mean()
                    aux = torch.nan_to_num(l2 / denom, nan=0.0, posinf=0.0)
                    loss = loss + cfg.aux_alpha * aux
        else:
            f = pre
            rb, rc = model.decode(f)
            recon = _recon_loss(xb, xc, rb, rc)
            nb, nc = model.decoder_norms()
            loss = recon + cfg.mu * (f * (nb + nc)).sum(dim=1).mean()
        with torch.no_grad():
            since_fired += bsz
            since_fired[(f > 0).any(dim=0)] = 0
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == cfg.steps - 1:
            last = {"final_recon_loss": float(recon.detach()), "final_aux_loss": float(aux.detach()),
                    "dead_fraction": float((since_fired >= cfg.dead_samples).float().mean())}
    info = {"train_seconds": time.time() - t0, **last}
    return model, info


@torch.no_grad()
def estimate_theta(model: Crosscoder, cfg: Config, hb: torch.Tensor, hc: torch.Tensor) -> float:
    """Eq. 14: theta = E_batch[min selected scaled activation] over training batches."""
    sel = model.selection_norm(cfg.selection_norm)
    mins = []
    for start in range(0, hb.shape[0] - cfg.batch_size + 1, cfg.batch_size):
        v = model.activations(hb[start:start + cfg.batch_size], hc[start:start + cfg.batch_size]) * sel
        mins.append(torch.topk(v.flatten(), cfg.batch_size * cfg.k, sorted=False).values.min())
    return float(torch.stack(mins).mean())


@torch.no_grad()
def latent_thresholds(model: Crosscoder, cfg: Config, theta: float | None) -> np.ndarray:
    """Per-latent threshold on f_j: Eq. 13 keeps f_j iff f_j * scale_j > theta."""
    if cfg.sparsity != "batchtopk" or theta is None:
        return np.zeros(model.n_latents, dtype=np.float64)
    scale = model.selection_norm(cfg.selection_norm).double().cpu().numpy()
    thr = theta / np.maximum(scale, 1e-12)
    return np.minimum(thr, 1e30)


@torch.no_grad()
def infer_features(model: Crosscoder, threshold: np.ndarray, hb: torch.Tensor, hc: torch.Tensor) -> torch.Tensor:
    pre = model.activations(hb, hc)
    thr = torch.as_tensor(threshold, dtype=pre.dtype, device=pre.device)
    return torch.where((pre > 0) & (pre > thr), pre, torch.zeros_like(pre))


@torch.no_grad()
def mean_l0(model: Crosscoder, threshold: np.ndarray, hb: torch.Tensor, hc: torch.Tensor) -> float:
    return float((infer_features(model, threshold, hb, hc) > 0).float().sum(dim=1).mean())


def export_artifact(model: Crosscoder, threshold: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        W_enc=model.W_enc.detach().cpu().numpy().astype(np.float32),
        b_enc=model.b_enc.detach().cpu().numpy().astype(np.float32),
        threshold=np.asarray(threshold, dtype=np.float64),
        W_dec_base=model.W_dec_base.detach().cpu().numpy().astype(np.float32),
        W_dec_chat=model.W_dec_chat.detach().cpu().numpy().astype(np.float32),
        b_dec_base=model.b_dec_base.detach().cpu().numpy().astype(np.float32),
        b_dec_chat=model.b_dec_chat.detach().cpu().numpy().astype(np.float32),
    )


def _fit_and_export(cfg: Config, hb, hc, seed: int, out: Path) -> dict:
    model, info = train_crosscoder(cfg, hb, hc, seed)
    theta = estimate_theta(model, cfg, hb, hc) if cfg.sparsity == "batchtopk" else None
    thr = latent_thresholds(model, cfg, theta)
    info.update({"theta": theta, "train_l0": mean_l0(model, thr, hb, hc)})
    export_artifact(model, thr, out / "crosscoder.npz")
    return info


def calibrate_l1(cfg: Config, hb, hc, seed: int, out: Path) -> dict:
    """Pick the L1 weight mu whose trained crosscoder reaches the target L0 (paper: L0 matched)."""
    probe = slice(0, min(8192, hb.shape[0]))
    zero = np.zeros(cfg.n_latents)
    runs: dict[float, tuple[float, Crosscoder, dict]] = {}

    def run(mu: float) -> float:
        c = replace(cfg, mu=mu)
        model, info = train_crosscoder(c, hb, hc, seed)
        l0 = mean_l0(model, zero, hb[probe], hc[probe])
        runs[mu] = (l0, model, info)
        print(f"[l1] {out}: mu={mu:.4g} L0={l0:.3f}", flush=True)
        return l0

    for mu in (0.15, 0.3, 0.6, 1.2, 2.4):
        run(mu)
    for _ in range(4):  # extend the grid until the target is bracketed
        if any(l0 <= cfg.target_l0 for l0, _, _ in runs.values()):
            break
        run(max(runs) * 2.0)
    for _ in range(4):
        if any(l0 > cfg.target_l0 for l0, _, _ in runs.values()):
            break
        run(min(runs) / 2.0)
    for _ in range(3):  # bisection in log(mu)
        above = [mu for mu, (l0, _, _) in runs.items() if l0 > cfg.target_l0]
        below = [mu for mu, (l0, _, _) in runs.items() if l0 <= cfg.target_l0]
        if not above or not below:
            break
        run(math.sqrt(max(above) * min(below)))
    feasible = {mu: r for mu, r in runs.items() if r[0] <= cfg.max_l0}
    pool = feasible if feasible else runs
    mu_best = min(pool, key=lambda m: abs(pool[m][0] - cfg.target_l0))
    l0, model, info = runs[mu_best]
    thr = latent_thresholds(model, replace(cfg, mu=mu_best), None)
    export_artifact(model, thr, out / "crosscoder.npz")
    info.update({"mu": mu_best, "train_l0": l0, "calibration": {f"{m:.6g}": r[0] for m, r in sorted(runs.items())}})
    return info


def run_task(task: dict) -> dict:
    """Worker entry point. ``task`` keys: kind, cfg, data, out, seed, device, [threads]."""
    threads = int(task.get("threads", 0))
    if threads > 0:
        torch.set_num_threads(threads)
    cfg = Config(**task["cfg"])
    out = Path(task["out"])
    done = out / "train_info.json"
    if task.get("resume", True) and done.exists() and (out / "crosscoder.npz").exists():
        record = json.loads(done.read_text())
        if record.get("config") == asdict(cfg) and record.get("seed") == task["seed"]:
            print(f"[train] reuse {out}", flush=True)
            return record
    out.mkdir(parents=True, exist_ok=True)
    hb, hc = load_pair(Path(task["data"]), task["device"])
    if task["kind"] == "l1_calibrate":
        info = calibrate_l1(cfg, hb, hc, task["seed"], out)
    else:
        info = _fit_and_export(cfg, hb, hc, task["seed"], out)
    record = {"config": asdict(cfg), "seed": task["seed"], "data": task["data"], **info}
    (out / "train_info.json").write_text(json.dumps(record, indent=1) + "\n")
    print(f"[train] done {out} ({info.get('train_seconds', 0):.0f}s, L0={info.get('train_l0')})", flush=True)
    return record


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def run_tasks(tasks: list[dict], workers: int, raise_errors: bool = True) -> list[dict]:
    """Run tasks (in spawned worker processes when workers > 1)."""
    results: list[dict] = []
    if workers <= 1:
        for t in tasks:
            try:
                results.append(run_task(t))
            except Exception as exc:
                if raise_errors:
                    raise
                print(f"[train] FAILED {t['out']}: {type(exc).__name__}: {exc}", flush=True)
        return results
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futures = [(t, pool.submit(run_task, t)) for t in tasks]
        for t, fut in futures:
            try:
                results.append(fut.result())
            except Exception as exc:
                if raise_errors:
                    raise
                print(f"[train] FAILED {t['out']}: {type(exc).__name__}: {exc}", flush=True)
    return results


def suite_tasks(method: str, out_root: Path, seed: int, cfg: Config, device: str, threads: int) -> list[dict]:
    tasks = []
    for experiment in EXPERIMENTS:
        for world in WORLDS:
            data = DATA_ROOT / experiment / world / "train.npz"
            if not data.exists():
                raise FileNotFoundError(f"{data} missing; run scripts/download.sh")
            tasks.append({
                "kind": "l1_calibrate" if method == "l1" else "train",
                "cfg": asdict(cfg), "data": str(data), "out": str(out_root / experiment / world),
                "seed": seed, "device": device, "threads": threads,
            })
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("batchtopk", "l1"), default="batchtopk")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    args = parser.parse_args()
    cfg = Config(sparsity=args.method)
    if args.steps is not None:
        cfg = replace(cfg, steps=args.steps)
    if args.k is not None:
        cfg = replace(cfg, k=args.k)
    device = default_device() if args.device == "auto" else args.device
    threads = max(1, (os.cpu_count() or 1) // max(1, args.workers))
    tasks = suite_tasks(args.method, args.out, args.seed, cfg, device, threads)
    records = run_tasks(tasks, args.workers)
    summary = {Path(r["data"]).parent.parent.name + "/" + Path(r["data"]).parent.name: r for r in records}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
