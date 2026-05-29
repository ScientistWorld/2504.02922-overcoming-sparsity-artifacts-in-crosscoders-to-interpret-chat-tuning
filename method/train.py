"""Train L1 and BatchTopK crosscoders on paired activation data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from method.crosscoder import Crosscoder


def _loss_l1(model: Crosscoder, base: torch.Tensor, chat: torch.Tensor, mu: float):
    out = model(base, chat, mode="l1")
    recon = 0.5 * F.mse_loss(out.base_recon, base) + 0.5 * F.mse_loss(out.chat_recon, chat)
    _, _, total_norm = model.decoder_norms()
    sparsity = (out.features * total_norm.unsqueeze(0)).sum(dim=1).mean()
    return recon + mu * sparsity, out, recon, sparsity


def _loss_batchtopk(model: Crosscoder, base: torch.Tensor, chat: torch.Tensor, k: int):
    out = model(base, chat, mode="batchtopk_train", k=k)
    recon = 0.5 * F.mse_loss(out.base_recon, base) + 0.5 * F.mse_loss(out.chat_recon, chat)
    return recon, out, recon, torch.tensor(0.0, device=base.device)


def train_one(
    name: str,
    train_base: np.ndarray,
    train_chat: np.ndarray,
    val_base: np.ndarray,
    val_chat: np.ndarray,
    out_dir: Path,
    n_latents: int,
    epochs: int,
    batch_size: int,
    lr: float,
    mu: float,
    k: int,
    seed: int,
    device: str,
) -> dict[str, float]:
    torch.manual_seed(seed)
    mode = "l1" if name == "l1" else "batchtopk"
    init_norm = 0.05 if mode == "l1" else 1.0
    model = Crosscoder(train_base.shape[1], n_latents, init_decoder_norm=init_norm).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    ds = TensorDataset(torch.from_numpy(train_base), torch.from_numpy(train_chat))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    for _ in range(epochs):
        model.train()
        for base, chat in loader:
            base = base.to(device)
            chat = chat.to(device)
            opt.zero_grad(set_to_none=True)
            if mode == "l1":
                loss, _, _, _ = _loss_l1(model, base, chat, mu)
            else:
                loss, _, _, _ = _loss_batchtopk(model, base, chat, k)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    with torch.no_grad():
        vb = torch.from_numpy(val_base).to(device)
        vc = torch.from_numpy(val_chat).to(device)
        if mode == "l1":
            out = model(vb, vc, mode="l1")
        else:
            out = model(vb, vc, mode="batchtopk_infer", k=k)
        mse_base = F.mse_loss(out.base_recon, vb).item()
        mse_chat = F.mse_loss(out.chat_recon, vc).item()
        l0 = (out.features > 0).float().sum(dim=1).mean().item()

    path = out_dir / name
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "d_model": train_base.shape[1],
            "n_latents": n_latents,
            "mode": mode,
            "k": k,
            "mu": mu,
        },
        path / "model.pt",
    )
    np.savez_compressed(
        path / "val_outputs.npz",
        features=out.features.cpu().numpy(),
        base_recon=out.base_recon.cpu().numpy(),
        chat_recon=out.chat_recon.cpu().numpy(),
        base_decoder=model.base_decoder.detach().cpu().numpy(),
        chat_decoder=model.chat_decoder.detach().cpu().numpy(),
        base_bias=model.base_bias.detach().cpu().numpy(),
        chat_bias=model.chat_bias.detach().cpu().numpy(),
    )
    metrics = {"val_mse_base": mse_base, "val_mse_chat": mse_chat, "val_l0": l0}
    (path / "train_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/synthetic_activations.npz"))
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/crosscoder_synthetic"))
    parser.add_argument("--n-latents", type=int, default=88)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mu", type=float, default=2e-3)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("l1", "batchtopk"),
        default=("l1", "batchtopk"),
        help="Which crosscoder variants to train.",
    )
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    data = np.load(args.data)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in args.methods:
        summary[name] = train_one(
            name,
            data["train_base"],
            data["train_chat"],
            data["val_base"],
            data["val_chat"],
            args.out_dir,
            args.n_latents,
            args.epochs,
            args.batch_size,
            args.lr,
            args.mu,
            args.k,
            args.seed + (0 if name == "l1" else 1),
            device,
        )
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
