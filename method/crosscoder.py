"""Crosscoder with a shared encoder and model-specific base/chat decoders (paper Sec. 2.1, App. B)."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class Crosscoder(nn.Module):
    """f_j(x) = ReLU(e_j^base h^base + e_j^chat h^chat + b_j);  h~^M = sum_j f_j d_j^M + b^M."""

    def __init__(self, d_model: int, n_latents: int, init_norm: float):
        super().__init__()
        self.d_model = d_model
        self.n_latents = n_latents
        self.W_enc = nn.Parameter(torch.empty(2 * d_model, n_latents))
        self.b_enc = nn.Parameter(torch.zeros(n_latents))
        self.W_dec_base = nn.Parameter(torch.empty(n_latents, d_model))
        self.W_dec_chat = nn.Parameter(torch.empty(n_latents, d_model))
        self.b_dec_base = nn.Parameter(torch.zeros(d_model))
        self.b_dec_chat = nn.Parameter(torch.zeros(d_model))
        with torch.no_grad():
            # App. K: decoder = encoder^T, base and chat paired with the same
            # initial weights, latent directions scaled to ``init_norm``.
            paired = F.normalize(torch.randn(n_latents, d_model), dim=1) * init_norm
            self.W_dec_base.copy_(paired)
            self.W_dec_chat.copy_(paired)
            self.W_enc[:d_model].copy_(paired.T)
            self.W_enc[d_model:].copy_(paired.T)

    def activations(self, base: torch.Tensor, chat: torch.Tensor) -> torch.Tensor:
        d = self.d_model
        return F.relu(base @ self.W_enc[:d] + chat @ self.W_enc[d:] + self.b_enc)

    def decoder_norms(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (torch.linalg.vector_norm(self.W_dec_base, dim=1),
                torch.linalg.vector_norm(self.W_dec_chat, dim=1))

    def selection_norm(self, kind: str = "sum") -> torch.Tensor:
        """Per-latent scale used by BatchTopK selection.

        ``sum``: ||d^base|| + ||d^chat|| (paper, App. B.2);
        ``concat``: ||[d^base, d^chat]|| (App. B.3 variant); ``max``: max of the two.
        """
        nb, nc = self.decoder_norms()
        if kind == "sum":
            return nb + nc
        if kind == "concat":
            return torch.sqrt(nb ** 2 + nc ** 2)
        if kind == "max":
            return torch.maximum(nb, nc)
        raise ValueError(f"unknown selection norm {kind!r}")

    def decode(self, f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return f @ self.W_dec_base + self.b_dec_base, f @ self.W_dec_chat + self.b_dec_chat
