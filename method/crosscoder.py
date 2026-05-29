"""Crosscoder modules matching the paper's shared-encoder/two-decoder setup."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ForwardOutput:
    features: torch.Tensor
    base_recon: torch.Tensor
    chat_recon: torch.Tensor


class Crosscoder(nn.Module):
    """Shared sparse encoder with model-specific base/chat decoders."""

    def __init__(self, d_model: int, n_latents: int, init_decoder_norm: float = 0.2):
        super().__init__()
        self.encoder = nn.Linear(2 * d_model, n_latents)
        self.base_decoder = nn.Parameter(torch.empty(n_latents, d_model))
        self.chat_decoder = nn.Parameter(torch.empty(n_latents, d_model))
        self.base_bias = nn.Parameter(torch.zeros(d_model))
        self.chat_bias = nn.Parameter(torch.zeros(d_model))
        self.reset_parameters(init_decoder_norm)

    def reset_parameters(self, init_decoder_norm: float) -> None:
        nn.init.kaiming_uniform_(self.encoder.weight, a=5**0.5)
        nn.init.zeros_(self.encoder.bias)
        with torch.no_grad():
            nn.init.normal_(self.base_decoder)
            nn.init.normal_(self.chat_decoder)
            self.base_decoder[:] = F.normalize(self.base_decoder, dim=1) * init_decoder_norm
            self.chat_decoder[:] = F.normalize(self.chat_decoder, dim=1) * init_decoder_norm

    def raw_features(self, base: torch.Tensor, chat: torch.Tensor) -> torch.Tensor:
        return F.relu(self.encoder(torch.cat([base, chat], dim=-1)))

    def decoder_norms(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base_norm = torch.linalg.vector_norm(self.base_decoder, dim=1)
        chat_norm = torch.linalg.vector_norm(self.chat_decoder, dim=1)
        return base_norm, chat_norm, base_norm + chat_norm

    def batch_topk_features(
        self, base: torch.Tensor, chat: torch.Tensor, k: int
    ) -> torch.Tensor:
        features = self.raw_features(base, chat)
        _, _, total_norm = self.decoder_norms()
        scaled = features * total_norm.unsqueeze(0)
        keep = min(features.shape[0] * k, scaled.numel())
        if keep <= 0:
            return torch.zeros_like(features)
        threshold = torch.topk(scaled.flatten(), keep, sorted=False).values.min()
        return features * (scaled >= threshold).to(features.dtype)

    def infer_topk_features(
        self, base: torch.Tensor, chat: torch.Tensor, k: int
    ) -> torch.Tensor:
        features = self.raw_features(base, chat)
        _, _, total_norm = self.decoder_norms()
        scaled = features * total_norm.unsqueeze(0)
        keep = min(k, features.shape[1])
        values, indices = torch.topk(scaled, keep, dim=1)
        mask = torch.zeros_like(features)
        mask.scatter_(1, indices, (values > 0).to(features.dtype))
        return features * mask

    def decode(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        base_recon = features @ self.base_decoder + self.base_bias
        chat_recon = features @ self.chat_decoder + self.chat_bias
        return base_recon, chat_recon

    def forward(
        self, base: torch.Tensor, chat: torch.Tensor, mode: str, k: int | None = None
    ) -> ForwardOutput:
        if mode == "l1":
            features = self.raw_features(base, chat)
        elif mode == "batchtopk_train":
            if k is None:
                raise ValueError("k is required for BatchTopK training")
            features = self.batch_topk_features(base, chat, k)
        elif mode == "batchtopk_infer":
            if k is None:
                raise ValueError("k is required for BatchTopK inference")
            features = self.infer_topk_features(base, chat, k)
        else:
            raise ValueError(f"unknown mode: {mode}")
        base_recon, chat_recon = self.decode(features)
        return ForwardOutput(features, base_recon, chat_recon)
