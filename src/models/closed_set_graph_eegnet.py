"""Closed-set EEGNet-style graph rescue model.

This module is intentionally separate from the original EDLSGraph stack.  It is
for the staged CE sanity rescue only: four known BCI IV-2a classes, ordinary
logits by default, and an optional closed-set EDL readout that is not used unless
requested by the runner.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ClosedSetGraphEEGNetConfig:
    n_channels: int = 22
    n_times: int = 1126
    n_classes: int = 4
    f1: int = 8
    depth: int = 2
    token_dim: int = 32
    dropout: float = 0.5
    graph_layers: int = 2
    use_graph: bool = True
    use_edl_head: bool = False
    edl_prior: float = 1.0


class ChannelGraphMixingBlock(nn.Module):
    """Small learned channel graph over per-channel EEGNet tokens."""

    def __init__(self, n_channels: int, token_dim: int, dropout: float = 0.25):
        super().__init__()
        self.adjacency_logits = nn.Parameter(torch.zeros(n_channels, n_channels))
        self.self_proj = nn.Linear(token_dim, token_dim, bias=False)
        self.neighbor_proj = nn.Linear(token_dim, token_dim, bias=False)
        self.norm = nn.LayerNorm(token_dim)
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim * 2, token_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Expected channel tokens [B, C, D], got {tuple(tokens.shape)}")
        adjacency = torch.softmax(self.adjacency_logits, dim=-1)
        mixed = torch.einsum("cd,bdh->bch", adjacency, tokens)
        h = self.norm(tokens + self.dropout(self.self_proj(tokens) + self.neighbor_proj(mixed)))
        return self.norm(h + self.dropout(self.ffn(h)))


class ClosedSetGraphEEGNet(nn.Module):
    """EEGNet-style temporal frontend followed by optional channel-graph mixing."""

    def __init__(self, config: ClosedSetGraphEEGNetConfig | None = None, **kwargs):
        super().__init__()
        cfg = config or ClosedSetGraphEEGNetConfig(**kwargs)
        if kwargs and config is not None:
            raise ValueError("Pass either config or keyword overrides, not both.")
        self.config = cfg
        f2 = cfg.f1 * cfg.depth
        kernel = min(64, max(16, cfg.n_times // 8))

        self.temporal_frontend = nn.Sequential(
            nn.Conv2d(1, cfg.f1, kernel_size=(1, kernel), padding=(0, kernel // 2), bias=False),
            nn.BatchNorm2d(cfg.f1),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(cfg.dropout),
            nn.Conv2d(cfg.f1, cfg.f1, kernel_size=(1, 16), padding=(0, 8), groups=cfg.f1, bias=False),
            nn.Conv2d(cfg.f1, f2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(cfg.dropout),
        )
        self.channel_projector = nn.Sequential(
            nn.Linear(f2, cfg.token_dim),
            nn.LayerNorm(cfg.token_dim),
            nn.ELU(),
        )
        self.graph_blocks = nn.ModuleList(
            [ChannelGraphMixingBlock(cfg.n_channels, cfg.token_dim, dropout=min(0.35, cfg.dropout)) for _ in range(cfg.graph_layers)]
        )
        self.readout = nn.Sequential(
            nn.LayerNorm(cfg.token_dim),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.token_dim, cfg.n_classes),
        )
        self.edl_readout = nn.Linear(cfg.token_dim, cfg.n_classes) if cfg.use_edl_head else None
        self.register_buffer("edl_prior", torch.tensor(float(cfg.edl_prior), dtype=torch.float32))

    def channel_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4:
            raise ValueError(f"Expected EEG input [B, C, T] or [B, 1, C, T], got {tuple(x.shape)}")
        h = self.temporal_frontend(x)
        # [B, F, C, T'] -> per-channel learned token [B, C, F].
        h = h.mean(dim=-1).transpose(1, 2).contiguous()
        return self.channel_projector(h)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.channel_tokens(x)
        if self.config.use_graph:
            for block in self.graph_blocks:
                tokens = block(tokens)
        return tokens.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        z = self.features(x)
        logits = self.readout(z)
        if self.edl_readout is None:
            return logits
        evidence = F.softplus(self.edl_readout(z))
        alpha = evidence + self.edl_prior.clamp_min(1e-6)
        strength = alpha.sum(dim=1)
        return {
            "logits": logits,
            "alpha": alpha,
            "p_hat": alpha / strength.unsqueeze(1).clamp_min(1e-12),
            "vacuity": alpha.size(1) / strength.clamp_min(1e-12),
            "S": strength,
        }


__all__ = ["ClosedSetGraphEEGNet", "ClosedSetGraphEEGNetConfig", "ChannelGraphMixingBlock"]

