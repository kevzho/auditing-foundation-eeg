"""Closed-set competence models for leakage-safe MI graph expansion.

These models are deliberately CE-only.  They are meant to answer a narrow
question before any evidential claim: can a small neural decoder learn useful
closed-set BCI IV-2a signal, and does graph mixing help after that sanity check?
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


BCI_IV_2A_CHANNELS = (
    "Fz",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "P1",
    "Pz",
    "P2",
    "POz",
)


# Approximate 2-D 10-20 coordinates for the 22 BCI IV-2a EEG channels.  The
# graph only needs relative distances, so these coarse scalp-plane coordinates
# are sufficient and easy to audit.
BCI_IV_2A_COORDS = torch.tensor(
    [
        [0.0, 2.0],  # Fz
        [-1.5, 1.2],  # FC3
        [-0.7, 1.3],  # FC1
        [0.0, 1.35],  # FCz
        [0.7, 1.3],  # FC2
        [1.5, 1.2],  # FC4
        [-2.2, 0.15],  # C5
        [-1.55, 0.0],  # C3
        [-0.65, 0.0],  # C1
        [0.0, 0.0],  # Cz
        [0.65, 0.0],  # C2
        [1.55, 0.0],  # C4
        [2.2, 0.15],  # C6
        [-1.5, -1.15],  # CP3
        [-0.65, -1.25],  # CP1
        [0.0, -1.3],  # CPz
        [0.65, -1.25],  # CP2
        [1.5, -1.15],  # CP4
        [-0.65, -2.15],  # P1
        [0.0, -2.25],  # Pz
        [0.65, -2.15],  # P2
        [0.0, -3.0],  # POz
    ],
    dtype=torch.float32,
)


@dataclass(frozen=True)
class CompetenceConfig:
    n_channels: int = 22
    n_times: int = 1126
    n_classes: int = 4
    variant: str = "raw_no_graph_compact"
    token_dim: int = 32
    temporal_filters: int = 16
    dropout: float = 0.5
    graph_dropout: float = 0.2
    graph_init_gate: float = 0.05
    residual_adj_scale: float = 0.05


def anatomical_adjacency(n_channels: int = 22, sigma: float = 1.05, k: int = 4) -> torch.Tensor:
    """Return row-stochastic approximate anatomical adjacency."""

    if n_channels != 22:
        return torch.eye(n_channels, dtype=torch.float32)
    coords = BCI_IV_2A_COORDS
    dist = torch.cdist(coords, coords)
    weights = torch.exp(-(dist**2) / (2.0 * sigma**2))
    weights.fill_diagonal_(1.0)
    if k > 0:
        keep = torch.zeros_like(weights, dtype=torch.bool)
        nn_idx = torch.topk(weights, k=min(k + 1, n_channels), dim=1).indices
        keep.scatter_(1, nn_idx, True)
        keep = keep | keep.T
        weights = weights * keep.float()
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return weights


class EEGNetCE(nn.Module):
    """Compact EEGNet-style closed-set baseline with validation TS in runner."""

    def __init__(self, cfg: CompetenceConfig):
        super().__init__()
        f1 = cfg.temporal_filters
        depth = 2
        f2 = f1 * depth
        kernel = min(64, max(16, cfg.n_times // 8))
        self.net = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=(1, kernel), padding=(0, kernel // 2), bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f2, kernel_size=(cfg.n_channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(cfg.dropout),
            nn.Conv2d(f2, f2, kernel_size=(1, 16), padding=(0, 8), groups=f2, bias=False),
            nn.Conv2d(f2, f2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(cfg.dropout),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(f2, cfg.n_classes),
        )

    def regularization_loss(self) -> torch.Tensor:
        return torch.zeros((), device=next(self.parameters()).device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        return self.net(x)


class ShallowConvNetCE(nn.Module):
    """Shallow ConvNet-style closed-set MI classifier for cropped trials."""

    def __init__(self, cfg: CompetenceConfig, filterbank: bool = False):
        super().__init__()
        branches = (32, 64, 128) if filterbank else (64,)
        branch_filters = max(8, cfg.temporal_filters)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, branch_filters, kernel_size=(1, min(k, cfg.n_times)), padding=(0, min(k, cfg.n_times) // 2), bias=False),
                    nn.Conv2d(branch_filters, branch_filters, kernel_size=(cfg.n_channels, 1), bias=False),
                    nn.BatchNorm2d(branch_filters),
                    nn.ELU(),
                    nn.AvgPool2d(kernel_size=(1, 8)),
                    nn.Dropout(cfg.dropout),
                    nn.Conv2d(branch_filters, branch_filters, kernel_size=(1, 16), padding=(0, 8), bias=False),
                    nn.BatchNorm2d(branch_filters),
                    nn.ELU(),
                    nn.AvgPool2d(kernel_size=(1, 8)),
                    nn.Dropout(cfg.dropout),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                )
                for k in branches
            ]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(branch_filters * len(branches), cfg.n_classes),
        )

    def regularization_loss(self) -> torch.Tensor:
        return torch.zeros((), device=next(self.parameters()).device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        parts = [branch(x) for branch in self.branches]
        return self.classifier(torch.cat(parts, dim=1))


class SqueezeExcite2d(nn.Module):
    """Light channel attention for temporal feature maps."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.ELU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.net(x).view(x.shape[0], x.shape[1], 1, 1)
        return x * scale


class SeparableTemporalResidualBlock(nn.Module):
    """Depthwise-separable temporal residual block without graph mixing."""

    def __init__(self, channels: int, kernel: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, kernel), padding=(0, kernel // 2), groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Conv2d(channels, channels, kernel_size=(1, kernel), padding=(0, kernel // 2), groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(channels),
            SqueezeExcite2d(channels),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(x + self.block(x))


class MultiScaleTemporalCE(nn.Module):
    """No-graph temporal-separable CE student for multi-scale cropped trials."""

    def __init__(self, cfg: CompetenceConfig):
        super().__init__()
        branch_filters = max(4, cfg.temporal_filters // 2)
        kernels = (16, 32, 64)
        self.branches = nn.ModuleList()
        for kernel in kernels:
            k = min(kernel, cfg.n_times)
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(1, branch_filters, kernel_size=(1, k), padding=(0, k // 2), bias=False),
                    nn.BatchNorm2d(branch_filters),
                    nn.ELU(),
                    nn.Conv2d(branch_filters, branch_filters * 2, kernel_size=(cfg.n_channels, 1), groups=branch_filters, bias=False),
                    nn.BatchNorm2d(branch_filters * 2),
                    nn.ELU(),
                    nn.AvgPool2d(kernel_size=(1, 8)),
                    nn.Conv2d(branch_filters * 2, branch_filters * 2, kernel_size=(1, 15), padding=(0, 7), groups=branch_filters * 2, bias=False),
                    nn.Conv2d(branch_filters * 2, branch_filters * 2, kernel_size=(1, 1), bias=False),
                    nn.BatchNorm2d(branch_filters * 2),
                    nn.ELU(),
                    nn.AvgPool2d(kernel_size=(1, 4)),
                    nn.Dropout(cfg.dropout),
                )
            )
        out_channels = branch_filters * 2 * len(kernels)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
            SeparableTemporalResidualBlock(out_channels, kernel=15, dropout=cfg.dropout),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(cfg.dropout),
            nn.Linear(out_channels, cfg.n_classes),
        )

    def regularization_loss(self) -> torch.Tensor:
        return torch.zeros((), device=next(self.parameters()).device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        parts = [branch(x) for branch in self.branches]
        min_t = min(part.shape[-1] for part in parts)
        if any(part.shape[-1] != min_t for part in parts):
            parts = [part[..., :min_t] for part in parts]
        return self.fuse(torch.cat(parts, dim=1))


class ChannelTokenFrontend(nn.Module):
    """Temporal frontend preserving channel tokens for graph reasoning."""

    def __init__(self, cfg: CompetenceConfig, filterbank: bool):
        super().__init__()
        branch_filters = max(4, cfg.temporal_filters // (3 if filterbank else 1))
        kernels = (16, 32, 64) if filterbank else (64,)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, branch_filters, kernel_size=(1, k), padding=(0, k // 2), bias=False),
                    nn.BatchNorm2d(branch_filters),
                    nn.ELU(),
                    nn.AvgPool2d(kernel_size=(1, 4)),
                    nn.Dropout(cfg.dropout),
                    nn.Conv2d(branch_filters, branch_filters, kernel_size=(1, 16), padding=(0, 8), groups=branch_filters, bias=False),
                    nn.Conv2d(branch_filters, branch_filters, kernel_size=(1, 1), bias=False),
                    nn.BatchNorm2d(branch_filters),
                    nn.ELU(),
                    nn.AvgPool2d(kernel_size=(1, 4)),
                    nn.Dropout(cfg.dropout),
                )
                for k in kernels
            ]
        )
        out_filters = branch_filters * len(kernels)
        self.project = nn.Sequential(nn.Linear(out_filters, cfg.token_dim), nn.LayerNorm(cfg.token_dim), nn.ELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        parts = [branch(x) for branch in self.branches]
        h = torch.cat(parts, dim=1)
        tokens = h.mean(dim=-1).transpose(1, 2).contiguous()
        return self.project(tokens)


class ResidualGraphBlock(nn.Module):
    """Residual channel graph update initialized to preserve the input."""

    def __init__(self, cfg: CompetenceConfig, mode: str):
        super().__init__()
        self.mode = mode
        self.fixed_graph = mode in {"anatomical_fixed", "anatomical_learned_residual"}
        self.learn_residual = mode in {"identity_learned", "anatomical_learned_residual"}
        base = torch.eye(cfg.n_channels) if mode == "identity_learned" else anatomical_adjacency(cfg.n_channels)
        self.register_buffer("base_adjacency", base)
        if self.learn_residual:
            self.residual_logits = nn.Parameter(torch.zeros(cfg.n_channels, cfg.n_channels))
        else:
            self.residual_logits = None
        initial_gate = torch.logit(torch.tensor(float(cfg.graph_init_gate)).clamp(1e-4, 1 - 1e-4))
        self.gate_logit = nn.Parameter(initial_gate.clone())
        self.mix = nn.Linear(cfg.token_dim, cfg.token_dim, bias=False)
        self.norm = nn.LayerNorm(cfg.token_dim)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.token_dim, cfg.token_dim * 2),
            nn.ELU(),
            nn.Dropout(cfg.graph_dropout),
            nn.Linear(cfg.token_dim * 2, cfg.token_dim),
        )
        self.dropout = nn.Dropout(cfg.graph_dropout)
        self.residual_adj_scale = float(cfg.residual_adj_scale)

    def adjacency(self) -> torch.Tensor:
        adj = self.base_adjacency
        if self.residual_logits is not None:
            residual = torch.tanh(self.residual_logits) * self.residual_adj_scale
            adj = torch.relu(adj + residual)
        adj = adj / adj.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return adj

    def regularization_loss(self) -> torch.Tensor:
        if self.residual_logits is None:
            return torch.zeros((), device=self.gate_logit.device)
        return torch.mean(torch.tanh(self.residual_logits) ** 2)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        adj = self.adjacency()
        mixed = torch.einsum("cd,bdh->bch", adj, tokens)
        gate = torch.sigmoid(self.gate_logit)
        h = tokens + gate * self.dropout(self.mix(mixed))
        h = self.norm(h)
        return self.norm(h + gate * self.dropout(self.ffn(h)))


class TokenGraphCompetenceNet(nn.Module):
    """Channel-token classifier with optional residual graph mixing."""

    def __init__(self, cfg: CompetenceConfig):
        super().__init__()
        self.cfg = cfg
        filterbank = "filterbank" in cfg.variant
        self.frontend = ChannelTokenFrontend(cfg, filterbank=filterbank)
        graph_mode = self._graph_mode(cfg.variant)
        self.graph = ResidualGraphBlock(cfg, graph_mode) if graph_mode is not None else None
        self.readout = nn.Sequential(
            nn.LayerNorm(cfg.token_dim),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.token_dim, cfg.n_classes),
        )

    @staticmethod
    def _graph_mode(variant: str) -> str | None:
        if variant == "raw_graph_residual_identity":
            return "identity_learned"
        if variant in {"raw_graph_anatomical_fixed", "raw_filterbank_graph_anatomical"}:
            return "anatomical_fixed"
        if variant == "raw_graph_anatomical_learned_residual":
            return "anatomical_learned_residual"
        return None

    def regularization_loss(self) -> torch.Tensor:
        if self.graph is None:
            return torch.zeros((), device=next(self.parameters()).device)
        return self.graph.regularization_loss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.frontend(x)
        if self.graph is not None:
            tokens = self.graph(tokens)
        return self.readout(tokens.mean(dim=1))


def build_competence_model(cfg: CompetenceConfig) -> nn.Module:
    if cfg.variant == "raw_eegnet_ce":
        return EEGNetCE(cfg)
    if cfg.variant == "raw_shallow_convnet_ce":
        return ShallowConvNetCE(cfg, filterbank=False)
    if cfg.variant == "raw_filterbank_shallow_ce":
        return ShallowConvNetCE(cfg, filterbank=True)
    if cfg.variant == "raw_multiscale_temporal_ce":
        return MultiScaleTemporalCE(cfg)
    return TokenGraphCompetenceNet(cfg)


__all__ = [
    "BCI_IV_2A_CHANNELS",
    "BCI_IV_2A_COORDS",
    "CompetenceConfig",
    "EEGNetCE",
    "MultiScaleTemporalCE",
    "ShallowConvNetCE",
    "TokenGraphCompetenceNet",
    "anatomical_adjacency",
    "build_competence_model",
]
