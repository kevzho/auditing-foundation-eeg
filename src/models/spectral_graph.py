from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import filtfilt, firwin

try:
    from torch_geometric.nn import GATConv as _PyGGATConv
except ModuleNotFoundError:
    _PyGGATConv = None


class _FallbackGATConv(nn.Module):
    """Small GATConv-compatible fallback used when torch_geometric is absent."""

    def __init__(self, in_channels: int, out_channels: int, heads: int = 1, edge_dim: int = 1, concat: bool = False):
        super().__init__()
        if heads != 1:
            raise ValueError("Fallback GATConv supports heads=1 only.")
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.att_src = nn.Parameter(torch.empty(out_channels))
        self.att_dst = nn.Parameter(torch.empty(out_channels))
        self.att_edge = nn.Linear(edge_dim, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.att_dst.unsqueeze(0))
        nn.init.xavier_uniform_(self.att_edge.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None = None) -> torch.Tensor:
        h = self.lin(x)
        src, dst = edge_index
        if edge_attr is None:
            edge_attr = torch.ones(src.numel(), 1, device=x.device, dtype=x.dtype)
        edge_h = self.att_edge(edge_attr)
        score = (h[src] * self.att_src).sum(-1) + (h[dst] * self.att_dst).sum(-1) + edge_h.sum(-1)
        score = F.leaky_relu(score, negative_slope=0.2)

        out = torch.zeros_like(h)
        for node in range(x.size(0)):
            mask = dst == node
            if not torch.any(mask):
                continue
            weights = torch.softmax(score[mask], dim=0)
            out[node] = (weights.unsqueeze(-1) * h[src[mask]]).sum(dim=0)
        return out + self.bias


GATConv = _PyGGATConv if _PyGGATConv is not None else _FallbackGATConv


class SpectralCovGraph(nn.Module):
    """Spectral covariance graph encoder for EEG trials shaped [batch, C, T]."""

    def __init__(
        self,
        n_channels: int = 22,
        d_s: int = 32,
        sfreq: int = 250,
        eps: float = 1e-5,
        top_k: int = 5,
        fir_order: int = 101,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.d_s = d_s
        self.sfreq = sfreq
        self.eps = eps
        self.top_k = top_k

        self.lambda_mu = nn.Parameter(torch.tensor(0.5))
        self.lambda_beta = nn.Parameter(torch.tensor(0.5))
        self.register_buffer("mu_filter", torch.tensor(firwin(fir_order, [8, 13], pass_zero=False, fs=sfreq), dtype=torch.float32))
        self.register_buffer("beta_filter", torch.tensor(firwin(fir_order, [13, 30], pass_zero=False, fs=sfreq), dtype=torch.float32))

        self.gat: nn.Module | None = None

    def _ensure_gat(self, n_times: int, device: torch.device) -> None:
        if self.gat is None:
            self.gat = GATConv(in_channels=n_times, out_channels=self.d_s, heads=1, edge_dim=1, concat=False)
            self.gat.to(device)

    def _bandpass(self, x: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        x_np = x.detach().cpu().numpy()
        b = coeffs.detach().cpu().numpy()
        filtered = filtfilt(b, [1.0], x_np, axis=-1).copy()
        return torch.as_tensor(filtered, device=x.device, dtype=x.dtype)

    def _band_covariance(self, x_band: torch.Tensor) -> torch.Tensor:
        centered = x_band - x_band.mean(dim=-1, keepdim=True)
        cov = centered @ centered.transpose(-1, -2) / max(x_band.size(-1) - 1, 1)
        trace = cov.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1)
        eye = torch.eye(self.n_channels, device=x_band.device, dtype=x_band.dtype).unsqueeze(0)
        return cov + self.eps * trace / self.n_channels * eye

    def compute_adjacency(self, x: torch.Tensor) -> torch.Tensor:
        x_mu = self._bandpass(x, self.mu_filter)
        x_beta = self._bandpass(x, self.beta_filter)
        sigma_mu = self._band_covariance(x_mu)
        sigma_beta = self._band_covariance(x_beta)
        a_mu = torch.softmax(sigma_mu, dim=-1)
        a_beta = torch.softmax(sigma_beta, dim=-1)
        a = self.lambda_mu * a_mu + self.lambda_beta * a_beta
        return a / a.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    def _topk_edges(self, a_trial: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        k = min(self.top_k, a_trial.size(1))
        values, indices = torch.topk(a_trial, k=k, dim=-1)
        rows = torch.arange(a_trial.size(0), device=a_trial.device).unsqueeze(1).expand_as(indices)
        edge_index = torch.stack([rows.reshape(-1), indices.reshape(-1)], dim=0)
        edge_attr = values.reshape(-1, 1)
        return edge_index, edge_attr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected x with shape [batch, C, T], got {tuple(x.shape)}")
        if x.size(1) != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {x.size(1)}")

        self._ensure_gat(x.size(-1), x.device)
        adjacency = self.compute_adjacency(x)
        outputs = []
        for trial, a_trial in zip(x, adjacency):
            edge_index, edge_attr = self._topk_edges(a_trial)
            outputs.append(self.gat(trial, edge_index, edge_attr=edge_attr))
        return torch.stack(outputs, dim=0)


__all__ = ["SpectralCovGraph"]
