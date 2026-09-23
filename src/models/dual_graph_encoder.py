from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .spectral_graph import SpectralCovGraph
except ImportError:  # Allows running this file directly from CalibMI/.
    try:
        from models.spectral_graph import SpectralCovGraph
    except ImportError:
        from spectral_graph import SpectralCovGraph


class _ReliabilityGAT(nn.Module):
    """Dense single-head GAT with source-channel reliability priors."""

    def __init__(self, in_channels: int, out_channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.att_src = nn.Parameter(torch.empty(out_channels))
        self.att_dst = nn.Parameter(torch.empty(out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.att_dst.unsqueeze(0))
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor, reliability: torch.Tensor) -> torch.Tensor:
        if adjacency.dim() == 2:
            adjacency = adjacency.unsqueeze(0).expand(x.size(0), -1, -1)

        h = self.lin(x)
        src_logits = torch.einsum("bcd,d->bc", h, self.att_src)
        dst_logits = torch.einsum("bcd,d->bc", h, self.att_dst)
        logits = dst_logits.unsqueeze(-1) + src_logits.unsqueeze(1)

        edge_mask = adjacency > 0
        edge_logits = torch.log(adjacency.clamp_min(self.eps))
        reliability_logits = torch.log(reliability.clamp_min(self.eps)).view(1, 1, -1)
        logits = F.leaky_relu(logits + edge_logits + reliability_logits, negative_slope=0.2)
        logits = logits.masked_fill(~edge_mask, torch.finfo(logits.dtype).min)

        alpha = torch.softmax(logits, dim=-1)
        alpha = torch.where(edge_mask, alpha, torch.zeros_like(alpha))
        return alpha @ h + self.bias


class DualGraphEncoder(nn.Module):
    """Fuse anatomical and spectral EEG graphs into [batch, C, d_s] node embeddings."""

    CHANNELS_22 = (
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

    ANATOMICAL_EDGES = (
        ("Fz", "FC1"),
        ("Fz", "FCz"),
        ("Fz", "FC2"),
        ("FC3", "FC1"),
        ("FC3", "C5"),
        ("FC3", "C3"),
        ("FC3", "C1"),
        ("FC1", "FCz"),
        ("FC1", "C3"),
        ("FC1", "C1"),
        ("FC1", "Cz"),
        ("FCz", "FC2"),
        ("FCz", "C1"),
        ("FCz", "Cz"),
        ("FCz", "C2"),
        ("FC2", "FC4"),
        ("FC2", "Cz"),
        ("FC2", "C2"),
        ("FC2", "C4"),
        ("FC4", "C2"),
        ("FC4", "C4"),
        ("FC4", "C6"),
        ("C5", "C3"),
        ("C5", "CP3"),
        ("C3", "C1"),
        ("C3", "Cz"),
        ("C3", "C4"),
        ("C3", "CP3"),
        ("C3", "CP1"),
        ("C1", "Cz"),
        ("C1", "CP3"),
        ("C1", "CP1"),
        ("C1", "CPz"),
        ("Cz", "C2"),
        ("Cz", "C4"),
        ("Cz", "CP1"),
        ("Cz", "CPz"),
        ("Cz", "CP2"),
        ("C2", "C4"),
        ("C2", "CPz"),
        ("C2", "CP2"),
        ("C2", "CP4"),
        ("C4", "C6"),
        ("C4", "CP2"),
        ("C4", "CP4"),
        ("C6", "CP4"),
        ("CP3", "CP1"),
        ("CP3", "P1"),
        ("CP1", "CPz"),
        ("CP1", "P1"),
        ("CP1", "Pz"),
        ("CPz", "CP2"),
        ("CPz", "P1"),
        ("CPz", "Pz"),
        ("CPz", "P2"),
        ("CP2", "CP4"),
        ("CP2", "Pz"),
        ("CP2", "P2"),
        ("CP4", "P2"),
        ("P1", "Pz"),
        ("P1", "POz"),
        ("Pz", "P2"),
        ("Pz", "POz"),
        ("P2", "POz"),
    )

    def __init__(
        self,
        n_channels: int = 22,
        d_s: int = 32,
        sfreq: int = 250,
        eps: float = 1e-5,
        k_anat: int = 4,
        k_spec: int = 5,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.d_s = d_s
        self.sfreq = sfreq
        self.eps = eps
        self.k_anat = k_anat
        self.k_spec = k_spec

        self.spectral_graph = SpectralCovGraph(
            n_channels=n_channels,
            d_s=d_s,
            sfreq=sfreq,
            eps=eps,
            top_k=k_spec,
        )
        self.r_logits = nn.Parameter(torch.zeros(n_channels))

        self.gat_anat: _ReliabilityGAT | None = None
        self.gat_spec: _ReliabilityGAT | None = None
        self.register_buffer("A_anat", self._build_anatomical_adjacency(n_channels, k_anat), persistent=False)

    @property
    def lambda_mu(self) -> nn.Parameter:
        return self.spectral_graph.lambda_mu

    @property
    def lambda_beta(self) -> nn.Parameter:
        return self.spectral_graph.lambda_beta

    def _build_anatomical_adjacency(self, n_channels: int, top_k: int) -> torch.Tensor:
        if n_channels != len(self.CHANNELS_22):
            raise ValueError("The hard-coded BCI IV 2a anatomical topology expects n_channels=22.")

        idx = {name: i for i, name in enumerate(self.CHANNELS_22)}
        adjacency = torch.zeros(n_channels, n_channels, dtype=torch.float32)
        for left, right in self.ANATOMICAL_EDGES:
            i, j = idx[left], idx[right]
            adjacency[i, j] = 1.0
            adjacency[j, i] = 1.0

        adjacency.fill_diagonal_(1.0)
        adjacency = self._topk_adjacency(adjacency, top_k)
        return self._masked_row_softmax(adjacency)

    def _ensure_gats(self, n_times: int, device: torch.device) -> None:
        if self.gat_anat is None:
            self.gat_anat = _ReliabilityGAT(n_times, self.d_s, eps=self.eps).to(device)
        if self.gat_spec is None:
            self.gat_spec = _ReliabilityGAT(n_times, self.d_s, eps=self.eps).to(device)

    def _topk_adjacency(self, adjacency: torch.Tensor, top_k: int) -> torch.Tensor:
        if top_k <= 0 or top_k >= adjacency.size(-1):
            return adjacency

        k = min(top_k, adjacency.size(-1))
        values, indices = torch.topk(adjacency, k=k, dim=-1)
        sparse = torch.zeros_like(adjacency)
        sparse.scatter_(-1, indices, values)
        return sparse

    def _masked_row_softmax(self, adjacency: torch.Tensor) -> torch.Tensor:
        mask = adjacency > 0
        logits = adjacency.masked_fill(~mask, torch.finfo(adjacency.dtype).min)
        normalized = torch.softmax(logits, dim=-1)
        return torch.where(mask, normalized, torch.zeros_like(normalized))

    def _effective_reliability(self, reliability_stats: torch.Tensor | None = None) -> torch.Tensor:
        reliability = torch.sigmoid(self.r_logits)
        if reliability_stats is None:
            return reliability

        stats = torch.as_tensor(reliability_stats, device=reliability.device, dtype=reliability.dtype)
        if stats.dim() == 2:
            stats = stats.mean(dim=0)
        if stats.shape != reliability.shape:
            return reliability

        stat_reliability = (1.0 - stats).clamp(self.eps, 1.0)
        return (reliability * stat_reliability).clamp(self.eps, 1.0)

    def forward(self, X: torch.Tensor, reliability_stats: torch.Tensor | None = None) -> torch.Tensor:
        if X.dim() != 3:
            raise ValueError(f"Expected X with shape [batch, C, T], got {tuple(X.shape)}")
        if X.size(1) != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {X.size(1)}")

        self._ensure_gats(X.size(-1), X.device)
        reliability = self._effective_reliability(reliability_stats).to(device=X.device, dtype=X.dtype)

        A_anat = self.A_anat.to(device=X.device, dtype=X.dtype)
        A_spec = self.spectral_graph.compute_adjacency(X)
        A_spec = self._masked_row_softmax(self._topk_adjacency(A_spec, self.k_spec))

        H_anat = self.gat_anat(X, A_anat, reliability)
        H_spec = self.gat_spec(X, A_spec, reliability)
        return H_anat + H_spec


__all__ = ["DualGraphEncoder"]


if __name__ == "__main__":
    encoder = DualGraphEncoder()
    x = torch.randn(2, 22, 750)
    h = encoder(x)
    print(h.shape)
