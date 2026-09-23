from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

try:
    from .dual_graph_encoder import DualGraphEncoder
    from .edl_head import EDLHead, edl_full_loss
except ImportError:  # Allows importing as models.* from CalibMI/.
    try:
        from models.dual_graph_encoder import DualGraphEncoder
        from models.edl_head import EDLHead, edl_full_loss
    except ImportError:  # Allows running this file from CalibMI/models.
        from dual_graph_encoder import DualGraphEncoder
        from edl_head import EDLHead, edl_full_loss

try:
    from .eegnet_v2 import DepthwiseTemporalBlock
except ImportError:
    try:
        from models.eegnet_v2 import DepthwiseTemporalBlock
    except ImportError:

        class DepthwiseTemporalBlock(nn.Module):
            """Small fallback for standalone smoke tests when eegnet_v2 is absent."""

            def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7):
                super().__init__()
                padding = kernel_size // 2
                self.net = nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        in_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                        groups=in_channels,
                        bias=False,
                    ),
                    nn.BatchNorm1d(in_channels),
                    nn.ELU(),
                    nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm1d(out_channels),
                    nn.ELU(),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.net(x).transpose(1, 2)


LossLike = Dict[str, torch.Tensor]
EDLOutputs = Dict[str, torch.Tensor]


class EDLSGraph(nn.Module):
    """Full EDL-SGraph model: dual graph encoder, temporal block, EDL head."""

    def __init__(
        self,
        n_channels: int = 22,
        T: int = 750,
        sfreq: int = 250,
        d_s: int = 32,
        d: int = 128,
        n_classes: int = 4,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.T = T
        self.sfreq = sfreq
        self.d_s = d_s
        self.d = d
        self.n_classes = n_classes

        self.graph_encoder = DualGraphEncoder(
            n_channels=n_channels,
            d_s=d_s,
            sfreq=sfreq,
        )
        self.temporal = self._build_temporal_block(d_s=d_s, d=d)
        self.edl_head = EDLHead(d=d, n_classes=n_classes)
        self.s_head = nn.Linear(d, 1)

    def _build_temporal_block(self, d_s: int, d: int) -> nn.Module:
        try:
            return DepthwiseTemporalBlock(in_channels=d_s, out_channels=d)
        except TypeError:
            try:
                return DepthwiseTemporalBlock(d_s, d)
            except TypeError:
                return DepthwiseTemporalBlock(in_ch=d_s, out_ch=d)

    def _temporal_forward(self, h: torch.Tensor) -> torch.Tensor:
        # H: [batch, C, d_s] -> [batch, d_s, C], treating channels as sequence.
        h_seq = h.transpose(1, 2).contiguous()
        z = self.temporal(h_seq)
        if z.dim() != 3:
            raise ValueError(f"Expected temporal output with 3 dims, got {tuple(z.shape)}")
        if z.size(-1) == self.d:
            return z
        if z.size(1) == self.d:
            return z.transpose(1, 2).contiguous()
        raise ValueError(f"Expected temporal output feature dim {self.d}, got {tuple(z.shape)}")

    def _split_loss_inputs(
        self,
        s_target: Optional[Any],
        lambdas: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[torch.Tensor], Optional[Any], Dict[str, Any]]:
        loss_lambdas = dict(lambdas or {})
        labels = loss_lambdas.pop("labels", loss_lambdas.pop("y", None))
        difficulty_target = s_target

        if isinstance(s_target, dict):
            labels = s_target.get("labels", s_target.get("y", labels))
            difficulty_target = s_target.get("s_target", s_target.get("difficulty"))

        return labels, difficulty_target, loss_lambdas

    def forward(
        self,
        x: torch.Tensor,
        s_target: Optional[Any] = None,
        lambdas: Optional[Dict[str, Any]] = None,
    ):
        h = self.graph_encoder(x)
        z_seq = self._temporal_forward(h)
        z = z_seq.mean(dim=1)
        edl_outputs = self.edl_head(z)
        s_pred = self.s_head(z).squeeze(-1)

        if lambdas is not None and s_target is not None:
            labels, difficulty_target, loss_lambdas = self._split_loss_inputs(s_target, lambdas)
            if labels is None:
                raise ValueError(
                    "Full EDL loss requires class labels. Pass them as lambdas['labels'], "
                    "lambdas['y'], or s_target={'labels': ..., 's_target': ...}."
                )
            labels = labels.to(device=x.device) if torch.is_tensor(labels) else torch.as_tensor(labels, device=x.device)
            loss = edl_full_loss(
                edl_outputs=edl_outputs,
                labels=labels,
                s_pred=s_pred,
                s_target=difficulty_target,
                lambdas=loss_lambdas,
            )
            return edl_outputs, s_pred, loss

        return edl_outputs, s_pred

    @torch.no_grad()
    def inference(self, x: torch.Tensor) -> EDLOutputs:
        edl_outputs, _ = self.forward(x)
        return {
            "p_hat": edl_outputs["p_hat"],
            "vacuity": edl_outputs["vacuity"],
            "p_unknown": edl_outputs["p_unknown"],
        }


__all__ = ["EDLSGraph"]


def _smoke_test() -> None:
    torch.manual_seed(0)
    model = EDLSGraph()
    x = torch.randn(2, 22, 750)
    labels = torch.tensor([0, 1], dtype=torch.long)
    s_target = torch.tensor([0.25, 0.75])
    lambdas = {"labels": labels}

    edl_outputs, s_pred = model(x)
    assert edl_outputs["alpha"].shape == (2, 5)
    assert edl_outputs["p_hat"].shape == (2, 5)
    assert edl_outputs["vacuity"].shape == (2,)
    assert edl_outputs["S"].shape == (2,)
    assert edl_outputs["p_unknown"].shape == (2,)
    assert s_pred.shape == (2,)

    inference = model.inference(x)
    assert set(inference) == {"p_hat", "vacuity", "p_unknown"}
    assert inference["p_hat"].shape == (2, 5)

    _, _, loss = model(x, s_target=s_target, lambdas=lambdas)
    assert torch.isfinite(loss["loss"])
    print("EDLSGraph smoke test PASSED")


if __name__ == "__main__":
    _smoke_test()
