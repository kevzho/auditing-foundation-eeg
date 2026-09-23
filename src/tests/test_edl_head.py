import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.edl_head import (
    EDLHead,
    compute_confidence_from_vacuity,
    edl_full_loss,
    edl_loss,
    selective_risk_surrogate,
)


def test_edl_head_unknown_class_and_learned_prior():
    model = EDLHead(d=128, n_classes=4)
    z = torch.randn(8, 128)
    y = torch.zeros(8, dtype=torch.long)

    out = model(z)
    loss = edl_loss(out, y)
    K0 = model.K0.detach()

    assert model.num_classes == 4
    assert model.n_output_classes == 5
    assert out["alpha"].shape == (8, 5)
    assert out["p_hat"].shape == (8, 5)
    assert out["vacuity"].shape == (8,)
    assert out["S"].shape == (8,)
    assert out["p_unknown"].shape == (8,)
    assert torch.all(out["alpha"] >= K0)
    assert torch.allclose(out["p_hat"].sum(-1),
                          torch.ones(8), atol=1e-5)
    assert torch.all(out["vacuity"] >= 0.0)
    assert torch.all(out["vacuity"] <= 1.0 + 1e-5)
    assert torch.allclose(out["p_unknown"], out["p_hat"][:, -1])
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_selective_surrogate_and_full_loss_are_finite():
    model = EDLHead(d=128, n_classes=4)
    z = torch.randn(8, 128)
    labels = torch.tensor([0, 1, 2, 3, 0, 1, 4, 4], dtype=torch.long)

    out = model(z)
    confidence = compute_confidence_from_vacuity(out)
    selective_loss = selective_risk_surrogate(
        out["p_hat"],
        labels,
        confidence,
        target_coverage=0.6,
        margin=0.05,
    )
    full_loss = edl_full_loss(out, labels)

    assert confidence.shape == (8,)
    assert torch.all(confidence >= 0.0)
    assert torch.all(confidence <= 1.0)
    assert selective_loss.ndim == 0 and torch.isfinite(selective_loss)
    assert full_loss["loss"].ndim == 0 and torch.isfinite(full_loss["loss"])
    assert full_loss["L_select"].ndim == 0 and torch.isfinite(full_loss["L_select"])


def main():
    test_edl_head_unknown_class_and_learned_prior()
    test_selective_surrogate_and_full_loss_are_finite()
    print("EDLHead + edl_loss smoke test PASSED")


if __name__ == "__main__":
    main()
