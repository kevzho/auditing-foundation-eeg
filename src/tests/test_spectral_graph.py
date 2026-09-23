import torch

from models.spectral_graph import SpectralCovGraph


def main():
    torch.manual_seed(0)
    batch, channels, time = 2, 22, 750
    model = SpectralCovGraph(n_channels=channels, d_s=32, sfreq=250, eps=1e-5)
    x = torch.randn(batch, channels, time)

    adjacency = model.compute_adjacency(x)
    assert (adjacency >= 0).all()
    assert torch.allclose(adjacency.sum(dim=-1), torch.ones(batch, channels), atol=1e-5)

    h = model(x)
    assert h.shape == (batch, channels, 32)

    loss = h.pow(2).mean()
    loss.backward()
    assert model.lambda_mu.grad is not None and torch.isfinite(model.lambda_mu.grad)
    assert model.lambda_beta.grad is not None and torch.isfinite(model.lambda_beta.grad)

    print("SpectralCovGraph smoke test PASSED")


if __name__ == "__main__":
    main()
