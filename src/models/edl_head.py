from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Optional


class EDLHead(nn.Module):
    """Dirichlet Evidential Output Head.

    Forward behavior:
      e = ReLU(W_out @ z + b)
      K0 = softplus(raw_K0)
      alpha = e + K0

    Outputs dict with keys:
      - 'alpha': [batch, K + 1]
      - 'p_hat': [batch, K + 1] expected class probabilities (alpha / S)
      - 'vacuity': [batch] scalar uncertainty u(x) = (K + 1) / S
      - 'S': [batch] total evidence sum (sum_k alpha_k)
      - 'p_unknown': [batch] unknown/artifact class probability
    """

    def __init__(
        self,
        in_dim: int = 128,
        num_classes: int = 4,
        d: Optional[int] = None,
        n_classes: Optional[int] = None,
    ):
        super().__init__()
        if d is not None:
            in_dim = d
        if n_classes is not None:
            num_classes = n_classes
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.n_output_classes = num_classes + 1
        self.linear = nn.Linear(in_dim, self.n_output_classes)
        self.raw_K0 = nn.Parameter(torch.log(torch.expm1(torch.tensor(1.0))))

        # Default initialization: small weights (leave PyTorch default),
        # unit test will zero-init weight to guarantee vacuity for untrained check.

    @property
    def K0(self) -> torch.Tensor:
        """Positive learned evidential prior shared by all output classes."""
        return F.softplus(self.raw_K0)

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute evidential outputs from latent z.

        Args:
            z: Tensor of shape [batch, in_dim]

        Returns:
            dict with 'alpha', 'p_hat', 'vacuity', 'S', 'p_unknown'
        """
        # Linear -> ReLU evidence
        logits = self.linear(z)
        e = F.relu(logits)
        alpha = e + self.K0

        S = alpha.sum(dim=1)  # shape [batch]
        # Avoid division by zero
        S_unsq = S.unsqueeze(1)
        p_hat = alpha / (S_unsq + 1e-12)

        vacuity = (self.n_output_classes / (S + 1e-12)).to(alpha.dtype)

        return {
            'alpha': alpha,
            'p_hat': p_hat,
            'vacuity': vacuity,
            'S': S,
            'p_unknown': p_hat[:, -1],
            'K0': self.K0,
        }


def edl_loss(outputs: Dict[str, torch.Tensor],
             targets: torch.Tensor,
             s_pred: Optional[torch.Tensor] = None,
             s_target: Optional[torch.Tensor] = None,
             lambda_B: float = 0.5,
             lambda_KL: float = 0.1,
             lambda_diff: float = 0.1) -> torch.Tensor:
    alpha = outputs["alpha"]
    S = alpha.sum(dim=-1, keepdim=True)
    p_hat = alpha / S

    y = F.one_hot(targets, num_classes=alpha.size(1)).float()

    L_ce = F.cross_entropy(p_hat, targets)
    L_brier = ((p_hat - y) ** 2).sum(dim=-1).mean()

    K0 = outputs.get("K0", torch.tensor(1.0, device=alpha.device, dtype=alpha.dtype))
    e = alpha - K0
    e_tilde = e * (1.0 - y)
    alpha_tilde = e_tilde + K0

    from torch.distributions.dirichlet import Dirichlet

    dir_q = Dirichlet(alpha_tilde)
    dir_p = Dirichlet(torch.ones_like(alpha_tilde) * K0)
    L_kl = torch.distributions.kl.kl_divergence(dir_q, dir_p).mean()

    if s_pred is not None and s_target is not None:
        L_diff = ((s_pred - s_target) ** 2).mean()
    else:
        L_diff = torch.tensor(0.0, device=alpha.device)

    return L_ce + lambda_B * L_brier + lambda_KL * L_kl + lambda_diff * L_diff


def compute_confidence_from_vacuity(edl_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Compute vacuity-based confidence g(x) = 1 - u(x)."""
    vacuity = edl_outputs["vacuity"]
    return (1.0 - vacuity).clamp(min=0.0, max=1.0)


def selective_risk_surrogate(
    probs: torch.Tensor,
    labels: torch.Tensor,
    confidence: torch.Tensor,
    target_coverage: float = 0.6,
    margin: float = 0.05,
) -> torch.Tensor:
    """Smooth selective-risk surrogate at a target confidence coverage.

    Unknown-class labels at index K are ignored for now. The selected-error
    term uses 1 - p_true instead of a hard misclassification indicator so the
    loss remains differentiable with respect to class probabilities.
    """
    if probs.dim() != 2:
        raise ValueError("probs must have shape [N, K+1]")
    if labels.dim() != 1 or labels.size(0) != probs.size(0):
        raise ValueError("labels must have shape [N]")
    if confidence.dim() != 1 or confidence.size(0) != probs.size(0):
        raise ValueError("confidence must have shape [N]")

    confidence = confidence.to(device=probs.device, dtype=probs.dtype).clamp(0.0, 1.0)
    labels = labels.to(device=probs.device).long()
    target_coverage = float(max(0.0, min(1.0, target_coverage)))
    margin = max(float(margin), 1e-6)

    known = (labels >= 0) & (labels < (probs.size(1) - 1))
    if not bool(known.any()):
        return probs.sum() * 0.0

    probs_known = probs[known]
    labels_known = labels[known]
    confidence_known = confidence[known]

    quantile_level = 1.0 - target_coverage
    t = torch.quantile(confidence_known, quantile_level)
    smooth_mask = torch.sigmoid((confidence_known - t) / margin)
    p_true = probs_known.gather(1, labels_known.unsqueeze(1)).squeeze(1)
    soft_error = 1.0 - p_true

    return (smooth_mask * soft_error).mean()


def edl_full_loss(
    edl_outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    s_pred: Optional[torch.Tensor] = None,
    s_target: Optional[Union[torch.Tensor, float]] = None,
    lambdas: Optional[Dict[str, float]] = None,
) -> Dict[str, torch.Tensor]:
    """Compute EDL loss terms plus a vacuity-based selective-risk surrogate."""
    lambda_values = {
        "lambda_B": 0.5,
        "lambda_KL": 0.1,
        "lambda_diff": 0.1,
        "lambda_sel": 0.05,
        "target_coverage": 0.6,
        "margin": 0.05,
    }
    if lambdas is not None:
        lambda_values.update(lambdas)

    alpha = edl_outputs["alpha"]
    p_hat = edl_outputs["p_hat"]
    S = edl_outputs["S"]
    K0 = edl_outputs.get("K0", torch.tensor(1.0, device=alpha.device, dtype=alpha.dtype))

    terms = evidential_loss(
        alpha=alpha,
        p_hat=p_hat,
        S=S,
        y=labels,
        s_pred=s_pred,
        s_target=s_target,
        prior_K0=K0,
        lambda_B=float(lambda_values["lambda_B"]),
        lambda_KL=float(lambda_values["lambda_KL"]),
        lambda_diff=float(lambda_values["lambda_diff"]),
    )
    confidence = compute_confidence_from_vacuity(edl_outputs)
    L_select = selective_risk_surrogate(
        probs=p_hat,
        labels=labels,
        confidence=confidence,
        target_coverage=float(lambda_values["target_coverage"]),
        margin=float(lambda_values["margin"]),
    )
    loss = terms["loss"] + float(lambda_values["lambda_sel"]) * L_select

    return {
        **terms,
        "L_select": L_select,
        "confidence": confidence,
        "loss": loss,
    }


def dirichlet_kl(alpha: torch.Tensor, beta: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute KL(Dir(alpha) || Dir(beta)) for batched alpha.

    Args:
        alpha: [batch, K]
        beta: [K] or [batch, K]. If None, use ones (i.e., uniform Dir(1)).

    Returns:
        kl: [batch] tensor of KL divergences
    """
    if beta is None:
        beta = torch.ones_like(alpha)

    # Ensure shapes
    if beta.dim() == 1:
        beta = beta.unsqueeze(0).expand_as(alpha)

    sum_alpha = alpha.sum(dim=1)
    sum_beta = beta.sum(dim=1)

    t1 = torch.lgamma(sum_alpha) - torch.lgamma(sum_beta)
    t2 = (torch.lgamma(beta) - torch.lgamma(alpha)).sum(dim=1)
    # digamma terms
    t3 = ((alpha - beta) * (torch.digamma(alpha) - torch.digamma(sum_alpha.unsqueeze(1)))).sum(dim=1)

    kl = t1 + t2 + t3
    return kl


def evidential_loss(alpha: torch.Tensor,
                    p_hat: torch.Tensor,
                    S: torch.Tensor,
                    y: torch.Tensor,
                    s_pred: Optional[torch.Tensor] = None,
                    s_target: Optional[Union[torch.Tensor, float]] = None,
                    prior_K0: Optional[torch.Tensor] = None,
                    lambda_B: float = 0.5,
                    lambda_KL: float = 0.1,
                    lambda_diff: float = 0.1) -> Dict[str, torch.Tensor]:
    """Compute the 4-term evidential loss.

    L = L_CE + lambda_B * L_Brier + lambda_KL * KL(Dir(alpha_tilde) || Dir(K0)) + lambda_diff * L_diff

    Args:
        alpha: [batch, K]
        p_hat: [batch, K]
        S: [batch]
        y: either [batch] int labels or [batch, K] one-hot float labels
        s_pred: [batch] predicted scalar difficulty (or None)
        s_target: scalar or batched target difficulty (or None)
        prior_K0: positive scalar Dirichlet prior. If None, use 1.0.
        lambda_B, lambda_KL, lambda_diff: weights

    Returns:
        dict with 'loss' total and per-term losses
    """
    eps = 1e-8
    # Convert y to one-hot
    if y.dim() == 1:
        y_long = y.long()
        y_onehot = torch.zeros_like(alpha).scatter_(1, y_long.unsqueeze(1), 1.0)
    else:
        y_onehot = y.float()

    # L_CE: cross-entropy on p_hat
    # clamp p_hat to avoid log(0)
    p = p_hat.clamp(min=eps, max=1.0)
    L_CE = - (y_onehot * torch.log(p)).sum(dim=1).mean()

    # L_Brier: multiclass Brier score (mean squared error)
    L_Brier = ((p_hat - y_onehot) ** 2).sum(dim=1).mean()

    # KL term: construct alpha_tilde = e * (1 - y_k) + K0
    if prior_K0 is None:
        prior_K0 = torch.tensor(1.0, device=alpha.device, dtype=alpha.dtype)
    else:
        prior_K0 = prior_K0.to(device=alpha.device, dtype=alpha.dtype)
    e = alpha - prior_K0
    alpha_tilde = e * (1.0 - y_onehot) + prior_K0
    KL_term = dirichlet_kl(alpha_tilde, beta=torch.ones_like(alpha_tilde) * prior_K0).mean()

    # L_diff: (s_pred - s_target)^2
    if (s_pred is None) or (s_target is None):
        L_diff = torch.tensor(0.0, device=alpha.device, dtype=alpha.dtype)
    else:
        # allow scalar or batched difficulty targets
        s = s_pred.to(device=alpha.device, dtype=alpha.dtype)
        target = torch.as_tensor(s_target, device=alpha.device, dtype=alpha.dtype)
        L_diff = ((s - target) ** 2).mean()

    loss = L_CE + lambda_B * L_Brier + lambda_KL * KL_term + lambda_diff * L_diff

    return {
        'loss': loss,
        'L_CE': L_CE,
        'L_Brier': L_Brier,
        'KL': KL_term,
        'L_diff': L_diff,
    }


if __name__ == '__main__':
    # Unit test
    torch.manual_seed(0)
    device = torch.device('cpu')
    model = EDLHead(in_dim=128, num_classes=4).to(device)

    # Force untrained-like uniform Dirichlet by zeroing weights and bias
    with torch.no_grad():
        model.linear.weight.zero_()
        if model.linear.bias is not None:
            model.linear.bias.zero_()

    # Random input
    batch = 16
    z = torch.randn(batch, 128, device=device)
    out = model(z)

    vacuity = out['vacuity']  # [batch]
    mean_vacuity = float(vacuity.mean().item())

    print(f"Mean vacuity: {mean_vacuity:.4f}")
    if mean_vacuity > 0.8:
        print('UNIT TEST PASS: vacuity > 0.8 for untrained-like model')
    else:
        print('UNIT TEST FAIL: vacuity <= 0.8')

    probs = torch.softmax(torch.randn(batch, 5, device=device), dim=1)
    labels = torch.randint(0, 5, (batch,), device=device)
    random_outputs = {"vacuity": torch.rand(batch, device=device)}
    confidence = compute_confidence_from_vacuity(random_outputs)
    surrogate = selective_risk_surrogate(
        probs=probs,
        labels=labels,
        confidence=confidence,
        target_coverage=0.6,
        margin=0.05,
    )
    print(f"Selective risk surrogate smoke loss: {float(surrogate.item()):.4f}")
