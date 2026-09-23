"""Optional EEGNet baseline for comparison against the reliability pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


@dataclass
class EEGNetConfig:
    epochs: int = 60
    batch_size: int = 32
    lr: float = 1e-3
    dropout: float = 0.5
    patience: int = 12
    random_state: int = 42


@dataclass
class EEGNetEDLConfig(EEGNetConfig):
    """Config for the closed-set EEGNet + EDL sanity baseline."""

    lambda_brier: float = 0.0
    lambda_kl: float = 0.0
    prior: float = 1.0


def _require_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "EEGNet requires PyTorch. Install it with `python3 -m pip install torch` "
            "or use `requirements-eegnet.txt`."
        ) from exc
    return torch, nn, DataLoader, TensorDataset


def _standardize_epochs(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (X_train - mean) / std, (X_test - mean) / std


def _build_eegnet(torch, nn, n_channels: int, n_times: int, n_classes: int, dropout: float):
    class EEGNet(nn.Module):
        def __init__(self):
            super().__init__()
            f1 = 8
            depth = 2
            f2 = f1 * depth
            kernel = min(64, max(16, n_times // 8))
            self.features = nn.Sequential(
                nn.Conv2d(1, f1, kernel_size=(1, kernel), padding=(0, kernel // 2), bias=False),
                nn.BatchNorm2d(f1),
                nn.Conv2d(f1, f2, kernel_size=(n_channels, 1), groups=f1, bias=False),
                nn.BatchNorm2d(f2),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, 4)),
                nn.Dropout(dropout),
                nn.Conv2d(f2, f2, kernel_size=(1, 16), padding=(0, 8), groups=f2, bias=False),
                nn.Conv2d(f2, f2, kernel_size=(1, 1), bias=False),
                nn.BatchNorm2d(f2),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, 8)),
                nn.Dropout(dropout),
                nn.Flatten(),
            )
            with torch.no_grad():
                dummy = torch.zeros(1, 1, n_channels, n_times)
                n_features = int(self.features(dummy).shape[1])
            self.classifier = nn.Linear(n_features, n_classes)

        def forward(self, x):
            return self.classifier(self.features(x))

    return EEGNet()


def _multiclass_brier_np(proba: np.ndarray, y: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    one_hot = np.zeros_like(proba, dtype=float)
    one_hot[np.arange(y.size), y] = 1.0
    return float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))


def _dirichlet_kl(torch, alpha, beta):
    sum_alpha = alpha.sum(dim=1)
    sum_beta = beta.sum(dim=1)
    return (
        torch.lgamma(sum_alpha)
        - torch.lgamma(sum_beta)
        + (torch.lgamma(beta) - torch.lgamma(alpha)).sum(dim=1)
        + ((alpha - beta) * (torch.digamma(alpha) - torch.digamma(sum_alpha.unsqueeze(1)))).sum(dim=1)
    )


def _edl_loss(torch, nn, outputs, labels, lambda_brier: float, lambda_kl: float, prior: float):
    alpha = outputs["alpha"]
    proba = outputs["p_hat"]
    y_onehot = nn.functional.one_hot(labels, num_classes=alpha.size(1)).to(dtype=alpha.dtype)
    nll = -(y_onehot * torch.log(proba.clamp_min(1e-8))).sum(dim=1).mean()
    brier = ((proba - y_onehot) ** 2).sum(dim=1).mean()

    prior_t = torch.as_tensor(float(prior), device=alpha.device, dtype=alpha.dtype)
    evidence = (alpha - prior_t).clamp_min(0.0)
    alpha_tilde = evidence * (1.0 - y_onehot) + prior_t
    beta = torch.ones_like(alpha_tilde) * prior_t
    kl = _dirichlet_kl(torch, alpha_tilde, beta).mean()
    loss = nll + float(lambda_brier) * brier + float(lambda_kl) * kl
    return {"loss": loss, "nll": nll, "brier": brier, "kl": kl}


def _build_eegnet_edl(torch, nn, n_channels: int, n_times: int, n_classes: int, dropout: float, prior: float):
    class EEGNetEDL(nn.Module):
        def __init__(self):
            super().__init__()
            base = _build_eegnet(torch, nn, n_channels, n_times, n_classes, dropout)
            self.features = base.features
            with torch.no_grad():
                dummy = torch.zeros(1, 1, n_channels, n_times)
                n_features = int(self.features(dummy).shape[1])
            self.evidence = nn.Linear(n_features, n_classes)
            self.register_buffer("prior", torch.tensor(float(prior), dtype=torch.float32))

        def forward(self, x):
            z = self.features(x)
            evidence = nn.functional.softplus(self.evidence(z))
            alpha = evidence + self.prior.clamp_min(1e-6)
            strength = alpha.sum(dim=1)
            return {
                "alpha": alpha,
                "p_hat": alpha / strength.unsqueeze(1).clamp_min(1e-12),
                "vacuity": alpha.size(1) / strength.clamp_min(1e-12),
                "S": strength,
            }

    return EEGNetEDL()


def fit_predict_eegnet(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: EEGNetConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit EEGNet and return encoded y_test plus probability predictions.

    EEGNet is included as a modern deep-learning comparison only. The main
    reliability claim remains the calibrated LDA/SVM ensemble.
    """

    torch, nn, DataLoader, TensorDataset = _require_torch()
    cfg = config or EEGNetConfig()
    torch.manual_seed(cfg.random_state)
    np.random.seed(cfg.random_state)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train).astype(np.int64)
    y_test_enc = le.transform(y_test).astype(np.int64)
    X_train_std, X_test_std = _standardize_epochs(X_train, X_test)

    indices = np.arange(y_train_enc.size)
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=cfg.random_state,
        stratify=y_train_enc,
    )

    X_train_t = torch.tensor(X_train_std[train_idx, None, :, :], dtype=torch.float32)
    y_train_t = torch.tensor(y_train_enc[train_idx], dtype=torch.long)
    X_val_t = torch.tensor(X_train_std[val_idx, None, :, :], dtype=torch.float32)
    y_val_t = torch.tensor(y_train_enc[val_idx], dtype=torch.long)
    X_test_t = torch.tensor(X_test_std[:, None, :, :], dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_eegnet(
        torch,
        nn,
        n_channels=int(X_train_std.shape[1]),
        n_times=int(X_train_std.shape[2]),
        n_classes=int(len(le.classes_)),
        dropout=cfg.dropout,
    ).to(device)

    loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    best_state = None
    best_val = float("inf")
    stale = 0

    for _ in range(cfg.epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t.to(device))
            val_loss = float(loss_fn(val_logits, y_val_t.to(device)).item())
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t.to(device))
        proba = torch.softmax(logits, dim=1).cpu().numpy()
    return y_test_enc, proba


def fit_predict_eegnet_edl(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: EEGNetEDLConfig | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, object]]:
    """Fit a closed-set EEGNet + EDL head and return probabilities plus diagnostics.

    This is the intentionally minimal calibration-first baseline: no graph
    encoder, no unknown class, no difficulty head, no reliability priors, and no
    selective training loss.
    """

    torch, nn, DataLoader, TensorDataset = _require_torch()
    cfg = config or EEGNetEDLConfig()
    torch.manual_seed(cfg.random_state)
    np.random.seed(cfg.random_state)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train).astype(np.int64)
    y_test_enc = le.transform(y_test).astype(np.int64)
    X_train_std, X_test_std = _standardize_epochs(X_train, X_test)

    indices = np.arange(y_train_enc.size)
    classes, counts = np.unique(y_train_enc, return_counts=True)
    stratify = y_train_enc if counts.size and counts.min() >= 2 else None
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=cfg.random_state,
        stratify=stratify,
    )

    X_train_t = torch.tensor(X_train_std[train_idx, None, :, :], dtype=torch.float32)
    y_train_t = torch.tensor(y_train_enc[train_idx], dtype=torch.long)
    X_val_t = torch.tensor(X_train_std[val_idx, None, :, :], dtype=torch.float32)
    y_val_t = torch.tensor(y_train_enc[val_idx], dtype=torch.long)
    X_test_t = torch.tensor(X_test_std[:, None, :, :], dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_eegnet_edl(
        torch,
        nn,
        n_channels=int(X_train_std.shape[1]),
        n_times=int(X_train_std.shape[2]),
        n_classes=int(len(le.classes_)),
        dropout=cfg.dropout,
        prior=cfg.prior,
    ).to(device)

    loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=cfg.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-3)
    best_state = None
    best_brier = float("inf")
    best_epoch = 0
    stale = 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(xb)
            loss_terms = _edl_loss(
                torch,
                nn,
                outputs,
                yb,
                lambda_brier=cfg.lambda_brier,
                lambda_kl=cfg.lambda_kl,
                prior=cfg.prior,
            )
            loss_terms["loss"].backward()
            optimizer.step()
            train_losses.append(float(loss_terms["loss"].detach().cpu().item()))

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t.to(device))
            val_proba = val_outputs["p_hat"].detach().cpu().numpy()
        val_brier = _multiclass_brier_np(val_proba, y_val_t.numpy())
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_brier": val_brier})
        if val_brier < best_brier - 1e-7:
            best_brier = val_brier
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_t.to(device))
        proba = test_outputs["p_hat"].detach().cpu().numpy()
        vacuity = test_outputs["vacuity"].detach().cpu().numpy()
        alpha = test_outputs["alpha"].detach().cpu().numpy()

    pred = proba.argmax(axis=1)
    correct = pred == y_test_enc
    vacuity_correct = float(vacuity[correct].mean()) if correct.any() else float("nan")
    vacuity_wrong = float(vacuity[~correct].mean()) if (~correct).any() else float("nan")
    info = {
        "classes": le.classes_.tolist(),
        "loss": {
            "lambda_brier": float(cfg.lambda_brier),
            "lambda_kl": float(cfg.lambda_kl),
            "prior": float(cfg.prior),
        },
        "best_epoch": int(best_epoch),
        "best_val_brier": float(best_brier),
        "epochs_ran": len(history),
        "history": history,
        "vacuity_correct_mean": vacuity_correct,
        "vacuity_wrong_mean": vacuity_wrong,
        "vacuity_wrong_minus_correct": float(vacuity_wrong - vacuity_correct),
    }
    return y_test_enc, {"p_hat": proba, "vacuity": vacuity, "alpha": alpha}, info


def fit_predict_eegnet_edl_calibrated(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    config: EEGNetEDLConfig | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray], dict[str, object]]:
    """Fit EDL once and return validation-calibration plus eval outputs.

    The model is trained on ``X_train`` only. ``X_calib`` is used only for
    post-hoc calibration, and ``X_eval`` is used only for final metrics.
    """

    torch, nn, DataLoader, TensorDataset = _require_torch()
    cfg = config or EEGNetEDLConfig()
    torch.manual_seed(cfg.random_state)
    np.random.seed(cfg.random_state)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train).astype(np.int64)
    y_calib_enc = le.transform(y_calib).astype(np.int64)
    y_eval_enc = le.transform(y_eval).astype(np.int64)
    X_train_std, X_calib_std = _standardize_epochs(X_train, X_calib)
    _, X_eval_std = _standardize_epochs(X_train, X_eval)

    indices = np.arange(y_train_enc.size)
    classes, counts = np.unique(y_train_enc, return_counts=True)
    stratify = y_train_enc if counts.size and counts.min() >= 2 else None
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=cfg.random_state,
        stratify=stratify,
    )

    X_train_t = torch.tensor(X_train_std[train_idx, None, :, :], dtype=torch.float32)
    y_train_t = torch.tensor(y_train_enc[train_idx], dtype=torch.long)
    X_val_t = torch.tensor(X_train_std[val_idx, None, :, :], dtype=torch.float32)
    y_val_t = torch.tensor(y_train_enc[val_idx], dtype=torch.long)
    X_calib_t = torch.tensor(X_calib_std[:, None, :, :], dtype=torch.float32)
    X_eval_t = torch.tensor(X_eval_std[:, None, :, :], dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_eegnet_edl(
        torch,
        nn,
        n_channels=int(X_train_std.shape[1]),
        n_times=int(X_train_std.shape[2]),
        n_classes=int(len(le.classes_)),
        dropout=cfg.dropout,
        prior=cfg.prior,
    ).to(device)

    loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=cfg.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-3)
    best_state = None
    best_brier = float("inf")
    best_epoch = 0
    stale = 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(xb)
            loss_terms = _edl_loss(
                torch,
                nn,
                outputs,
                yb,
                lambda_brier=cfg.lambda_brier,
                lambda_kl=cfg.lambda_kl,
                prior=cfg.prior,
            )
            loss_terms["loss"].backward()
            optimizer.step()
            train_losses.append(float(loss_terms["loss"].detach().cpu().item()))

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t.to(device))
            val_proba = val_outputs["p_hat"].detach().cpu().numpy()
        val_brier = _multiclass_brier_np(val_proba, y_val_t.numpy())
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_brier": val_brier})
        if val_brier < best_brier - 1e-7:
            best_brier = val_brier
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    def collect(X_t):
        model.eval()
        with torch.no_grad():
            outputs = model(X_t.to(device))
            return {
                "p_hat": outputs["p_hat"].detach().cpu().numpy(),
                "vacuity": outputs["vacuity"].detach().cpu().numpy(),
                "alpha": outputs["alpha"].detach().cpu().numpy(),
            }

    calib_outputs = collect(X_calib_t)
    eval_outputs = collect(X_eval_t)
    pred = eval_outputs["p_hat"].argmax(axis=1)
    correct = pred == y_eval_enc
    vacuity = eval_outputs["vacuity"]
    vacuity_correct = float(vacuity[correct].mean()) if correct.any() else float("nan")
    vacuity_wrong = float(vacuity[~correct].mean()) if (~correct).any() else float("nan")
    info = {
        "classes": le.classes_.tolist(),
        "loss": {
            "lambda_brier": float(cfg.lambda_brier),
            "lambda_kl": float(cfg.lambda_kl),
            "prior": float(cfg.prior),
        },
        "best_epoch": int(best_epoch),
        "best_val_brier": float(best_brier),
        "epochs_ran": len(history),
        "history": history,
        "vacuity_correct_mean": vacuity_correct,
        "vacuity_wrong_mean": vacuity_wrong,
        "vacuity_wrong_minus_correct": float(vacuity_wrong - vacuity_correct),
    }
    return y_calib_enc, calib_outputs, y_eval_enc, eval_outputs, info
