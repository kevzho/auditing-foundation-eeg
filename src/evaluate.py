"""Evaluation utilities for calibration-focused EEG BCI experiments."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def _to_numpy(x: Any) -> np.ndarray:
    """Convert NumPy-like or torch tensors to a NumPy array."""

    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x)


def _as_probability_matrix(probs: Any) -> np.ndarray:
    probs = _to_numpy(probs).astype(float, copy=False)
    if probs.ndim != 2:
        raise ValueError(f"Expected probs with shape (n_trials, n_classes), got {probs.shape}")
    if probs.shape[0] == 0:
        return probs
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums <= 0.0, 1.0, row_sums)
    return probs / row_sums


def _as_label_vector(labels: Any, n_trials: int | None = None) -> np.ndarray:
    labels = _to_numpy(labels).astype(int, copy=False).reshape(-1)
    if n_trials is not None and labels.size != n_trials:
        raise ValueError(f"Expected {n_trials} labels, got {labels.size}")
    return labels


def _as_score_vector(scores: Any, n_trials: int) -> np.ndarray:
    scores = _to_numpy(scores).astype(float, copy=False).reshape(-1)
    if scores.size != n_trials:
        raise ValueError(f"Expected {n_trials} abstention scores, got {scores.size}")
    return scores


def _one_hot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    if labels.size == 0:
        return np.zeros((0, n_classes), dtype=float)
    if labels.min() < 0 or labels.max() >= n_classes:
        raise ValueError(
            f"Labels must be in [0, {n_classes - 1}] for probs with {n_classes} classes"
        )
    y = np.zeros((labels.size, n_classes), dtype=float)
    y[np.arange(labels.size), labels] = 1.0
    return y


def _extract_probs(model_outputs: Any) -> np.ndarray:
    if isinstance(model_outputs, dict):
        for key in ("probs", "proba", "probabilities", "p_hat"):
            if key in model_outputs:
                return _as_probability_matrix(model_outputs[key])
        raise KeyError("Model output dict must contain one of: probs, proba, probabilities, p_hat")
    return _as_probability_matrix(model_outputs)


def compute_ece(probs: Any, labels: Any, n_bins: int = 15) -> tuple[float, dict[str, np.ndarray]]:
    """Compute top-label ECE with equal-mass confidence bins.

    Returns:
        (ece, bin_data), where bin_data has arrays ``conf``, ``acc``, and
        ``counts``. Confidence is the max predicted class probability.
    """

    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    probs = _as_probability_matrix(probs)
    labels = _as_label_vector(labels, probs.shape[0])
    n_trials = labels.size
    if n_trials == 0:
        empty = np.array([], dtype=float)
        return float("nan"), {"conf": empty, "acc": empty, "counts": empty.astype(int)}

    _one_hot(labels, probs.shape[1])
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)

    order = np.argsort(conf, kind="mergesort")
    bin_indices = np.array_split(order, min(n_bins, n_trials))

    bin_conf = np.array([conf[idx].mean() for idx in bin_indices], dtype=float)
    bin_acc = np.array([correct[idx].mean() for idx in bin_indices], dtype=float)
    counts = np.array([idx.size for idx in bin_indices], dtype=int)

    ece = np.sum((counts / n_trials) * np.abs(bin_acc - bin_conf))
    return float(ece), {"conf": bin_conf, "acc": bin_acc, "counts": counts}


def compute_brier(probs: Any, labels: Any) -> float:
    """Compute multiclass Brier score: mean sum_k (p_k - y_k)^2."""

    probs = _as_probability_matrix(probs)
    labels = _as_label_vector(labels, probs.shape[0])
    y = _one_hot(labels, probs.shape[1])
    if labels.size == 0:
        return float("nan")
    return float(np.mean(np.sum((probs - y) ** 2, axis=1)))


def compute_risk_coverage(
    probs: Any,
    labels: Any,
    abstention_scores: Any,
    n_thresholds: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Sweep abstention-score thresholds and return coverage/risk arrays.

    Larger abstention scores are treated as more uncertain, so a trial is
    covered when ``abstention_score <= threshold``.
    """

    if n_thresholds <= 0:
        raise ValueError("n_thresholds must be positive")

    probs = _as_probability_matrix(probs)
    labels = _as_label_vector(labels, probs.shape[0])
    scores = _as_score_vector(abstention_scores, probs.shape[0])
    if labels.size == 0:
        empty = np.array([], dtype=float)
        return empty, empty

    _one_hot(labels, probs.shape[1])
    pred = probs.argmax(axis=1)
    thresholds = np.linspace(np.min(scores), np.max(scores), n_thresholds)
    coverages = np.empty(n_thresholds, dtype=float)
    risks = np.empty(n_thresholds, dtype=float)

    for i, threshold in enumerate(thresholds):
        covered = scores <= threshold
        coverages[i] = covered.mean()
        risks[i] = np.mean(pred[covered] != labels[covered]) if covered.any() else np.nan

    return coverages, risks


def compute_selective_acc(
    probs: Any,
    labels: Any,
    abstention_scores: Any,
    target_coverage: float,
) -> tuple[float, float]:
    """Return selective accuracy at the threshold closest to target coverage."""

    probs = _as_probability_matrix(probs)
    labels = _as_label_vector(labels, probs.shape[0])
    scores = _as_score_vector(abstention_scores, probs.shape[0])
    target_coverage = float(target_coverage)
    if not 0.0 <= target_coverage <= 1.0:
        raise ValueError("target_coverage must be in [0, 1]")
    if labels.size == 0:
        return float("nan"), float("nan")

    _one_hot(labels, probs.shape[1])
    pred = probs.argmax(axis=1)
    thresholds = np.unique(scores)
    coverages = np.array([(scores <= threshold).mean() for threshold in thresholds], dtype=float)
    best = int(np.argmin(np.abs(coverages - target_coverage)))
    covered = scores <= thresholds[best]
    actual_coverage = float(covered.mean())
    selective_acc = float(np.mean(pred[covered] == labels[covered])) if covered.any() else float("nan")
    return selective_acc, actual_coverage


def edl_abstention_vacuity(edl_outputs: dict[str, Any]) -> np.ndarray:
    """Return EDL vacuity u(x), using ``vacuity`` or deriving it from alpha."""

    if "vacuity" in edl_outputs:
        return _to_numpy(edl_outputs["vacuity"]).astype(float, copy=False).reshape(-1)
    if "alpha" not in edl_outputs:
        raise KeyError("edl_outputs must contain 'vacuity' or 'alpha'")

    alpha = _to_numpy(edl_outputs["alpha"]).astype(float, copy=False)
    if alpha.ndim != 2:
        raise ValueError(f"Expected alpha with shape (n_trials, n_classes), got {alpha.shape}")
    total_evidence = alpha.sum(axis=1)
    return alpha.shape[1] / np.maximum(total_evidence, 1e-12)


def edl_abstention_unknown(edl_outputs: dict[str, Any]) -> np.ndarray:
    """Return EDL unknown/artifact probability p_unknown."""

    if "p_unknown" in edl_outputs:
        return _to_numpy(edl_outputs["p_unknown"]).astype(float, copy=False).reshape(-1)
    if "p_hat" in edl_outputs:
        p_hat = _as_probability_matrix(edl_outputs["p_hat"])
        return p_hat[:, -1]
    if "alpha" in edl_outputs:
        alpha = _as_probability_matrix(edl_outputs["alpha"])
        return alpha[:, -1]
    raise KeyError("edl_outputs must contain 'p_unknown', 'p_hat', or 'alpha'")


def evaluate_model(
    model_fn: Callable[[Any], Any],
    X_eval: Any,
    y_eval: Any,
    abstention_fn: Callable[[Any], Any],
) -> dict[str, Any]:
    """Run model evaluation and return calibration/selective metrics."""

    model_outputs = model_fn(X_eval)
    probs = _extract_probs(model_outputs)
    labels = _as_label_vector(y_eval, probs.shape[0])
    pred = probs.argmax(axis=1)

    try:
        abstention_scores = abstention_fn(model_outputs)
    except (AttributeError, KeyError, TypeError, ValueError):
        abstention_scores = abstention_fn(probs)
    abstention_scores = _as_score_vector(abstention_scores, probs.shape[0])

    ece, ece_bins = compute_ece(probs, labels)
    brier = compute_brier(probs, labels)
    coverages, risks = compute_risk_coverage(probs, labels, abstention_scores)
    selective = {
        coverage: {
            "accuracy": acc,
            "coverage": actual_coverage,
        }
        for coverage, (acc, actual_coverage) in (
            (coverage, compute_selective_acc(probs, labels, abstention_scores, coverage))
            for coverage in (0.4, 0.6, 0.8)
        )
    }

    return {
        "accuracy": float(np.mean(pred == labels)) if labels.size else float("nan"),
        "ece": ece,
        "ece_bins": ece_bins,
        "brier": brier,
        "selective_acc": selective,
        "risk_coverage": {
            "coverages": coverages,
            "risks": risks,
        },
        "abstention_scores": abstention_scores,
    }
