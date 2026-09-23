"""Reliability metrics for calibrated selective BCI decoding."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


def _weighted_reliability_bins(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bin_id, bin_df in df.groupby("bin"):
        valid = bin_df.dropna(subset=["confidence", "accuracy"])
        count = float(valid["count"].sum())
        if count <= 0:
            continue
        confidence = float((valid["confidence"] * valid["count"]).sum() / count)
        accuracy = float((valid["accuracy"] * valid["count"]).sum() / count)
        rows.append({"bin": bin_id, "confidence": confidence, "accuracy": accuracy, "count": int(count)})
    return pd.DataFrame(rows)


def _as_probability_matrix(proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba, dtype=float)
    if proba.ndim != 2:
        raise ValueError(f"Expected proba with shape (n_trials, n_classes), got {proba.shape}")
    row_sums = proba.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
    return proba / row_sums


def _one_hot(y_true: np.ndarray, n_classes: int) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=int)
    out = np.zeros((y_true.size, n_classes), dtype=float)
    out[np.arange(y_true.size), y_true] = 1.0
    return out


def brier_score(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Return multiclass Brier score; lower is better."""

    proba = _as_probability_matrix(proba)
    y = _one_hot(np.asarray(y_true, dtype=int), proba.shape[1])
    return float(np.mean(np.sum((proba - y) ** 2, axis=1)))


def reliability_bins(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Top-label reliability diagram data.

    Bins are confidence intervals over max predicted probability. The returned
    table can be plotted directly as confidence vs empirical accuracy.
    """

    proba = _as_probability_matrix(proba)
    y_true = np.asarray(y_true, dtype=int)
    conf = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        rows.append(
            {
                "bin": i,
                "bin_lower": lo,
                "bin_upper": hi,
                "count": int(mask.sum()),
                "confidence": float(conf[mask].mean()) if mask.any() else float("nan"),
                "accuracy": float(correct[mask].mean()) if mask.any() else float("nan"),
                "gap": float(abs(correct[mask].mean() - conf[mask].mean())) if mask.any() else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    """Return top-label Expected Calibration Error."""

    bins = reliability_bins(y_true, proba, n_bins=n_bins)
    total = float(bins["count"].sum())
    if total == 0.0:
        return float("nan")
    weighted_gap = (bins["count"].astype(float) * bins["gap"].fillna(0.0)).sum()
    return float(weighted_gap / total)


def risk_coverage_rows(
    *,
    dataset: str,
    subject: str | int,
    decoder: str,
    y_true: np.ndarray,
    proba: np.ndarray,
    thresholds: tuple[float, ...],
) -> list[dict[str, float | int | str]]:
    proba = _as_probability_matrix(proba)
    y_true = np.asarray(y_true, dtype=int)
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    rows: list[dict[str, float | int | str]] = []
    for threshold in thresholds:
        mask = conf >= float(threshold)
        rows.append(
            {
                "dataset": dataset,
                "subject": subject,
                "decoder": decoder,
                "threshold": float(threshold),
                "coverage": float(mask.mean()),
                "risk": float(np.mean(pred[mask] != y_true[mask])) if mask.any() else float("nan"),
                "accuracy": float(np.mean(pred[mask] == y_true[mask])) if mask.any() else float("nan"),
                "n_covered": int(mask.sum()),
            }
        )
    return rows


def metrics_row(
    *,
    dataset: str,
    subject: str | int,
    decoder: str,
    y_true: np.ndarray,
    proba: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float | int | str]:
    proba = _as_probability_matrix(proba)
    y_true = np.asarray(y_true, dtype=int)
    pred = proba.argmax(axis=1)
    return {
        "dataset": dataset,
        "subject": subject,
        "decoder": decoder,
        "n_trials": int(y_true.size),
        "accuracy_all": float(np.mean(pred == y_true)),
        "mean_confidence": float(proba.max(axis=1).mean()),
        "ece": expected_calibration_error(y_true, proba, n_bins=n_bins),
        "brier": brier_score(y_true, proba),
    }


def reliability_rows(
    *,
    dataset: str,
    subject: str | int,
    decoder: str,
    y_true: np.ndarray,
    proba: np.ndarray,
    n_bins: int = 10,
) -> list[dict[str, float | int | str]]:
    rows = reliability_bins(y_true, proba, n_bins=n_bins).to_dict("records")
    for row in rows:
        row["dataset"] = dataset
        row["subject"] = subject
        row["decoder"] = decoder
    return rows


def write_reliability_diagrams(reliability_df: pd.DataFrame, out_dir: Path) -> None:
    """Write one reliability diagram PNG per dataset when matplotlib is available."""

    mpl_config = out_dir / ".mplconfig"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return

    plot_dir = out_dir / "reliability_diagrams"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for dataset, ds_df in reliability_df.groupby("dataset"):
        fig, ax = plt.subplots(figsize=(6, 5))
        for decoder, dec_df in ds_df.groupby("decoder"):
            grouped = _weighted_reliability_bins(dec_df)
            if not grouped.empty:
                ax.plot(grouped["confidence"], grouped["accuracy"], marker="o", label=decoder)
        ax.plot([0, 1], [0, 1], color="0.4", linestyle="--", linewidth=1, label="perfect")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted confidence")
        ax.set_ylabel("Empirical accuracy")
        ax.set_title(f"Reliability diagram: {dataset}")
        ax.legend()
        fig.tight_layout()
        safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(dataset))
        fig.savefig(plot_dir / f"{safe_name}.png", dpi=180)
        plt.close(fig)
