#!/usr/bin/env python3
"""Diagnose negative transfer on BNCI2014_001 and BNCI2014_004."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DATASETS = ("BNCI2014_001", "BNCI2014_004")
METHOD_ORDER = (
    "baseline",
    "seed_ensemble",
    "augmentation",
    "teacher_student_mixture",
    "mdrm_t_ea",
    "validation_selected_portfolio",
)
METHOD_LABELS = {
    "baseline": "Baseline",
    "seed_ensemble": "Seed ensemble",
    "augmentation": "Augmentation",
    "teacher_student_mixture": "Teacher/student",
    "mdrm_t_ea": "MDRM-T + EA",
    "validation_selected_portfolio": "Portfolio",
}
METRICS = ("accuracy", "Brier", "ECE", "NLL")
EPS = 1e-12

IEEE_FIGURE_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def confidence_from_probability_dir(probability_dir: Path) -> dict[tuple[str, str, str], float]:
    out: dict[tuple[str, str, str], float] = {}
    if not probability_dir.exists():
        return out
    for path in sorted(probability_dir.glob("*_probabilities.npz")):
        loaded = np.load(path)
        metadata = json.loads(str(loaded["metadata_json"]))
        dataset = str(metadata.get("dataset"))
        subject = str(metadata.get("subject"))
        for key in loaded.files:
            if not key.endswith("__eval_proba"):
                continue
            method = key[: -len("__eval_proba")]
            proba = np.asarray(loaded[key], dtype=float)
            row_sums = proba.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums <= 0.0, 1.0, row_sums)
            proba = proba / row_sums
            out[(dataset, subject, method)] = float(proba.max(axis=1).mean())
    return out


def load_dataset_rows(results_root: Path, dataset: str) -> pd.DataFrame:
    result_dir = results_root / f"calibration_workflow_{dataset.lower()}"
    metrics_path = result_dir / f"validation_only_selection_{dataset.lower()}_subject_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    rows = pd.read_csv(metrics_path)
    rows["selected_method"] = ""
    portfolio_path = result_dir / "validation_selected_portfolio_subject_metrics.csv"
    if portfolio_path.exists():
        portfolio = pd.read_csv(portfolio_path)
        portfolio["experiment_key"] = "validation_selected_portfolio"
        rows = pd.concat([rows, portfolio[rows.columns.intersection(portfolio.columns)]], ignore_index=True, sort=False)
        for column in rows.columns:
            if column not in portfolio.columns and column == "selected_method":
                rows[column] = rows[column].fillna("")
    rows = rows[rows["experiment_key"].isin(METHOD_ORDER)].copy()
    rows["subject"] = rows["subject"].astype(str)
    return rows


def attach_confidence(rows: pd.DataFrame, confidence: dict[tuple[str, str, str], float]) -> pd.DataFrame:
    out = rows.copy()
    mean_conf = []
    sources = []
    for row in out.itertuples(index=False):
        dataset = str(row.dataset)
        subject = str(row.subject)
        method = str(row.experiment_key)
        selected = str(getattr(row, "selected_method", "") or "")
        key = (dataset, subject, method)
        source = method
        if method == "validation_selected_portfolio" and key not in confidence and selected:
            key = (dataset, subject, selected)
            source = selected
        value = confidence.get(key, float("nan"))
        mean_conf.append(value)
        sources.append(source if np.isfinite(value) else "")
    out["mean_heldout_confidence"] = mean_conf
    out["confidence_source_method"] = sources
    out["confidence_minus_accuracy"] = out["mean_heldout_confidence"] - out["heldout_accuracy"].astype(float)
    return out


def add_baseline_deltas(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    baseline = out[out["experiment_key"] == "baseline"][
        ["dataset", "subject", *[f"heldout_{metric}" for metric in METRICS]]
    ].rename(columns={f"heldout_{metric}": f"baseline_{metric}" for metric in METRICS})
    out = out.merge(baseline, on=["dataset", "subject"], how="left", validate="many_to_one")
    for metric in METRICS:
        out[f"delta_{metric}_vs_baseline"] = out[f"heldout_{metric}"].astype(float) - out[f"baseline_{metric}"].astype(float)
    return out


def add_brier_ranks(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    out["validation_Brier_rank"] = out.groupby(["dataset", "subject"])["val_Brier"].rank(method="min", ascending=True)
    out["heldout_Brier_rank"] = out.groupby(["dataset", "subject"])["heldout_Brier"].rank(method="min", ascending=True)
    out["heldout_minus_validation_Brier_rank"] = out["heldout_Brier_rank"] - out["validation_Brier_rank"]
    return out


def win_tie_loss(values: pd.Series, lower_is_better: bool) -> tuple[int, int, int]:
    arr = values.to_numpy(dtype=float)
    if lower_is_better:
        wins = int((arr < -EPS).sum())
        losses = int((arr > EPS).sum())
    else:
        wins = int((arr > EPS).sum())
        losses = int((arr < -EPS).sum())
    ties = int(len(arr) - wins - losses)
    return wins, ties, losses


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (dataset, method), part in rows.groupby(["dataset", "experiment_key"]):
        record: dict[str, Any] = {
            "dataset": dataset,
            "method": method,
            "method_label": METHOD_LABELS.get(method, method),
            "n_subjects": int(len(part)),
            "mean_confidence_minus_accuracy": float(part["confidence_minus_accuracy"].mean()),
            "mean_Brier_rank_shift": float(part["heldout_minus_validation_Brier_rank"].mean()),
        }
        for metric in METRICS:
            delta_col = f"delta_{metric}_vs_baseline"
            lower_is_better = metric in {"Brier", "ECE", "NLL"}
            wins, ties, losses = win_tie_loss(part[delta_col], lower_is_better)
            record[f"mean_delta_{metric}_vs_baseline"] = float(part[delta_col].mean())
            record[f"median_delta_{metric}_vs_baseline"] = float(part[delta_col].median())
            record[f"{metric}_wins_ties_losses_vs_baseline"] = f"{wins}/{ties}/{losses}"
        records.append(record)
    return pd.DataFrame(records).sort_values(["dataset", "method"])


def plot_heatmap(rows: pd.DataFrame, out_dir: Path) -> None:
    methods = [m for m in METHOD_ORDER if m in set(rows["experiment_key"])]
    pivot = rows.pivot_table(
        index=["dataset", "subject"],
        columns="experiment_key",
        values="delta_Brier_vs_baseline",
        aggfunc="first",
    ).reindex(columns=methods)
    labels = [f"{dataset} S{subject}" for dataset, subject in pivot.index]
    values = pivot.to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1.0
    vmax = max(vmax, 1e-3)
    with plt.rc_context(IEEE_FIGURE_RC):
        fig, ax = plt.subplots(figsize=(7.16, max(3.2, 0.28 * len(labels))))
        im = ax.imshow(values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(np.arange(len(methods)))
        ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], rotation=35, ha="right")
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_title("Heldout Brier delta vs. baseline")
        cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
        cbar.set_label("Delta Brier")
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(out_dir / f"negative_transfer_heatmap.{ext}", dpi=240, bbox_inches="tight")
        plt.close(fig)


def interpretation_markdown(rows: pd.DataFrame, summary: pd.DataFrame) -> str:
    nonbaseline = rows[rows["experiment_key"] != "baseline"].copy()
    loss_rows = nonbaseline.sort_values("delta_Brier_vs_baseline", ascending=False).head(6)
    lines = [
        "# BNCI Negative-Transfer Diagnostic",
        "",
        "Conclusions are descriptive unless paired statistics are added separately.",
        "",
        "## Subjects Dominating Losses",
        "",
    ]
    for row in loss_rows.itertuples(index=False):
        lines.append(
            f"- {row.dataset} subject {row.subject}, {METHOD_LABELS.get(row.experiment_key, row.experiment_key)}: "
            f"delta Brier {row.delta_Brier_vs_baseline:+.4f}, "
            f"delta accuracy {row.delta_accuracy_vs_baseline:+.4f}, "
            f"delta ECE {row.delta_ECE_vs_baseline:+.4f}."
        )
    lines.extend(["", "## Failure Mode Summary", ""])
    for row in summary[summary["method"] != "baseline"].itertuples(index=False):
        acc_loss = getattr(row, "mean_delta_accuracy_vs_baseline") < -EPS
        brier_loss = getattr(row, "mean_delta_Brier_vs_baseline") > EPS
        ece_loss = getattr(row, "mean_delta_ECE_vs_baseline") > EPS
        modes = []
        if acc_loss:
            modes.append("accuracy")
        if brier_loss:
            modes.append("Brier/calibration")
        if ece_loss:
            modes.append("ECE")
        if not modes:
            modes.append("no mean loss versus baseline")
        lines.append(
            f"- {row.dataset} {METHOD_LABELS.get(row.method, row.method)}: {', '.join(modes)}; "
            f"mean validation-to-heldout Brier rank shift {row.mean_Brier_rank_shift:+.2f}; "
            f"mean confidence minus accuracy {row.mean_confidence_minus_accuracy:+.3f}."
        )
    lines.extend(["", "## Classical Anchor / MDRM-T", ""])
    if "mdrm_t_ea" in set(rows["experiment_key"]):
        lines.append("MDRM-T + EA rows are present and can be compared directly in the CSV outputs.")
    else:
        lines.append("MDRM-T + EA rows are not present for the retained BNCI runs, so no classical EA anchor conclusion is supported yet.")
    lines.extend(["", "## Selection Mismatch", ""])
    mismatch = summary[summary["method"] != "baseline"].sort_values("mean_Brier_rank_shift", ascending=False).head(4)
    for row in mismatch.itertuples(index=False):
        lines.append(
            f"- {row.dataset} {METHOD_LABELS.get(row.method, row.method)}: "
            f"mean heldout-minus-validation Brier rank shift {row.mean_Brier_rank_shift:+.2f}."
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default=Path("results"), type=Path)
    parser.add_argument("--out-dir", default=Path("results") / "paper_calibration_stats" / "negative_transfer", type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frames = [load_dataset_rows(args.results_root, dataset) for dataset in DATASETS]
    rows = pd.concat(frames, ignore_index=True)
    confidence: dict[tuple[str, str, str], float] = {}
    for dataset in DATASETS:
        confidence.update(
            confidence_from_probability_dir(args.results_root / f"calibration_workflow_{dataset.lower()}" / "probabilities")
        )
    rows = attach_confidence(rows, confidence)
    rows = add_baseline_deltas(rows)
    rows = add_brier_ranks(rows)
    summary = summarize(rows)

    rows.to_csv(args.out_dir / "subject_diagnostics.csv", index=False)
    summary.to_csv(args.out_dir / "dataset_summary.csv", index=False)
    plot_heatmap(rows, args.out_dir)
    (args.out_dir / "interpretation.md").write_text(interpretation_markdown(rows, summary), encoding="utf-8")


if __name__ == "__main__":
    main()
