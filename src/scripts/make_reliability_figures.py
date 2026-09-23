#!/usr/bin/env python3
"""Build reliability-diagram artifacts from saved per-trial probabilities."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reliability import reliability_bins, metrics_row


METHOD_LABELS = {
    "baseline": "Baseline",
    "seed_ensemble": "Seed ensemble",
    "augmentation": "Augmentation",
    "teacher_student_mixture": "Teacher/student",
}

METHOD_COLORS = {
    "baseline": "#4d4d4d",
    "seed_ensemble": "#1b9e77",
    "augmentation": "#e6ab02",
    "teacher_student_mixture": "#7570b3",
}

DATASET_ORDER = ["BCI IV-2a", "BCI IIIa", "BNCI2014_001", "BNCI2014_004"]

IEEE_FIGURE_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def _dataset_label(metadata: dict[str, object]) -> str:
    dataset = str(metadata.get("dataset", "unknown"))
    if dataset == "bci4_2a":
        return "BCI IV-2a"
    if dataset == "bci_iiia":
        return "BCI IIIa"
    return dataset


def _iter_probability_files(probability_dirs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in probability_dirs:
        files.extend(sorted(root.rglob("*_probabilities.npz")))
    return sorted(files)


def _read_probability_file(path: Path, n_bins: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    loaded = np.load(path)
    metadata = json.loads(str(loaded["metadata_json"]))
    dataset = _dataset_label(metadata)
    subject = metadata.get("subject", path.stem)
    y_eval = np.asarray(loaded["y_eval"], dtype=int)
    bin_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for key in sorted(loaded.files):
        if not key.endswith("__eval_proba"):
            continue
        method = key[: -len("__eval_proba")]
        if method not in METHOD_LABELS:
            continue
        proba = np.asarray(loaded[key], dtype=float)
        bins = reliability_bins(y_eval, proba, n_bins=n_bins)
        for row in bins.to_dict("records"):
            row.update(
                {
                    "dataset": dataset,
                    "subject": subject,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "source_file": str(path),
                }
            )
            bin_rows.append(row)
        metric = metrics_row(dataset=dataset, subject=subject, decoder=method, y_true=y_eval, proba=proba, n_bins=n_bins)
        metric["method"] = method
        metric["method_label"] = METHOD_LABELS[method]
        metric["source_file"] = str(path)
        metric_rows.append(metric)
    return bin_rows, metric_rows


def _weighted_bins(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, method, bin_id), part in df.groupby(["dataset", "method", "bin"]):
        valid = part.dropna(subset=["confidence", "accuracy"])
        count = float(valid["count"].sum())
        if count <= 0:
            continue
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "bin": int(bin_id),
                "bin_lower": float(valid["bin_lower"].iloc[0]),
                "bin_upper": float(valid["bin_upper"].iloc[0]),
                "count": int(count),
                "confidence": float((valid["confidence"] * valid["count"]).sum() / count),
                "accuracy": float((valid["accuracy"] * valid["count"]).sum() / count),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["gap"] = (out["accuracy"] - out["confidence"]).abs()
    return out.sort_values(["dataset", "method", "bin"])


def _plot_combined(aggregate_bins: pd.DataFrame, out_dir: Path) -> None:
    datasets = [name for name in DATASET_ORDER if name in set(aggregate_bins["dataset"])]
    if not datasets:
        return
    n_cols = 2
    n_rows = int(np.ceil(len(datasets) / n_cols))
    with plt.rc_context(IEEE_FIGURE_RC):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.16, 3.35 * n_rows), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, dataset in zip(axes.ravel(), datasets):
            ax.axis("on")
            ds = aggregate_bins[aggregate_bins["dataset"] == dataset]
            for method in METHOD_LABELS:
                part = ds[ds["method"] == method]
                if part.empty:
                    continue
                ax.plot(
                    part["confidence"],
                    part["accuracy"],
                    marker="o",
                    linewidth=1.5,
                    markersize=3.5,
                    color=METHOD_COLORS.get(method),
                    label=METHOD_LABELS[method],
                )
            ax.plot([0, 1], [0, 1], color="0.45", linestyle="--", linewidth=1.0)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(dataset)
            ax.set_xlabel("Predicted confidence")
            ax.set_ylabel("Empirical accuracy")
            ax.grid(color="0.9", linewidth=0.7)
            ax.set_axisbelow(True)
        handles, labels = axes.ravel()[0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=4, loc="upper center", frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.94), pad=0.8)
        for ext in ("png", "pdf"):
            fig.savefig(out_dir / f"reliability_diagrams.{ext}", dpi=240, bbox_inches="tight")
        plt.close(fig)


def _plot_individual_datasets(aggregate_bins: pd.DataFrame, out_dir: Path) -> None:
    plot_dir = out_dir / "reliability_by_dataset"
    plot_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(IEEE_FIGURE_RC):
        for dataset, ds in aggregate_bins.groupby("dataset"):
            fig, ax = plt.subplots(figsize=(4.0, 3.4))
            for method in METHOD_LABELS:
                part = ds[ds["method"] == method]
                if part.empty:
                    continue
                ax.plot(
                    part["confidence"],
                    part["accuracy"],
                    marker="o",
                    linewidth=1.5,
                    markersize=3.5,
                    color=METHOD_COLORS.get(method),
                    label=METHOD_LABELS[method],
                )
            ax.plot([0, 1], [0, 1], color="0.45", linestyle="--", linewidth=1.0)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(dataset)
            ax.set_xlabel("Predicted confidence")
            ax.set_ylabel("Empirical accuracy")
            ax.grid(color="0.9", linewidth=0.7)
            ax.set_axisbelow(True)
            ax.legend(frameon=False)
            fig.tight_layout(pad=0.6)
            safe_dataset = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in dataset)
            for ext in ("png", "pdf"):
                fig.savefig(plot_dir / f"reliability_{safe_dataset}.{ext}", dpi=240, bbox_inches="tight")
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probability-dir", action="append", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    bin_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for path in _iter_probability_files(args.probability_dir):
        rows, metrics = _read_probability_file(path, args.bins)
        bin_rows.extend(rows)
        metric_rows.extend(metrics)
    if not bin_rows:
        raise SystemExit("No probability artifacts found. Re-run experiments with --save-probabilities first.")

    bins = pd.DataFrame(bin_rows).sort_values(["dataset", "method", "subject", "bin"])
    metrics = pd.DataFrame(metric_rows).sort_values(["dataset", "method", "subject"])
    aggregate_bins = _weighted_bins(bins)

    bins.to_csv(args.out_dir / "reliability_subject_bins.csv", index=False)
    metrics.to_csv(args.out_dir / "reliability_subject_metrics.csv", index=False)
    aggregate_bins.to_csv(args.out_dir / "reliability_aggregate_bins.csv", index=False)
    _plot_combined(aggregate_bins, args.out_dir)
    _plot_individual_datasets(aggregate_bins, args.out_dir)


if __name__ == "__main__":
    main()
