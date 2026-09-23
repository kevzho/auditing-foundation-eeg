#!/usr/bin/env python3
"""Build validation-locked selective prediction/risk-coverage artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reliability import _as_probability_matrix


METHOD_LABELS = {
    "baseline": "Baseline",
    "seed_ensemble": "Seed ensemble",
    "augmentation": "Augmentation",
    "teacher_student_mixture": "Teacher/student",
    "mdrm_t_ea": "MDRM-T + EA",
    "validation_selected_portfolio": "Portfolio",
}
METHOD_COLORS = {
    "baseline": "#4d4d4d",
    "seed_ensemble": "#1b9e77",
    "augmentation": "#e6ab02",
    "teacher_student_mixture": "#7570b3",
    "mdrm_t_ea": "#d95f02",
    "validation_selected_portfolio": "#2c7fb8",
}
DATASET_ORDER = ["BCI IV-2a", "BCI IIIa", "BNCI2014_001", "BNCI2014_004", "PhysionetMI"]
TARGET_COVERAGES = (0.40, 0.60, 0.80, 0.90)
EPS = 1e-12

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


def dataset_label(metadata: dict[str, Any]) -> str:
    dataset = str(metadata.get("dataset", "unknown"))
    if dataset == "bci4_2a":
        return "BCI IV-2a"
    if dataset == "bci_iiia":
        return "BCI IIIa"
    return dataset


def iter_probability_files(probability_dirs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in probability_dirs:
        files.extend(sorted(root.rglob("*_probabilities.npz")))
    return sorted(dict.fromkeys(files))


def threshold_for_target(confidence: np.ndarray, target_coverage: float) -> float:
    if confidence.size == 0:
        return float("nan")
    if not 0.0 < float(target_coverage) <= 1.0:
        raise ValueError(f"coverage target must be in (0, 1], got {target_coverage}")
    kth = int(np.ceil((1.0 - float(target_coverage)) * confidence.size))
    kth = min(max(kth, 0), confidence.size - 1)
    return float(np.sort(confidence)[kth])


def load_subject_json(probability_path: Path, search_dirs: list[Path]) -> dict[str, Any]:
    stem = probability_path.stem.removesuffix("_probabilities")
    candidates = [probability_path.parent.parent / f"{stem}.json"]
    candidates.extend(root / f"{stem}.json" for root in search_dirs)
    for path in candidates:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def stored_thresholds_by_method(subject_payload: dict[str, Any]) -> dict[str, dict[float, float]]:
    out: dict[str, dict[float, float]] = {}
    for method, experiment in subject_payload.get("experiments", {}).items():
        rows = experiment.get("metadata", {}).get("abstention_thresholds", [])
        method_thresholds: dict[float, float] = {}
        for row in rows:
            try:
                method_thresholds[round(float(row["target_coverage"]), 6)] = float(row["threshold"])
            except (KeyError, TypeError, ValueError):
                continue
        if method_thresholds:
            out[method] = method_thresholds
    return out


def selective_row(
    *,
    dataset: str,
    subject: object,
    method: str,
    y_val: np.ndarray,
    y_eval: np.ndarray,
    val_proba: np.ndarray,
    eval_proba: np.ndarray,
    target_coverage: float,
    stored_threshold: float | None,
    source_file: Path,
) -> dict[str, object]:
    val_proba = _as_probability_matrix(val_proba)
    eval_proba = _as_probability_matrix(eval_proba)
    val_conf = val_proba.max(axis=1)
    eval_conf = eval_proba.max(axis=1)
    threshold = (
        float(stored_threshold)
        if stored_threshold is not None
        else threshold_for_target(val_conf, target_coverage)
    )
    val_mask = val_conf >= threshold
    eval_mask = eval_conf >= threshold
    val_pred = val_proba.argmax(axis=1)
    eval_pred = eval_proba.argmax(axis=1)
    y_val = np.asarray(y_val, dtype=int)
    y_eval = np.asarray(y_eval, dtype=int)
    return {
        "dataset": dataset,
        "subject": subject,
        "method": method,
        "method_label": METHOD_LABELS.get(method, method),
        "target_coverage": float(target_coverage),
        "threshold": float(threshold),
        "threshold_source": "stored_validation_locked_threshold" if stored_threshold is not None else "validation_confidence_quantile",
        "threshold_selection_split": "validation",
        "eval_labels_used_for_threshold": False,
        "heldout_curve_type": "validation_selected_threshold",
        "val_coverage": float(val_mask.mean()),
        "val_selective_accuracy": float(np.mean(val_pred[val_mask] == y_val[val_mask])) if val_mask.any() else float("nan"),
        "val_risk": float(np.mean(val_pred[val_mask] != y_val[val_mask])) if val_mask.any() else float("nan"),
        "heldout_coverage": float(eval_mask.mean()),
        "heldout_selective_accuracy": float(np.mean(eval_pred[eval_mask] == y_eval[eval_mask])) if eval_mask.any() else float("nan"),
        "heldout_risk": float(np.mean(eval_pred[eval_mask] != y_eval[eval_mask])) if eval_mask.any() else float("nan"),
        "heldout_n_covered": int(eval_mask.sum()),
        "heldout_accuracy_all": float(np.mean(eval_pred == y_eval)),
        "mean_val_confidence": float(val_conf.mean()),
        "mean_heldout_confidence": float(eval_conf.mean()),
        "source_file": str(source_file),
    }


def read_probability_file(path: Path, targets: tuple[float, ...], subject_json_dirs: list[Path]) -> list[dict[str, object]]:
    loaded = np.load(path)
    metadata = json.loads(str(loaded["metadata_json"]))
    if metadata.get("validation_discipline", {}).get("eval_labels_used_for_selection") is not False:
        raise ValueError(f"{path}: probability metadata does not prove eval_labels_used_for_selection=false")
    subject_payload = load_subject_json(path, subject_json_dirs)
    if subject_payload and subject_payload.get("validation_discipline", {}).get("eval_labels_used_for_selection") is not False:
        raise ValueError(f"{path}: subject JSON does not prove eval_labels_used_for_selection=false")
    stored_thresholds = stored_thresholds_by_method(subject_payload)
    dataset = dataset_label(metadata)
    subject = metadata.get("subject", path.stem)
    y_val = np.asarray(loaded["y_val"], dtype=int)
    y_eval = np.asarray(loaded["y_eval"], dtype=int)
    rows: list[dict[str, object]] = []
    for key in sorted(loaded.files):
        if not key.endswith("__val_proba"):
            continue
        method = key[: -len("__val_proba")]
        eval_key = f"{method}__eval_proba"
        if eval_key not in loaded.files:
            continue
        for target in targets:
            stored_threshold = stored_thresholds.get(method, {}).get(round(float(target), 6))
            rows.append(
                selective_row(
                    dataset=dataset,
                    subject=subject,
                    method=method,
                    y_val=y_val,
                    y_eval=y_eval,
                    val_proba=np.asarray(loaded[key], dtype=float),
                    eval_proba=np.asarray(loaded[eval_key], dtype=float),
                    target_coverage=target,
                    stored_threshold=stored_threshold,
                    source_file=path,
                )
            )
    return rows


def summarize(subject_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, method, target), part in subject_rows.groupby(["dataset", "method", "target_coverage"]):
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "target_coverage": float(target),
                "n_subjects": int(part["subject"].nunique()),
                "heldout_coverage_mean": float(part["heldout_coverage"].mean()),
                "heldout_coverage_std": float(part["heldout_coverage"].std(ddof=0)),
                "heldout_selective_accuracy_mean": float(part["heldout_selective_accuracy"].mean()),
                "heldout_selective_accuracy_std": float(part["heldout_selective_accuracy"].std(ddof=0)),
                "heldout_risk_mean": float(part["heldout_risk"].mean()),
                "heldout_risk_std": float(part["heldout_risk"].std(ddof=0)),
                "heldout_n_covered_sum": int(part["heldout_n_covered"].sum()),
                "eval_labels_used_for_threshold": False,
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "target_coverage", "method"])


def wilcoxon_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    nonzero = values[~np.isclose(values, 0.0)]
    if nonzero.size == 0:
        return 1.0
    return float(wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox").pvalue)


def paired_stats(subject_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, ds in subject_rows.groupby("dataset"):
        for target, target_df in ds.groupby("target_coverage"):
            for reference in ("baseline", "seed_ensemble"):
                ref = target_df[target_df["method"] == reference][["subject", "heldout_selective_accuracy", "heldout_risk"]]
                if ref.empty:
                    continue
                ref = ref.rename(
                    columns={
                        "heldout_selective_accuracy": "reference_selective_accuracy",
                        "heldout_risk": "reference_risk",
                    }
                )
                for method, method_df in target_df.groupby("method"):
                    if method == reference:
                        continue
                    merged = method_df[["subject", "heldout_selective_accuracy", "heldout_risk"]].merge(
                        ref,
                        on="subject",
                        validate="one_to_one",
                    )
                    if merged.empty:
                        continue
                    acc_delta = merged["heldout_selective_accuracy"] - merged["reference_selective_accuracy"]
                    risk_delta = merged["heldout_risk"] - merged["reference_risk"]
                    rows.append(
                        {
                            "dataset": dataset,
                            "target_coverage": float(target),
                            "method": method,
                            "method_label": METHOD_LABELS.get(method, method),
                            "reference_method": reference,
                            "n_subjects": int(len(merged)),
                            "mean_delta_selective_accuracy": float(acc_delta.mean()),
                            "median_delta_selective_accuracy": float(acc_delta.median()),
                            "wilcoxon_p_selective_accuracy": wilcoxon_p(acc_delta.to_numpy(dtype=float)),
                            "wins_selective_accuracy": int((acc_delta > EPS).sum()),
                            "ties_selective_accuracy": int(np.isclose(acc_delta, 0.0).sum()),
                            "losses_selective_accuracy": int((acc_delta < -EPS).sum()),
                            "mean_delta_risk": float(risk_delta.mean()),
                            "median_delta_risk": float(risk_delta.median()),
                            "wilcoxon_p_risk": wilcoxon_p(risk_delta.to_numpy(dtype=float)),
                            "eval_labels_used_for_threshold": False,
                        }
                    )
    return pd.DataFrame(rows).sort_values(["dataset", "target_coverage", "reference_method", "method"])


def plot_curves(summary_df: pd.DataFrame, out_dir: Path) -> None:
    datasets = [d for d in DATASET_ORDER if d in set(summary_df["dataset"])]
    datasets.extend(sorted(set(summary_df["dataset"]) - set(datasets)))
    if not datasets:
        return
    n_cols = 2
    n_rows = int(np.ceil(len(datasets) / n_cols))
    with plt.rc_context(IEEE_FIGURE_RC):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.16, 3.15 * n_rows), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, dataset in zip(axes.ravel(), datasets):
            ax.axis("on")
            ds = summary_df[summary_df["dataset"] == dataset]
            method_order = [m for m in METHOD_LABELS if m in set(ds["method"])]
            method_order.extend(sorted(set(ds["method"]) - set(method_order)))
            for method in method_order:
                part = ds[ds["method"] == method].sort_values("heldout_coverage_mean")
                if part.empty:
                    continue
                ax.plot(
                    part["heldout_coverage_mean"],
                    part["heldout_risk_mean"],
                    marker="o",
                    linewidth=1.5,
                    markersize=3.5,
                    color=METHOD_COLORS.get(method),
                    label=METHOD_LABELS.get(method, method),
                )
            ax.set_title(dataset)
            ax.set_xlabel("Heldout coverage from validation thresholds")
            ax.set_ylabel("Heldout risk")
            ax.set_xlim(0.0, 1.02)
            ax.set_ylim(bottom=0.0)
            ax.grid(color="0.9", linewidth=0.7)
            ax.set_axisbelow(True)
        handles, labels = axes.ravel()[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, ncol=min(4, len(handles)), loc="upper center", frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.94), pad=0.8)
        for ext in ("pdf", "png"):
            fig.savefig(out_dir / f"risk_coverage_curves.{ext}", dpi=240, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probability-dir", action="append", required=True, type=Path)
    parser.add_argument("--subject-json-dir", action="append", default=[], type=Path)
    parser.add_argument("--out-dir", default=Path("results") / "paper_calibration_stats" / "risk_coverage", type=Path)
    parser.add_argument("--targets", type=float, nargs="+", default=list(TARGET_COVERAGES))
    args = parser.parse_args()

    targets = tuple(float(t) for t in args.targets)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for path in iter_probability_files(args.probability_dir):
        rows.extend(read_probability_file(path, targets, args.subject_json_dir))
    if not rows:
        raise SystemExit("No probability artifacts found. Re-run experiments with --save-probabilities first.")

    subject_df = pd.DataFrame(rows).sort_values(["dataset", "subject", "method", "target_coverage"])
    summary_df = summarize(subject_df)
    paired_df = paired_stats(subject_df)

    subject_df.to_csv(args.out_dir / "risk_coverage_subject.csv", index=False)
    summary_df.to_csv(args.out_dir / "risk_coverage_summary.csv", index=False)
    paired_df.to_csv(args.out_dir / "risk_coverage_paired_stats.csv", index=False)
    plot_curves(summary_df, args.out_dir)


if __name__ == "__main__":
    main()
