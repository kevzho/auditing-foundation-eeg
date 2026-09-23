#!/usr/bin/env python3
"""Readout-only statistics and figure generation for the CalibMI paper."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


METHOD_LABELS = {
    "seed_ensemble": "Seed ensemble",
    "augmentation": "Augmentation",
    "teacher_student_mixture": "Teacher/student",
}

EXTERNAL_DATASETS = ["BCI IIIa", "BNCI2014_001", "BNCI2014_004"]
STATS_METRICS = ["Brier", "accuracy", "ECE", "NLL"]
PAPER_TABLE_METRICS = ["Brier", "accuracy"]
DATASET_DISPLAY_LABELS = {
    "BCI IV-2a": "BCI IV-2a",
    "BCI IIIa": "BCI IIIa",
    "BNCI2014_001": "BNCI 2014-001",
    "BNCI2014_004": "BNCI 2014-004",
}

IEEE_FIGURE_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def _ieee_rc(**overrides: object) -> dict[str, object]:
    rc = IEEE_FIGURE_RC.copy()
    rc.update(overrides)
    return rc


def _metric_columns(metric: str) -> tuple[str, str]:
    return f"heldout_{metric}", f"delta_{metric}"


def _signed_delta(df: pd.DataFrame, method: str, metric: str) -> pd.DataFrame:
    value_col, delta_col = _metric_columns(metric)
    baseline = (
        df[df["experiment_key"] == "baseline"][["subject", value_col]]
        .rename(columns={value_col: "baseline"})
        .copy()
    )
    candidate = (
        df[df["experiment_key"] == method][["subject", value_col]]
        .rename(columns={value_col: "candidate"})
        .copy()
    )
    merged = candidate.merge(baseline, on="subject", validate="one_to_one")
    merged["experiment_key"] = method
    merged[delta_col] = merged["candidate"] - merged["baseline"]
    return merged[["subject", "experiment_key", "baseline", "candidate", delta_col]]


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True)
    means = draws.mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _wilcoxon_p(values: np.ndarray) -> float:
    nonzero = values[~np.isclose(values, 0.0)]
    if len(nonzero) == 0:
        return 1.0
    return float(wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox").pvalue)


def _summarize(
    df: pd.DataFrame,
    dataset: str,
    method: str,
    metric: str,
    rng: np.random.Generator,
    n_boot: int,
) -> dict[str, object]:
    deltas = _signed_delta(df, method, metric)
    delta_col = f"delta_{metric}"
    values = deltas[delta_col].to_numpy(dtype=float)
    ci_low, ci_high = _bootstrap_ci(values, rng, n_boot)
    lower_is_better = metric in {"Brier", "ECE", "NLL"}
    if lower_is_better:
        wins = int((values < -1e-12).sum())
        losses = int((values > 1e-12).sum())
    else:
        wins = int((values > 1e-12).sum())
        losses = int((values < -1e-12).sum())
    ties = int(len(values) - wins - losses)
    return {
        "dataset": dataset,
        "method": method,
        "method_label": METHOD_LABELS[method],
        "metric": metric,
        "n": len(values),
        "mean_delta": float(values.mean()),
        "median_delta": float(np.median(values)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "wilcoxon_p": _wilcoxon_p(values),
        "wins": wins,
        "ties": ties,
        "losses": losses,
    }


def _fmt_num(x: float, digits: int = 3) -> str:
    if abs(x) < 0.5 * 10 ** (-digits):
        x = 0.0
    return f"{x:.{digits}f}"


def _fmt_p(x: float) -> str:
    if x < 0.001:
        return "$<0.001$"
    return f"{x:.3f}"


def _tex_escape(text: object) -> str:
    return str(text).replace("_", r"\_")


def _write_tex_table(stats: pd.DataFrame, out_path: Path) -> None:
    rows = []
    for row in stats.itertuples(index=False):
        rows.append(
            " & ".join(
                [
                    _tex_escape(row.dataset),
                    _tex_escape(row.method_label),
                    _tex_escape(row.metric),
                    str(row.n),
                    _fmt_num(row.mean_delta, 4 if row.metric == "Brier" else 3),
                    f"[{_fmt_num(row.ci_low, 4 if row.metric == 'Brier' else 3)}, "
                    f"{_fmt_num(row.ci_high, 4 if row.metric == 'Brier' else 3)}]",
                    _fmt_p(row.wilcoxon_p),
                    f"{row.wins}/{row.ties}/{row.losses}",
                ]
            )
            + r" \\"
        )
    table = "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Paired held-out deltas versus baseline. For Brier, negative deltas indicate improvement; for accuracy, positive deltas indicate improvement. CIs are 95\% bootstrap intervals over subjects; W/T/L reports win/tie/loss counts.}",
            r"\label{tab:paired-stats}",
            r"\begin{tabular}{llclcccc}",
            r"\toprule",
            r"Dataset & Method & Metric & $n$ & Mean $\Delta$ & 95\% CI & Wilcoxon $p$ & W/T/L \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    out_path.write_text(table)


def _plot_seed_brier_deltas(deltas: pd.DataFrame, out_dir: Path) -> None:
    datasets = ["BCI IV-2a", "BCI IIIa", "BNCI2014_001", "BNCI2014_004"]
    colors = {
        "BCI IV-2a": "#1b9e77",
        "BCI IIIa": "#7570b3",
        "BNCI2014_001": "#d95f02",
        "BNCI2014_004": "#2c7fb8",
    }
    with plt.rc_context(
        _ieee_rc(
            **{
                "font.size": 11,
                "axes.labelsize": 12,
                "axes.titlesize": 12.8,
                "xtick.labelsize": 10.8,
                "ytick.labelsize": 10.8,
            }
        )
    ):
        fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.45), sharex=True)
        for ax, dataset in zip(axes.ravel(), datasets):
            part = deltas[
                (deltas["dataset"] == dataset)
                & (deltas["experiment_key"] == "seed_ensemble")
                & (deltas["metric"] == "Brier")
            ].copy()
            part = part.sort_values("subject")
            y = np.arange(len(part))
            ax.axvline(0, color="0.25", linewidth=0.9)
            ax.barh(y, part["delta"].to_numpy(), color=colors[dataset], alpha=0.88)
            ax.set_yticks(y)
            ax.set_yticklabels(part["subject"].astype(str))
            ax.invert_yaxis()
            ax.set_title(dataset)
            ax.set_xlabel(r"$\Delta$ Brier vs. baseline")
            ax.grid(axis="x", color="0.88", linewidth=0.7)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        axes[0, 0].set_ylabel("Subject")
        axes[1, 0].set_ylabel("Subject")
        fig.subplots_adjust(left=0.095, right=0.995, bottom=0.12, top=0.925, hspace=0.48, wspace=0.19)
        for ext in ("pdf", "png"):
            fig.savefig(out_dir / f"seed_brier_subject_deltas.{ext}", dpi=240, bbox_inches="tight")
        plt.close(fig)


def _plot_aggregate_accuracy_brier(combined: pd.DataFrame, out_dir: Path) -> None:
    methods = ["baseline", "seed_ensemble", "augmentation", "teacher_student_mixture"]
    method_labels = {
        "baseline": "Baseline",
        "seed_ensemble": "Seed ensemble",
        "augmentation": "Augmentation",
        "teacher_student_mixture": "Teacher/student",
    }
    datasets = ["BCI IV-2a", "BCI IIIa", "BNCI2014_001", "BNCI2014_004"]
    dataset_labels = ["BCI IV-2a", "BCI IIIa", "BNCI\n2014-001", "BNCI\n2014-004"]
    colors = {
        "baseline": "#565656",
        "seed_ensemble": "#1b9e77",
        "augmentation": "#d9a300",
        "teacher_student_mixture": "#807dba",
    }
    with plt.rc_context(
        _ieee_rc(
            **{
                "font.size": 11,
                "axes.labelsize": 11.5,
                "axes.titlesize": 12,
                "xtick.labelsize": 10.5,
                "ytick.labelsize": 10.5,
                "legend.fontsize": 10.5,
            }
        )
    ):
        fig, axes = plt.subplots(2, 1, figsize=(7.16, 4.35), sharex=True)
        x = np.arange(len(datasets))
        width = 0.18
        for i, method in enumerate(methods):
            offsets = x + (i - 1.5) * width
            rows = []
            for dataset in datasets:
                part = combined[(combined["dataset"] == dataset) & (combined["experiment_key"] == method)]
                rows.append(part.iloc[0] if len(part) else None)
            acc = [float(row["heldout_accuracy_mean"]) if row is not None else np.nan for row in rows]
            brier = [float(row["heldout_Brier_mean"]) if row is not None else np.nan for row in rows]
            bar_kwargs = {
                "width": width,
                "label": method_labels[method],
                "color": colors[method],
                "alpha": 0.94,
                "edgecolor": "white",
                "linewidth": 0.35,
            }
            axes[0].bar(offsets, acc, **bar_kwargs)
            axes[1].bar(offsets, brier, **bar_kwargs)
        axes[0].set_ylabel("Accuracy")
        axes[1].set_ylabel("Brier score")
        axes[0].set_title("(a) Held-out accuracy", loc="left", pad=7)
        axes[1].set_title("(b) Held-out Brier score", loc="left", pad=7)
        axes[0].set_ylim(0.0, 0.86)
        axes[1].set_ylim(0.0, 0.46)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(dataset_labels)
        for ax in axes:
            ax.grid(axis="y", color="0.88", linewidth=0.55)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            ncol=4,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            frameon=False,
            columnspacing=0.95,
            handlelength=1.25,
            handletextpad=0.4,
            borderaxespad=0.0,
        )
        fig.subplots_adjust(left=0.095, right=0.995, bottom=0.16, top=0.79, hspace=0.46)
        for ext in ("pdf", "png"):
            fig.savefig(out_dir / f"aggregate_heldout_accuracy_brier.{ext}", dpi=300)
        plt.close(fig)


def _plot_aggregate_ece_nll(combined: pd.DataFrame, out_dir: Path) -> None:
    methods = ["baseline", "seed_ensemble", "augmentation", "teacher_student_mixture"]
    method_labels = {
        "baseline": "Baseline",
        "seed_ensemble": "Seed ensemble",
        "augmentation": "Augmentation",
        "teacher_student_mixture": "Teacher/student",
    }
    datasets = ["BCI IV-2a", "BCI IIIa", "BNCI2014_001", "BNCI2014_004"]
    dataset_labels = ["BCI IV-2a", "BCI IIIa", "BNCI\n2014-001", "BNCI\n2014-004"]
    colors = {
        "baseline": "#565656",
        "seed_ensemble": "#1b9e77",
        "augmentation": "#d9a300",
        "teacher_student_mixture": "#807dba",
    }
    with plt.rc_context(
        _ieee_rc(
            **{
            "font.size": 9.5,
            "axes.labelsize": 10,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            }
        )
    ):
        fig, axes = plt.subplots(2, 1, figsize=(7.16, 4.05), sharex=True)
        x = np.arange(len(datasets))
        width = 0.18
        for i, method in enumerate(methods):
            offsets = x + (i - 1.5) * width
            rows = []
            for dataset in datasets:
                part = combined[(combined["dataset"] == dataset) & (combined["experiment_key"] == method)]
                rows.append(part.iloc[0] if len(part) else None)
            ece = [float(row["heldout_ECE_mean"]) if row is not None else np.nan for row in rows]
            nll = [float(row["heldout_NLL_mean"]) if row is not None else np.nan for row in rows]
            bar_kwargs = {
                "width": width,
                "label": method_labels[method],
                "color": colors[method],
                "alpha": 0.94,
                "edgecolor": "white",
                "linewidth": 0.35,
            }
            axes[0].bar(offsets, ece, **bar_kwargs)
            axes[1].bar(offsets, nll, **bar_kwargs)
        axes[0].set_ylabel("ECE")
        axes[1].set_ylabel("NLL")
        axes[0].set_title("(a) Held-out expected calibration error", loc="left", pad=7)
        axes[1].set_title("(b) Held-out negative log-likelihood", loc="left", pad=7)
        axes[0].set_ylim(0.0, 0.22)
        axes[1].set_ylim(0.0, 1.25)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(dataset_labels)
        for ax in axes:
            ax.grid(axis="y", color="0.88", linewidth=0.55)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            ncol=4,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            frameon=False,
            columnspacing=0.95,
            handlelength=1.25,
            handletextpad=0.4,
            borderaxespad=0.0,
        )
        fig.subplots_adjust(left=0.085, right=0.995, bottom=0.155, top=0.80, hspace=0.45)
        for ext in ("pdf", "png"):
            fig.savefig(out_dir / f"aggregate_heldout_ece_nll.{ext}", dpi=300)
        plt.close(fig)


def _plot_validation_heldout_gap(combined_subjects: pd.DataFrame, out_dir: Path) -> None:
    methods = ["baseline", "seed_ensemble", "augmentation", "teacher_student_mixture"]
    method_labels = {
        "baseline": "Baseline",
        "seed_ensemble": "Seed ensemble",
        "augmentation": "Augmentation",
        "teacher_student_mixture": "Teacher/student",
    }
    datasets = ["BCI IV-2a", "BCI IIIa", "BNCI2014_001", "BNCI2014_004"]
    dataset_labels = ["BCI IV-2a", "BCI IIIa", "BNCI\n2014-001", "BNCI\n2014-004"]
    colors = {
        "baseline": "#565656",
        "seed_ensemble": "#1b9e77",
        "augmentation": "#d9a300",
        "teacher_student_mixture": "#807dba",
    }
    rows = []
    for dataset in datasets:
        for method in methods:
            part = combined_subjects[(combined_subjects["dataset"] == dataset) & (combined_subjects["experiment_key"] == method)]
            if part.empty:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "experiment_key": method,
                    "accuracy_gap": float((part["heldout_accuracy"] - part["val_accuracy"]).mean()),
                    "brier_gap": float((part["heldout_Brier"] - part["val_Brier"]).mean()),
                }
            )
    gaps = pd.DataFrame(rows)
    gaps.to_csv(out_dir / "validation_heldout_gaps.csv", index=False)
    with plt.rc_context(
        _ieee_rc(
            **{
            "font.size": 9.5,
            "axes.labelsize": 10,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            }
        )
    ):
        fig, axes = plt.subplots(2, 1, figsize=(7.16, 4.05), sharex=True)
        x = np.arange(len(datasets))
        width = 0.18
        for i, method in enumerate(methods):
            offsets = x + (i - 1.5) * width
            rows = []
            for dataset in datasets:
                part = gaps[(gaps["dataset"] == dataset) & (gaps["experiment_key"] == method)]
                rows.append(part.iloc[0] if len(part) else None)
            acc_gap = [float(row["accuracy_gap"]) if row is not None else np.nan for row in rows]
            brier_gap = [float(row["brier_gap"]) if row is not None else np.nan for row in rows]
            bar_kwargs = {
                "width": width,
                "label": method_labels[method],
                "color": colors[method],
                "alpha": 0.94,
                "edgecolor": "white",
                "linewidth": 0.35,
            }
            axes[0].bar(offsets, acc_gap, **bar_kwargs)
            axes[1].bar(offsets, brier_gap, **bar_kwargs)
        axes[0].axhline(0, color="0.2", linewidth=0.75)
        axes[1].axhline(0, color="0.2", linewidth=0.75)
        axes[0].set_ylabel(r"$\Delta$ accuracy")
        axes[1].set_ylabel(r"$\Delta$ Brier")
        axes[0].set_title("(a) Held-out minus validation accuracy", loc="left", pad=7)
        axes[1].set_title("(b) Held-out minus validation Brier", loc="left", pad=7)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(dataset_labels)
        for ax in axes:
            ax.grid(axis="y", color="0.88", linewidth=0.55)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            ncol=4,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            frameon=False,
            columnspacing=0.95,
            handlelength=1.25,
            handletextpad=0.4,
            borderaxespad=0.0,
        )
        fig.subplots_adjust(left=0.095, right=0.995, bottom=0.155, top=0.80, hspace=0.45)
        for ext in ("pdf", "png"):
            fig.savefig(out_dir / f"validation_heldout_gap.{ext}", dpi=300)
        plt.close(fig)


def _plot_seed_effect_ci(stats: pd.DataFrame, out_dir: Path) -> None:
    part = stats[
        (stats["method"] == "seed_ensemble")
        & (stats["metric"].isin(["Brier", "accuracy", "ECE", "NLL"]))
    ].copy()
    metric_order = {"Brier": 0, "accuracy": 1, "ECE": 2, "NLL": 3}
    dataset_order = {"BCI IV-2a": 0, "BCI IIIa": 1, "BNCI2014_001": 2, "BNCI2014_004": 3}
    part["order"] = part["dataset"].map(dataset_order) * 10 + part["metric"].map(metric_order)
    part = part.sort_values("order")
    labels = [f"{DATASET_DISPLAY_LABELS.get(row.dataset, row.dataset)} / {row.metric}" for row in part.itertuples(index=False)]
    values = part["mean_delta"].to_numpy(dtype=float)
    lows = part["ci_low"].to_numpy(dtype=float)
    highs = part["ci_high"].to_numpy(dtype=float)
    xerr = np.vstack([values - lows, highs - values])
    colors = ["#1b9e77" if metric == "accuracy" else "#d95f02" for metric in part["metric"]]
    with plt.rc_context(
        _ieee_rc(
            **{
            "font.size": 8.8,
            "axes.labelsize": 10,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.4,
            }
        )
    ):
        fig, ax = plt.subplots(figsize=(7.16, 5.35))
        y = np.arange(len(part))
        ax.axvline(0, color="0.25", linewidth=0.8)
        ax.errorbar(values, y, xerr=xerr, fmt="none", ecolor="0.35", elinewidth=0.8, capsize=2.2)
        ax.scatter(values, y, s=18, c=colors, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel(r"Mean paired $\Delta$ vs. baseline (95\% bootstrap CI)")
        ax.set_title("Seed ensemble paired effects across datasets and metrics", loc="left", pad=6)
        ax.grid(axis="x", color="0.88", linewidth=0.55)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.subplots_adjust(left=0.37, right=0.99, bottom=0.12, top=0.91)
        for ext in ("pdf", "png"):
            fig.savefig(out_dir / f"seed_ensemble_effect_ci.{ext}", dpi=300)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bci4-metrics", required=True, type=Path)
    parser.add_argument("--external-comparison", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--bootstrap", default=10000, type=int)
    parser.add_argument("--seed", default=20260630, type=int)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    bci4 = pd.read_csv(args.bci4_metrics)
    bci4["dataset"] = "BCI IV-2a"
    external = pd.read_csv(args.external_comparison)
    external = external[external["dataset"].isin(["BNCI2014_001", "BNCI2014_004", "bci_iiia"])].copy()
    external["dataset"] = external["dataset"].replace({"bci_iiia": "BCI IIIa"})
    combined = pd.concat([bci4, external], ignore_index=True)

    if bci4["eval_labels_used_for_selection"].astype(str).str.lower().eq("true").any():
        raise SystemExit("BCI IV-2a contains eval-label-selected rows.")
    if external["eval_labels_used_for_selection"].astype(str).str.lower().eq("true").any():
        raise SystemExit("External comparison contains eval-label-selected rows.")

    stats_rows = []
    delta_rows = []
    datasets = [
        ("BCI IV-2a", bci4),
        *[(dataset, external[external["dataset"] == dataset]) for dataset in EXTERNAL_DATASETS],
    ]
    for dataset_name, df in datasets:
        for method in METHOD_LABELS:
            if method not in set(df["experiment_key"]):
                continue
            for metric in STATS_METRICS:
                stats_rows.append(_summarize(df, dataset_name, method, metric, rng, args.bootstrap))
                deltas = _signed_delta(df, method, metric)
                for row in deltas.itertuples(index=False):
                    delta_rows.append(
                        {
                            "dataset": dataset_name,
                            "subject": row.subject,
                            "experiment_key": method,
                            "method_label": METHOD_LABELS[method],
                            "metric": metric,
                            "baseline": row.baseline,
                            "candidate": row.candidate,
                            "delta": getattr(row, f"delta_{metric}"),
                        }
                    )

    stats = pd.DataFrame(stats_rows)
    deltas = pd.DataFrame(delta_rows)
    stats.to_csv(args.out_dir / "paired_subject_stats.csv", index=False)
    deltas.to_csv(args.out_dir / "paired_subject_deltas.csv", index=False)

    paper_stats = stats[
        (stats["method"] == "seed_ensemble")
        & (stats["metric"].isin(PAPER_TABLE_METRICS))
        & (stats["dataset"].isin(["BCI IV-2a", *EXTERNAL_DATASETS]))
    ].copy()
    _write_tex_table(paper_stats, args.out_dir / "paired_stats_table.tex")
    _plot_seed_brier_deltas(deltas, args.out_dir)
    aggregate = (
        combined.groupby(["dataset", "experiment_key"], as_index=False)
        .agg(
            heldout_accuracy_mean=("heldout_accuracy", "mean"),
            heldout_Brier_mean=("heldout_Brier", "mean"),
            heldout_ECE_mean=("heldout_ECE", "mean"),
            heldout_NLL_mean=("heldout_NLL", "mean"),
        )
    )
    _plot_aggregate_accuracy_brier(aggregate, args.out_dir)
    _plot_aggregate_ece_nll(aggregate, args.out_dir)
    _plot_validation_heldout_gap(combined, args.out_dir)
    _plot_seed_effect_ci(stats, args.out_dir)


if __name__ == "__main__":
    main()
