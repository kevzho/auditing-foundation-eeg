#!/usr/bin/env python3
"""Report the full audited grid and correct for how many tests were actually run.

The manuscript reports four methods. The workflow ran eleven. Every intervention
was compared against the same baseline on the same subjects, so the reported
p-values come from a family of tests, not from one. This script makes that
family explicit and applies Holm and Benjamini-Hochberg corrections to it.

It also reports the *attainable* p-value floor. The exact two-sided Wilcoxon
signed-rank test on n non-zero pairs cannot produce a p below 2^(1-n), so for
small subject counts a corrected threshold can be unreachable no matter how
large the effect is. That is a property of the design, not of the result, and
belongs in the paper next to any corrected claim.

Outputs (default results/paper_calibration_stats/multiplicity/):
  aggregate_all_methods.csv   every method, every dataset, held-out means
  multiplicity_tests.csv      every paired test with raw/Holm/BH p-values
  power_ceiling.csv           attainable p floor by subject count
  multiplicity_table.tex      LaTeX table for the manuscript
  multiplicity_summary.json   machine-readable rollup

Usage:
    python src/scripts/make_multiplicity_table.py
    python src/scripts/make_multiplicity_table.py --dataset bci4_2a --alpha 0.05
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

COMPARISON_CSV = Path("results") / "validation_only_selection_dataset_comparison.csv"
DEFAULT_OUT = Path("results") / "paper_calibration_stats" / "multiplicity"

BASELINE = "baseline"
METRICS = ("Brier", "accuracy", "ECE", "NLL")
LOWER_IS_BETTER = {"Brier", "ECE", "NLL"}
EPS = 1e-12

METHOD_LABELS = {
    "baseline": "Baseline",
    "seed_ensemble": "Seed ensemble",
    "augmentation": "Augmentation",
    "teacher_student_mixture": "Teacher/student",
    "architecture": "Architecture variant",
    "brier_loss": "Brier regularization",
    "calibration_checkpoint": "Calibration checkpoint",
    "calibration_grid": "Post-hoc calibration grid",
    "crop_aggregation": "Crop aggregation",
    "stronger_teacher": "Stronger teacher",
    "swa_ema": "SWA/EMA",
    "mdrm_t_ea": "MDRM-T + EA",
}
DATASET_LABELS = {"bci4_2a": "BCI IV-2a", "bci_iiia": "BCI IIIa"}


def wilcoxon_p(values: np.ndarray) -> float:
    """Two-sided exact Wilcoxon. Matches make_calibration_paper_stats.py."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    nonzero = values[~np.isclose(values, 0.0)]
    if nonzero.size == 0:
        return 1.0
    return float(wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox").pvalue)


def attainable_p_floor(n_nonzero: int) -> float:
    """Smallest two-sided exact Wilcoxon p reachable with n non-zero pairs.

    Both tails of the single most extreme rank assignment: 2 / 2^n.
    """
    if n_nonzero < 1:
        return 1.0
    return min(1.0, 2.0 ** (1 - n_nonzero))


def holm(pvals: list[float], alpha: float) -> tuple[list[float], list[bool]]:
    """Holm-Bonferroni step-down. Returns adjusted p-values and reject flags."""
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist(), [bool(p <= alpha) for p in adjusted]


def benjamini_hochberg(pvals: list[float], alpha: float) -> tuple[list[float], list[bool]]:
    """BH step-up FDR control. Returns adjusted p-values and reject flags."""
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m, dtype=float)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        running = min(running, m / (rank + 1) * pvals[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist(), [bool(p <= alpha) for p in adjusted]


def aggregate_table(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (dataset, method), part in df.groupby(["dataset", "experiment_key"]):
        record = {
            "dataset": DATASET_LABELS.get(dataset, dataset),
            "dataset_key": dataset,
            "method": method,
            "method_label": METHOD_LABELS.get(method, method),
            "n_subjects": int(len(part)),
            "n_classes": int(part["n_classes"].iloc[0]),
            "reported_in_manuscript": method in {"baseline", "seed_ensemble", "augmentation", "teacher_student_mixture"},
        }
        for metric in METRICS:
            values = pd.to_numeric(part[f"heldout_{metric}"], errors="coerce").dropna()
            record[f"heldout_{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            record[f"heldout_{metric}_std"] = float(values.std(ddof=0)) if len(values) else np.nan
        records.append(record)
    return pd.DataFrame(records).sort_values(["dataset", "method"]).reset_index(drop=True)


def paired_tests(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """One family of tests per dataset: every non-baseline method x every metric."""
    frames = []
    for dataset, part in df.groupby("dataset"):
        base = part[part["experiment_key"] == BASELINE].set_index("subject")
        if base.empty:
            continue
        records = []
        for method, sub in part.groupby("experiment_key"):
            if method == BASELINE:
                continue
            sub = sub.set_index("subject")
            shared = sorted(set(base.index) & set(sub.index), key=str)
            if not shared:
                continue
            for metric in METRICS:
                col = f"heldout_{metric}"
                a = pd.to_numeric(base.loc[shared, col], errors="coerce")
                b = pd.to_numeric(sub.loc[shared, col], errors="coerce")
                delta = (b - a).dropna()
                if delta.empty:
                    continue
                arr = delta.to_numpy(dtype=float)
                lower = metric in LOWER_IS_BETTER
                wins = int((arr < -EPS).sum()) if lower else int((arr > EPS).sum())
                losses = int((arr > EPS).sum()) if lower else int((arr < -EPS).sum())
                n_nonzero = int(np.count_nonzero(~np.isclose(arr, 0.0)))
                records.append(
                    {
                        "dataset": DATASET_LABELS.get(dataset, dataset),
                        "dataset_key": dataset,
                        "method": method,
                        "method_label": METHOD_LABELS.get(method, method),
                        "metric": metric,
                        "n_subjects": int(len(arr)),
                        "n_nonzero_pairs": n_nonzero,
                        "mean_delta": float(arr.mean()),
                        "median_delta": float(np.median(arr)),
                        "wins": wins,
                        "ties": int(len(arr) - wins - losses),
                        "losses": losses,
                        "p_raw": wilcoxon_p(arr),
                        "attainable_p_floor": attainable_p_floor(n_nonzero),
                        "reported_in_manuscript": method
                        in {"seed_ensemble", "augmentation", "teacher_student_mixture"},
                    }
                )
        if not records:
            continue
        family = pd.DataFrame(records)
        pvals = family["p_raw"].tolist()
        family["n_tests_in_family"] = len(pvals)
        family["bonferroni_threshold"] = alpha / len(pvals)
        # A test whose attainable floor exceeds the Bonferroni threshold cannot
        # reach significance at any effect size. Flag it rather than let a large
        # adjusted p read as a weak effect.
        family["bonferroni_unreachable"] = family["attainable_p_floor"] > family["bonferroni_threshold"]
        holm_p, holm_rej = holm(pvals, alpha)
        bh_p, bh_rej = benjamini_hochberg(pvals, alpha)
        family["p_holm"] = holm_p
        family["reject_holm"] = holm_rej
        family["p_bh"] = bh_p
        family["reject_bh"] = bh_rej
        frames.append(family)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["dataset", "p_raw", "method", "metric"]).reset_index(drop=True)


def power_ceiling_table(max_n: int = 40) -> pd.DataFrame:
    rows = []
    for n in range(3, max_n + 1):
        rows.append({"n_nonzero_pairs": n, "attainable_p_floor": attainable_p_floor(n)})
    return pd.DataFrame(rows)


def fmt_p(value: float) -> str:
    if value >= 0.001:
        return f"{value:.3f}"
    return f"{value:.1e}"


def latex_table(tests: pd.DataFrame, dataset: str, alpha: float) -> str:
    part = tests[tests["dataset"] == dataset].copy()
    if part.empty:
        return ""
    part = part.sort_values("p_raw")
    n_tests = int(part["n_tests_in_family"].iloc[0])
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        f"\\caption{{All {n_tests} paired held-out tests run on {dataset}, corrected for multiplicity. "
        "Negative deltas improve Brier, ECE and NLL; positive deltas improve accuracy. "
        f"No test is significant at $\\alpha={alpha}$ after Holm or Benjamini--Hochberg correction. "
        "The attainable floor column gives the smallest $p$ the exact test can produce at this "
        "subject count.}",
        "\\label{tab:multiplicity}",
        "\\begin{tabular}{llcccccc}",
        "\\toprule",
        "Method & Metric & Mean $\\Delta$ & W/T/L & $p_{\\mathrm{raw}}$ & $p_{\\mathrm{Holm}}$ & "
        "$p_{\\mathrm{BH}}$ & floor \\\\",
        "\\midrule",
    ]
    for row in part.itertuples():
        star = "$^{*}$" if row.reported_in_manuscript else ""
        lines.append(
            f"{row.method_label}{star} & {row.metric} & {row.mean_delta:+.4f} & "
            f"{row.wins}/{row.ties}/{row.losses} & {fmt_p(row.p_raw)} & "
            f"{fmt_p(row.p_holm)} & {fmt_p(row.p_bh)} & {fmt_p(row.attainable_p_floor)} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\\\[2pt]",
        "\\footnotesize $^{*}$ reported in the original manuscript.",
        "\\end{table*}",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comparison-csv", type=Path, default=COMPARISON_CSV)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--dataset", default=None, help="Restrict to one dataset key (e.g. bci4_2a).")
    ap.add_argument("--latex-dataset", default="BCI IV-2a", help="Dataset label for the LaTeX table.")
    args = ap.parse_args()

    if not args.comparison_csv.exists():
        raise SystemExit(f"missing {args.comparison_csv}; run the workflow with --merge-summaries first")
    df = pd.read_csv(args.comparison_csv)
    if args.dataset:
        df = df[df["dataset"] == args.dataset]
        if df.empty:
            raise SystemExit(f"no rows for dataset {args.dataset}")

    unsafe = df[df["eval_labels_used_for_selection"].astype(str).str.lower() == "true"]
    if not unsafe.empty:
        raise SystemExit(f"refusing to report: {len(unsafe)} rows have eval_labels_used_for_selection=true")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    aggregate = aggregate_table(df)
    tests = paired_tests(df, args.alpha)
    ceiling = power_ceiling_table()

    aggregate.to_csv(args.out_dir / "aggregate_all_methods.csv", index=False)
    tests.to_csv(args.out_dir / "multiplicity_tests.csv", index=False)
    ceiling.to_csv(args.out_dir / "power_ceiling.csv", index=False)
    tex = latex_table(tests, args.latex_dataset, args.alpha)
    if tex:
        (args.out_dir / "multiplicity_table.tex").write_text(tex, encoding="utf-8")

    rollup = []
    for dataset, part in tests.groupby("dataset"):
        n_tests = int(part["n_tests_in_family"].iloc[0])
        rollup.append(
            {
                "dataset": dataset,
                "n_methods_run": int(aggregate[aggregate["dataset"] == dataset]["method"].nunique()),
                "n_methods_reported": int(
                    aggregate[(aggregate["dataset"] == dataset) & aggregate["reported_in_manuscript"]]["method"].nunique()
                ),
                "n_tests": n_tests,
                "bonferroni_threshold": float(part["bonferroni_threshold"].iloc[0]),
                "n_raw_significant": int((part["p_raw"] <= args.alpha).sum()),
                "n_holm_significant": int(part["reject_holm"].sum()),
                "n_bh_significant": int(part["reject_bh"].sum()),
                "n_tests_bonferroni_unreachable": int(part["bonferroni_unreachable"].sum()),
                "min_attainable_p": float(part["attainable_p_floor"].min()),
            }
        )
    summary = {"alpha": args.alpha, "source": str(args.comparison_csv), "by_dataset": rollup}
    (args.out_dir / "multiplicity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"{'dataset':14s} {'run':>4s} {'rpt':>4s} {'tests':>6s} {'raw<a':>6s} {'Holm':>5s} {'BH':>4s} {'unreach':>8s}")
    for row in rollup:
        print(
            f"{row['dataset']:14s} {row['n_methods_run']:4d} {row['n_methods_reported']:4d} "
            f"{row['n_tests']:6d} {row['n_raw_significant']:6d} {row['n_holm_significant']:5d} "
            f"{row['n_bh_significant']:4d} {row['n_tests_bonferroni_unreachable']:8d}"
        )
    print(f"\nWrote {args.out_dir}")


if __name__ == "__main__":
    main()
