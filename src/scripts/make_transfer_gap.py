#!/usr/bin/env python3
"""Quantify what validation-locked model selection actually costs at deployment.

The protocol guarantees that no evaluation label informs any choice. It does not
guarantee that the choice was a good one. This script measures the difference.

Definitions
-----------
For a candidate set M and subject s, with a lower-is-better metric (default the
multiclass Brier score):

    v(s) = argmin_{m in M} metric_val(s, m)        validation-selected method
    o(s) = argmin_{m in M} metric_heldout(s, m)    oracle (uses eval labels;
                                                   reported only, never selected on)

    selection regret   R(s)  = metric_heldout(s, v(s)) - metric_heldout(s, o(s))
    normalized regret  R~(s) = R(s) / metric_chance(s)

R(s) >= 0 by construction and is expressed in the metric's own units: it is what
a practitioner pays for choosing on validation rather than knowing the answer.
Normalizing by the chance value makes 2-class and 4-class datasets comparable.

The same quantity for a *fixed* strategy f (always pick one method, never look at
validation) is

    R_f(s) = metric_heldout(s, f) - metric_heldout(s, o(s))

which gives the decisive practical comparison: if mean R exceeds mean R_f for
some fixed f, then validation-based per-subject selection is worse than not
selecting at all.

Also reported: Spearman rho between the validation and held-out method rankings
per subject, and mean |rank shift| normalized by (|M| - 1) so datasets with
different candidate-set sizes can be compared.

Usage
-----
    python src/scripts/make_transfer_gap.py
    python src/scripts/make_transfer_gap.py --metric NLL --min-methods 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_multiplicity_table import benjamini_hochberg, holm, wilcoxon_p  # noqa: E402

COMPARISON_CSV = Path("results") / "validation_only_selection_dataset_comparison.csv"
DEFAULT_OUT = Path("results") / "paper_calibration_stats" / "transfer_gap"
DATASET_LABELS = {"bci4_2a": "BCI IV-2a", "bci_iiia": "BCI IIIa"}
LOWER_IS_BETTER = {"Brier", "ECE", "NLL"}


def chance_value(part: pd.DataFrame, metric: str) -> float:
    """Reference value for normalization. Only Brier has a principled one here."""
    if metric == "Brier" and "chance_Brier" in part:
        return float(part["chance_Brier"].iloc[0])
    if metric == "accuracy" and "chance_accuracy" in part:
        return float(part["chance_accuracy"].iloc[0])
    return float("nan")


def per_subject_rows(df: pd.DataFrame, metric: str, min_methods: int) -> pd.DataFrame:
    val_col, held_col = f"val_{metric}", f"heldout_{metric}"
    lower = metric in LOWER_IS_BETTER
    sign = 1.0 if lower else -1.0
    rows = []
    for (dataset, subject), part in df.groupby(["dataset", "subject"]):
        part = part.dropna(subset=[val_col, held_col])
        if part["experiment_key"].nunique() < min_methods:
            continue
        part = part.drop_duplicates(subset=["experiment_key"])
        val = sign * part[val_col].to_numpy(dtype=float)
        held = sign * part[held_col].to_numpy(dtype=float)
        methods = part["experiment_key"].to_numpy()

        selected_idx = int(np.argmin(val))
        oracle_idx = int(np.argmin(held))
        regret = float(held[selected_idx] - held[oracle_idx])

        # Spearman is undefined when a split has no variance across methods.
        if np.ptp(val) == 0 or np.ptp(held) == 0:
            rho = float("nan")
        else:
            rho = float(spearmanr(val, held).statistic)

        val_rank = pd.Series(val).rank(method="average").to_numpy()
        held_rank = pd.Series(held).rank(method="average").to_numpy()
        n_methods = len(methods)
        row = {
            "dataset": DATASET_LABELS.get(dataset, dataset),
            "dataset_key": dataset,
            "subject": subject,
            "n_methods": n_methods,
            "metric": metric,
            "selected_method": str(methods[selected_idx]),
            "oracle_method": str(methods[oracle_idx]),
            "selection_correct": bool(selected_idx == oracle_idx),
            "heldout_selected": float(sign * held[selected_idx]),
            "heldout_oracle": float(sign * held[oracle_idx]),
            "regret": regret,
            "spearman_rho": rho,
            "mean_abs_rank_shift": float(np.mean(np.abs(held_rank - val_rank))),
            "norm_rank_shift": float(np.mean(np.abs(held_rank - val_rank)) / max(n_methods - 1, 1)),
        }
        chance = chance_value(part, metric)
        row["chance_value"] = chance
        row["norm_regret"] = regret / chance if chance and np.isfinite(chance) else float("nan")
        # Fixed-strategy regrets: what you would have paid by never selecting.
        for method in part["experiment_key"]:
            idx = int(np.where(methods == method)[0][0])
            row[f"regret_fixed__{method}"] = float(held[idx] - held[oracle_idx])
        rows.append(row)
    return pd.DataFrame(rows)


def strategy_comparison(subject_rows: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Validation selection vs every fixed single-method strategy, paired per subject."""
    records = []
    for dataset, part in subject_rows.groupby("dataset"):
        fixed_cols = [c for c in part.columns if c.startswith("regret_fixed__")]
        for col in fixed_cols:
            method = col[len("regret_fixed__") :]
            sub = part.dropna(subset=[col, "regret"])
            if sub.empty:
                continue
            # positive delta => validation selection cost more than the fixed choice
            delta = (sub["regret"] - sub[col]).to_numpy(dtype=float)
            records.append(
                {
                    "dataset": dataset,
                    "fixed_method": method,
                    "n_subjects": int(len(delta)),
                    "mean_regret_validation": float(sub["regret"].mean()),
                    "mean_regret_fixed": float(sub[col].mean()),
                    "mean_delta": float(delta.mean()),
                    "validation_better": int((delta < -1e-12).sum()),
                    "ties": int(np.isclose(delta, 0.0).sum()),
                    "fixed_better": int((delta > 1e-12).sum()),
                    "p_raw": wilcoxon_p(delta),
                }
            )
    if not records:
        return pd.DataFrame()
    out = pd.DataFrame(records)
    frames = []
    for _, part in out.groupby("dataset"):
        part = part.copy()
        ps = part["p_raw"].tolist()
        part["n_tests_in_family"] = len(ps)
        part["p_holm"], _ = holm(ps, alpha)
        part["p_bh"], _ = benjamini_hochberg(ps, alpha)
        frames.append(part)
    return pd.concat(frames, ignore_index=True).sort_values(["dataset", "mean_regret_fixed"])


def dataset_summary(subject_rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for dataset, part in subject_rows.groupby("dataset"):
        rho = part["spearman_rho"].dropna()
        records.append(
            {
                "dataset": dataset,
                "n_subjects": int(len(part)),
                "n_methods": int(part["n_methods"].max()),
                "selection_correct_count": int(part["selection_correct"].sum()),
                "selection_correct_rate": float(part["selection_correct"].mean()),
                "mean_regret": float(part["regret"].mean()),
                "median_regret": float(part["regret"].median()),
                "max_regret": float(part["regret"].max()),
                "mean_norm_regret": float(part["norm_regret"].mean()),
                "mean_spearman_rho": float(rho.mean()) if len(rho) else float("nan"),
                "median_spearman_rho": float(rho.median()) if len(rho) else float("nan"),
                "frac_rho_negative": float((rho < 0).mean()) if len(rho) else float("nan"),
                "mean_norm_rank_shift": float(part["norm_rank_shift"].mean()),
            }
        )
    return pd.DataFrame(records).sort_values("dataset")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comparison-csv", type=Path, default=COMPARISON_CSV)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--metric", default="Brier", choices=["Brier", "accuracy", "ECE", "NLL"])
    ap.add_argument("--min-methods", type=int, default=3, help="Skip subjects with fewer candidates than this.")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    if not args.comparison_csv.exists():
        raise SystemExit(f"missing {args.comparison_csv}")
    df = pd.read_csv(args.comparison_csv)
    unsafe = df[df["eval_labels_used_for_selection"].astype(str).str.lower() == "true"]
    if not unsafe.empty:
        raise SystemExit(f"refusing to report: {len(unsafe)} rows have eval_labels_used_for_selection=true")

    subject_rows = per_subject_rows(df, args.metric, args.min_methods)
    if subject_rows.empty:
        raise SystemExit("no subjects with enough candidate methods")
    summary = dataset_summary(subject_rows)
    strategies = strategy_comparison(subject_rows, args.alpha)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    subject_rows.to_csv(args.out_dir / f"transfer_gap_subject_{args.metric}.csv", index=False)
    summary.to_csv(args.out_dir / f"transfer_gap_summary_{args.metric}.csv", index=False)
    if not strategies.empty:
        strategies.to_csv(args.out_dir / f"transfer_gap_strategies_{args.metric}.csv", index=False)
    (args.out_dir / f"transfer_gap_{args.metric}.json").write_text(
        json.dumps(
            {
                "metric": args.metric,
                "alpha": args.alpha,
                "summary": summary.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"=== transfer gap, metric={args.metric} ===")
    print(
        f"{'dataset':14s} {'n':>2s} {'M':>2s} {'oracle-hit':>10s} {'meanRegret':>11s} "
        f"{'normRegret':>11s} {'rho':>7s} {'rho<0':>6s}"
    )
    for row in summary.itertuples():
        print(
            f"{row.dataset:14s} {row.n_subjects:2d} {row.n_methods:2d} "
            f"{row.selection_correct_count:3d}/{row.n_subjects:<6d} {row.mean_regret:11.5f} "
            f"{row.mean_norm_regret:11.5f} {row.mean_spearman_rho:7.3f} {row.frac_rho_negative:6.2f}"
        )
    if not strategies.empty:
        print("\nvalidation selection vs best fixed strategy (positive delta = selection cost more):")
        for dataset, part in strategies.groupby("dataset"):
            best = part.loc[part["mean_regret_fixed"].idxmin()]
            verdict = "WORSE than fixed" if best.mean_delta > 0 else "better than fixed"
            print(
                f"  {dataset:14s} best fixed = {best.fixed_method:24s} "
                f"regret {best.mean_regret_fixed:.5f} vs selection {best.mean_regret_validation:.5f} "
                f"-> {verdict} (p_raw={best.p_raw:.4f})"
            )
    print(f"\nWrote {args.out_dir}")


if __name__ == "__main__":
    main()
