#!/usr/bin/env python3
"""Measure how far each method's confidence distribution moves between splits.

A validation-selected abstention threshold only transfers if the confidence
distribution it was cut from transfers. This script measures that directly:
for every method it compares the top-label confidence distribution on the
validation split against the held-out split, and reports what a
validation-calibrated coverage target would actually deliver at evaluation time.

Motivating result (BCI IV-2a subject 1):

    baseline          val_conf 0.6082  eval_conf 0.6102  shift +0.0020
    ea_seed_ensemble  val_conf 0.6214  eval_conf 0.6217  shift +0.0003
    mdrm_t_ea         val_conf 0.7260  eval_conf 0.4863  shift -0.2397

MDRM-T's threshold is computed correctly -- its validation coverage lands on
target -- and still collapses on held-out data, while a neural model with the
*same* train-only Euclidean Alignment does not move at all. The failure is
specific to the classifier, not to the preprocessing, and it happens while
MDRM-T is the more accurate model. That is a calibration-transfer failure, not
a performance failure.

Usage:
    python src/scripts/make_confidence_shift_diagnostic.py
    python src/scripts/make_confidence_shift_diagnostic.py --results-root results --include-smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_OUT = Path("results") / "paper_calibration_stats" / "confidence_shift"
COVERAGE_TARGETS = (0.40, 0.60, 0.80, 0.90)


def iter_result_dirs(root: Path, include_smoke: bool) -> list[Path]:
    dirs = [p for p in root.glob("calibration_workflow*") if p.is_dir()]
    if not include_smoke:
        dirs = [p for p in dirs if "_smoke" not in p.name]
    return sorted(dirs)


def threshold_for_target(confidence: np.ndarray, target: float) -> float:
    """Confidence cut that covers `target` fraction of the split.

    Same convention as make_risk_coverage_results.threshold_for_target.
    """
    if confidence.size == 0:
        return float("inf")
    kth = int(np.ceil((1.0 - float(target)) * confidence.size))
    ordered = np.sort(confidence)
    if kth <= 0:
        return float(ordered[0])
    if kth >= confidence.size:
        return float(ordered[-1])
    return float(ordered[kth])


def rows_from_probabilities(path: Path, result_dir: Path) -> list[dict[str, Any]]:
    loaded = np.load(path, allow_pickle=True)
    if "metadata_json" not in loaded.files:
        return []
    meta = json.loads(str(loaded["metadata_json"]))
    y_eval = loaded["y_eval"] if "y_eval" in loaded.files else None
    out: list[dict[str, Any]] = []
    for key in loaded.files:
        if not key.endswith("__val_proba"):
            continue
        method = key[: -len("__val_proba")]
        eval_key = f"{method}__eval_proba"
        if eval_key not in loaded.files:
            continue
        val_conf = np.asarray(loaded[key], dtype=float).max(axis=1)
        eval_conf = np.asarray(loaded[eval_key], dtype=float).max(axis=1)
        if val_conf.size == 0 or eval_conf.size == 0:
            continue
        row: dict[str, Any] = {
            "result_dir": result_dir.name,
            "dataset": meta.get("dataset", result_dir.name),
            "subject": meta.get("subject"),
            "method": method,
            "n_val": int(val_conf.size),
            "n_eval": int(eval_conf.size),
            "val_conf_mean": float(val_conf.mean()),
            "eval_conf_mean": float(eval_conf.mean()),
            "conf_shift": float(eval_conf.mean() - val_conf.mean()),
            "val_conf_std": float(val_conf.std(ddof=0)),
            "eval_conf_std": float(eval_conf.std(ddof=0)),
            "val_conf_median": float(np.median(val_conf)),
            "eval_conf_median": float(np.median(eval_conf)),
        }
        if y_eval is not None:
            pred = np.asarray(loaded[eval_key], dtype=float).argmax(axis=1)
            classes = loaded["classes"] if "classes" in loaded.files else None
            truth = np.asarray(y_eval)
            if classes is not None:
                pred = np.asarray(classes)[pred]
            row["eval_accuracy_all"] = float(np.mean(pred == truth))
        # What a validation-calibrated coverage target actually delivers.
        for target in COVERAGE_TARGETS:
            thr = threshold_for_target(val_conf, target)
            row[f"val_coverage@{target:g}"] = float((val_conf >= thr).mean())
            row[f"eval_coverage@{target:g}"] = float((eval_conf >= thr).mean())
            row[f"coverage_gap@{target:g}"] = float((eval_conf >= thr).mean() - target)
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", type=Path, default=Path("results"))
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--include-smoke", action="store_true", help="Include *_smoke result directories.")
    args = ap.parse_args()

    rows: list[dict[str, Any]] = []
    for result_dir in iter_result_dirs(args.results_root, args.include_smoke):
        prob_dir = result_dir / "probabilities"
        if not prob_dir.is_dir():
            continue
        for path in sorted(prob_dir.glob("*_probabilities.npz")):
            rows.extend(rows_from_probabilities(path, result_dir))

    if not rows:
        raise SystemExit("no probability files found; run the workflow first")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    subject = pd.DataFrame(rows).sort_values(["dataset", "method", "subject"])
    subject.to_csv(args.out_dir / "confidence_shift_subject.csv", index=False)

    agg_cols = ["conf_shift", "val_conf_mean", "eval_conf_mean"] + [
        f"eval_coverage@{t:g}" for t in COVERAGE_TARGETS
    ]
    summary = (
        subject.groupby(["dataset", "method"])
        .agg(n_subjects=("subject", "nunique"), **{c: (c, "mean") for c in agg_cols})
        .reset_index()
        .sort_values(["dataset", "conf_shift"])
    )
    summary.to_csv(args.out_dir / "confidence_shift_summary.csv", index=False)

    print(f"{'dataset':14s} {'method':22s} {'n':>2s} {'val':>7s} {'eval':>7s} {'shift':>8s} {'cov@0.4':>8s}")
    for row in summary.itertuples():
        print(
            f"{str(row.dataset):14s} {str(row.method):22s} {row.n_subjects:2d} "
            f"{row.val_conf_mean:7.4f} {row.eval_conf_mean:7.4f} {row.conf_shift:+8.4f} "
            f"{getattr(row, '_7'):8.3f}"
        )
    print(f"\nWrote {args.out_dir}")


if __name__ == "__main__":
    main()
