#!/usr/bin/env python3
"""Quantify how much held-out metrics move when the compute platform changes.

Bit-identical results across backends are impossible: floating-point addition is
not associative, and cpu/cuda/mps use different kernels and reduction orders.
The scientifically useful question is not "are they identical" but "is the
platform-induced shift small relative to the effect being claimed".

This script answers that by putting three quantities on the same scale:

  1. the platform delta        (same subject, same method, two platforms)
  2. the seed-to-seed spread   (within-platform noise floor)
  3. the claimed effect        (e.g. seed ensemble vs baseline)

If (1) is far below (3), the conclusion is robust to the migration. If (1) is
comparable to (3), the claimed effect is at hardware-noise level and must be
reported as such.

Usage
-----
    python src/scripts/compare_platform_runs.py \
        --reference results/calibration_workflow/validation_only_selection_bci4_2a_subject_metrics.csv \
        --candidate results/calibration_workflow_cuda/validation_only_selection_bci4_2a_subject_metrics.csv \
        --reference-label cpu-arm64 --candidate-label cuda-a10g \
        --out-dir results/paper_calibration_stats/platform_check
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path
from typing import Any

METRICS = ("heldout_accuracy", "heldout_Brier", "heldout_ECE", "heldout_NLL")
KEY_FIELDS = ("dataset", "subject", "experiment_key")


def read_rows(path: Path) -> dict[tuple[str, ...], dict[str, str]]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"no rows in {path}")
    missing = [f for f in KEY_FIELDS if f not in rows[0]]
    if missing:
        raise SystemExit(f"{path} is missing required columns: {missing}")
    return {tuple(str(r[f]) for f in KEY_FIELDS): r for r in rows}


def as_float(row: dict[str, str], col: str) -> float | None:
    raw = row.get(col)
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def paired_deltas(
    ref: dict[tuple[str, ...], dict[str, str]],
    cand: dict[tuple[str, ...], dict[str, str]],
) -> tuple[list[dict[str, Any]], list[tuple[str, ...]], list[tuple[str, ...]]]:
    shared = sorted(set(ref) & set(cand))
    only_ref = sorted(set(ref) - set(cand))
    only_cand = sorted(set(cand) - set(ref))
    out: list[dict[str, Any]] = []
    for key in shared:
        row: dict[str, Any] = dict(zip(KEY_FIELDS, key))
        for metric in METRICS:
            a, b = as_float(ref[key], metric), as_float(cand[key], metric)
            row[f"delta_{metric}"] = None if a is None or b is None else b - a
        out.append(row)
    return out, only_ref, only_cand


def effect_size(
    rows: dict[tuple[str, ...], dict[str, str]],
    method: str,
    baseline: str,
    metric: str,
) -> float | None:
    """Mean paired delta of `method` vs `baseline` within one platform."""
    deltas = []
    for key, row in rows.items():
        dataset, subject, experiment = key
        if experiment != method:
            continue
        base = rows.get((dataset, subject, baseline))
        if base is None:
            continue
        a, b = as_float(base, metric), as_float(row, metric)
        if a is not None and b is not None:
            deltas.append(b - a)
    return st.mean(deltas) if deltas else None


def summarize(deltas: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    vals = [r[f"delta_{metric}"] for r in deltas if r.get(f"delta_{metric}") is not None]
    if not vals:
        return {"metric": metric, "n": 0}
    absvals = [abs(v) for v in vals]
    return {
        "metric": metric,
        "n": len(vals),
        "mean_delta": st.mean(vals),
        "mean_abs_delta": st.mean(absvals),
        "max_abs_delta": max(absvals),
        "std_delta": st.pstdev(vals) if len(vals) > 1 else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", type=Path, required=True, help="Subject-metrics CSV from the reference platform.")
    ap.add_argument("--candidate", type=Path, required=True, help="Subject-metrics CSV from the new platform.")
    ap.add_argument("--reference-label", default="reference")
    ap.add_argument("--candidate-label", default="candidate")
    ap.add_argument("--effect-method", default="seed_ensemble", help="Method whose effect sets the comparison scale.")
    ap.add_argument("--effect-baseline", default="baseline")
    ap.add_argument("--out-dir", type=Path, default=Path("results/paper_calibration_stats/platform_check"))
    args = ap.parse_args()

    ref, cand = read_rows(args.reference), read_rows(args.candidate)
    deltas, only_ref, only_cand = paired_deltas(ref, cand)
    if not deltas:
        raise SystemExit("no shared (dataset, subject, experiment_key) rows between the two files")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    per_row = args.out_dir / "platform_delta_rows.csv"
    cols = list(KEY_FIELDS) + [f"delta_{m}" for m in METRICS]
    with per_row.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols)
        writer.writeheader()
        writer.writerows(deltas)

    summary_rows = []
    for metric in METRICS:
        row = summarize(deltas, metric)
        eff = effect_size(ref, args.effect_method, args.effect_baseline, metric)
        row["reference_effect"] = eff
        # How large is the platform shift relative to the effect being claimed?
        # <0.1 is comfortable; >0.5 means the claim is near hardware noise.
        row["ratio_platform_to_effect"] = (
            None if not eff or not row.get("mean_abs_delta") else row["mean_abs_delta"] / abs(eff)
        )
        summary_rows.append(row)

    summary_csv = args.out_dir / "platform_delta_summary.csv"
    fields = [
        "metric", "n", "mean_delta", "mean_abs_delta", "max_abs_delta",
        "std_delta", "reference_effect", "ratio_platform_to_effect",
    ]
    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)

    meta = {
        "reference": {"label": args.reference_label, "path": str(args.reference), "n_rows": len(ref)},
        "candidate": {"label": args.candidate_label, "path": str(args.candidate), "n_rows": len(cand)},
        "n_paired_rows": len(deltas),
        "rows_only_in_reference": [list(k) for k in only_ref],
        "rows_only_in_candidate": [list(k) for k in only_cand],
        "effect_method": args.effect_method,
        "effect_baseline": args.effect_baseline,
        "summary": summary_rows,
    }
    (args.out_dir / "platform_delta_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"paired rows: {len(deltas)}  ({args.reference_label} -> {args.candidate_label})")
    if only_ref or only_cand:
        print(f"unmatched rows: {len(only_ref)} reference-only, {len(only_cand)} candidate-only")
    print(f"{'metric':18s} {'mean|d|':>10s} {'max|d|':>10s} {'effect':>10s} {'|d|/effect':>11s}")
    for row in summary_rows:
        if not row.get("n"):
            print(f"{row['metric']:18s} {'no data':>10s}")
            continue
        eff = row["reference_effect"]
        ratio = row["ratio_platform_to_effect"]
        print(
            f"{row['metric']:18s} {row['mean_abs_delta']:10.5f} {row['max_abs_delta']:10.5f} "
            f"{(f'{eff:.5f}' if eff is not None else 'n/a'):>10s} "
            f"{(f'{ratio:.3f}' if ratio is not None else 'n/a'):>11s}"
        )
    print(f"\nWrote {summary_csv}")


if __name__ == "__main__":
    main()
