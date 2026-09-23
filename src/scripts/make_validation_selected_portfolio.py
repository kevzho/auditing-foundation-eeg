#!/usr/bin/env python3
"""Build readout-only validation-selected portfolio artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


PRIMARY_CANDIDATES = (
    "baseline",
    "seed_ensemble",
    "augmentation",
    "teacher_student_mixture",
    "mdrm_t_ea",
)
OPTIONAL_CANDIDATES = (
    "brier_loss",
    "crop_aggregation",
    "swa_ema",
)
SELECTION_RULE = "lowest_validation_Brier"
TIE_BREAKS = ("highest_validation_accuracy", "lowest_validation_NLL", "method_name")
EPS = 1e-12


def dataset_label(payload: dict[str, Any]) -> str:
    dataset = str(payload.get("dataset", "unknown"))
    if dataset == "bci4_2a":
        return "BCI IV-2a"
    if dataset == "bci_iiia":
        return "BCI IIIa"
    return dataset


def safe_subject_id(subject: object) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(subject))


def iter_result_dirs(results_root: Path, include_smoke: bool) -> list[Path]:
    dirs = [path for path in results_root.glob("calibration_workflow*") if path.is_dir()]
    if not include_smoke:
        dirs = [path for path in dirs if not path.name.endswith("_smoke") and "_smoke" not in path.name]
    return sorted(dirs)


def iter_subject_jsons(result_dir: Path) -> list[Path]:
    paths = []
    for path in sorted(result_dir.glob("validation_only_selection_*_subject*.json")):
        name = path.name
        if "summary" in name or "protocol_inspection" in name:
            continue
        if name.startswith("validation_only_selection_subject"):
            continue
        paths.append(path)
    return paths


def metric(experiment: dict[str, Any], split: str, name: str) -> float:
    return float(experiment[f"{split}_metrics"][name])


def candidate_records(payload: dict[str, Any], include_optional: bool) -> list[dict[str, Any]]:
    experiments = payload.get("experiments", {})
    candidate_names = list(PRIMARY_CANDIDATES)
    if include_optional:
        candidate_names.extend(OPTIONAL_CANDIDATES)
    records = []
    for name in candidate_names:
        experiment = experiments.get(name)
        if not experiment:
            continue
        if experiment.get("selection_audit", {}).get("eval_labels_used_for_selection") is not False:
            raise ValueError(f"{payload.get('dataset')} subject {payload.get('subject')} {name}: unsafe selection audit")
        try:
            records.append(
                {
                    "method": name,
                    "validation_Brier": metric(experiment, "validation", "Brier"),
                    "validation_accuracy": metric(experiment, "validation", "accuracy"),
                    "validation_NLL": metric(experiment, "validation", "NLL"),
                    "experiment": experiment,
                }
            )
        except KeyError as exc:
            raise ValueError(f"{payload.get('dataset')} subject {payload.get('subject')} {name}: missing metric {exc}") from exc
    if not records:
        raise ValueError(f"{payload.get('dataset')} subject {payload.get('subject')}: no portfolio candidates found")
    return records


def select_candidate(records: list[dict[str, Any]]) -> dict[str, Any]:
    return min(
        records,
        key=lambda row: (
            float(row["validation_Brier"]),
            -float(row["validation_accuracy"]),
            float(row["validation_NLL"]),
            str(row["method"]),
        ),
    )


def portfolio_row(payload: dict[str, Any], source_json: Path, include_optional: bool) -> dict[str, Any]:
    records = candidate_records(payload, include_optional=include_optional)
    selected = select_candidate(records)
    selected_exp = selected["experiment"]
    val = selected_exp["validation_metrics"]
    held = selected_exp["heldout_metrics"]
    candidate_set = [row["method"] for row in records]
    return {
        "dataset": dataset_label(payload),
        "dataset_key": payload.get("dataset"),
        "dataset_label": payload.get("dataset_label", payload.get("dataset")),
        "subject": payload.get("subject"),
        "n_classes": payload.get("n_classes"),
        "experiment_key": "validation_selected_portfolio",
        "experiment": "validation_selected_portfolio",
        "selected_method": selected["method"],
        "candidate_set": json.dumps(candidate_set),
        "selected_by": SELECTION_RULE,
        "tie_breaks": json.dumps(list(TIE_BREAKS)),
        "val_accuracy": val.get("accuracy"),
        "val_Brier": val.get("Brier"),
        "val_ECE": val.get("ECE"),
        "val_NLL": val.get("NLL"),
        "heldout_accuracy": held.get("accuracy"),
        "heldout_Brier": held.get("Brier"),
        "heldout_ECE": held.get("ECE"),
        "heldout_NLL": held.get("NLL"),
        "chance_accuracy": payload.get("chance_accuracy", val.get("chance_accuracy")),
        "chance_Brier": payload.get("chance_Brier", val.get("chance_Brier")),
        "selection_split": selected_exp.get("selection_audit", {}).get(
            "selection_split",
            payload.get("validation_discipline", {}).get("selection_split"),
        ),
        "heldout_split": selected_exp.get("selection_audit", {}).get(
            "heldout_split",
            payload.get("validation_discipline", {}).get("heldout_split"),
        ),
        "eval_labels_used_for_selection": False,
        "source_json": str(source_json),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_portfolio_probabilities(result_dir: Path, rows: list[dict[str, Any]], out_dir: Path) -> None:
    probability_dir = result_dir / "probabilities"
    if not probability_dir.exists():
        return
    out_probability_dir = out_dir / "probabilities"
    row_by_subject = {str(row["subject"]): row for row in rows}
    for src in sorted(probability_dir.glob("*_probabilities.npz")):
        loaded = np.load(src)
        metadata = json.loads(str(loaded["metadata_json"]))
        subject = str(metadata.get("subject", ""))
        row = row_by_subject.get(subject)
        if row is None:
            continue
        selected_method = str(row["selected_method"])
        val_key = f"{selected_method}__val_proba"
        eval_key = f"{selected_method}__eval_proba"
        if val_key not in loaded.files or eval_key not in loaded.files:
            continue
        out_probability_dir.mkdir(parents=True, exist_ok=True)
        metadata["experiments"] = {
            "validation_selected_portfolio": {
                "experiment": "validation_selected_portfolio",
                "selected_method": selected_method,
                "candidate_set": json.loads(str(row["candidate_set"])),
                "selected_by": row["selected_by"],
                "tie_breaks": json.loads(str(row["tie_breaks"])),
                "validation_metrics": {
                    "accuracy": row["val_accuracy"],
                    "Brier": row["val_Brier"],
                    "ECE": row["val_ECE"],
                    "NLL": row["val_NLL"],
                },
                "heldout_metrics": {
                    "accuracy": row["heldout_accuracy"],
                    "Brier": row["heldout_Brier"],
                    "ECE": row["heldout_ECE"],
                    "NLL": row["heldout_NLL"],
                },
                "selection_audit": {
                    "selection_split": row["selection_split"],
                    "heldout_split": row["heldout_split"],
                    "eval_labels_used_for_selection": False,
                },
            }
        }
        arrays: dict[str, Any] = {
            "y_val": loaded["y_val"],
            "y_eval": loaded["y_eval"],
            "classes": loaded["classes"],
            "metadata_json": np.asarray(json.dumps(metadata)),
            "validation_selected_portfolio__val_proba": loaded[val_key],
            "validation_selected_portfolio__eval_proba": loaded[eval_key],
        }
        np.savez_compressed(out_probability_dir / src.name, **arrays)


def summarize(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    records = []
    for (dataset, method), part in df.groupby(["dataset", "selected_method"]):
        record: dict[str, Any] = {
            "dataset": dataset,
            "selected_method": method,
            "n_subjects": int(len(part)),
        }
        for metric_name in ("accuracy", "Brier", "ECE", "NLL"):
            for prefix in ("val", "heldout"):
                values = part[f"{prefix}_{metric_name}"].astype(float)
                record[f"{prefix}_{metric_name}_mean"] = float(values.mean())
                record[f"{prefix}_{metric_name}_std"] = float(values.std(ddof=0))
        records.append(record)
    method_counts = (
        df.groupby(["dataset", "selected_method"]).size().reset_index(name="selected_count")
    )
    return pd.DataFrame(records).merge(method_counts, on=["dataset", "selected_method"]).sort_values(["dataset", "selected_method"])


def aggregate_portfolio(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    records = []
    for dataset, part in df.groupby("dataset"):
        record: dict[str, Any] = {
            "dataset": dataset,
            "experiment_key": "validation_selected_portfolio",
            "experiment": "validation_selected_portfolio",
            "n_subjects": int(len(part)),
        }
        for metric_name in ("accuracy", "Brier", "ECE", "NLL"):
            for prefix in ("val", "heldout"):
                values = part[f"{prefix}_{metric_name}"].astype(float)
                record[f"{prefix}_{metric_name}_mean"] = float(values.mean())
                record[f"{prefix}_{metric_name}_std"] = float(values.std(ddof=0))
        records.append(record)
    return pd.DataFrame(records).sort_values("dataset")


def load_reference_rows(result_dir: Path) -> pd.DataFrame:
    paths = sorted(result_dir.glob("validation_only_selection*subject_metrics*.csv"))
    paths = [p for p in paths if not p.name.startswith("validation_selected_portfolio")]
    if not paths:
        return pd.DataFrame()
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        if "dataset" not in df.columns:
            if result_dir.name == "calibration_workflow":
                df["dataset"] = "BCI IV-2a"
            else:
                df["dataset"] = result_dir.name.replace("calibration_workflow_", "")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(subset=["dataset", "subject", "experiment_key"], keep="first")


def wilcoxon_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    nonzero = values[~np.isclose(values, 0.0)]
    if nonzero.size == 0:
        return 1.0
    return float(wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox").pvalue)


def paired_deltas(portfolio_rows: list[dict[str, Any]], reference_frames: list[pd.DataFrame]) -> pd.DataFrame:
    portfolio = pd.DataFrame(portfolio_rows)
    refs = pd.concat([df for df in reference_frames if not df.empty], ignore_index=True)
    records = []
    for reference in ("baseline", "seed_ensemble"):
        ref = refs[refs["experiment_key"] == reference].copy()
        if ref.empty:
            continue
        ref["dataset"] = ref["dataset"].replace({"bci4_2a": "BCI IV-2a", "bci_iiia": "BCI IIIa"})
        for dataset, part in portfolio.groupby("dataset"):
            base = ref[ref["dataset"].eq(dataset)]
            merged = part.merge(base, on=["dataset", "subject"], suffixes=("_portfolio", "_reference"))
            if merged.empty:
                continue
            for metric_name in ("accuracy", "Brier", "ECE", "NLL"):
                delta = merged[f"heldout_{metric_name}_portfolio"].astype(float) - merged[f"heldout_{metric_name}_reference"].astype(float)
                lower_is_better = metric_name in {"Brier", "ECE", "NLL"}
                wins = int((delta < -EPS).sum()) if lower_is_better else int((delta > EPS).sum())
                losses = int((delta > EPS).sum()) if lower_is_better else int((delta < -EPS).sum())
                ties = int(len(delta) - wins - losses)
                records.append(
                    {
                        "dataset": dataset,
                        "reference_method": reference,
                        "metric": metric_name,
                        "n_subjects": int(len(delta)),
                        "mean_delta": float(delta.mean()),
                        "median_delta": float(delta.median()),
                        "wilcoxon_p": wilcoxon_p(delta.to_numpy(dtype=float)),
                        "wins": wins,
                        "ties": ties,
                        "losses": losses,
                    }
                )
    return pd.DataFrame(records).sort_values(["dataset", "reference_method", "metric"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default=Path("results"), type=Path)
    parser.add_argument("--out-dir", default=Path("results") / "paper_calibration_stats" / "validation_selected_portfolio", type=Path)
    parser.add_argument("--include-smoke", action="store_true")
    parser.add_argument("--no-optional-candidates", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    reference_frames: list[pd.DataFrame] = []
    for result_dir in iter_result_dirs(args.results_root, include_smoke=args.include_smoke):
        subject_paths = iter_subject_jsons(result_dir)
        if not subject_paths:
            continue
        rows = [
            portfolio_row(json.loads(path.read_text(encoding="utf-8")), path, include_optional=not args.no_optional_candidates)
            for path in subject_paths
        ]
        write_csv(result_dir / "validation_selected_portfolio_subject_metrics.csv", rows)
        write_portfolio_probabilities(result_dir, rows, args.out_dir / result_dir.name)
        all_rows.extend(rows)
        reference_frames.append(load_reference_rows(result_dir))

    if not all_rows:
        raise SystemExit("No validation-only subject JSON files found.")

    write_csv(args.out_dir / "validation_selected_portfolio_subject_metrics.csv", all_rows)
    summarize(all_rows).to_csv(args.out_dir / "validation_selected_portfolio_selection_counts.csv", index=False)
    aggregate_portfolio(all_rows).to_csv(args.out_dir / "validation_selected_portfolio_summary.csv", index=False)
    paired_deltas(all_rows, reference_frames).to_csv(args.out_dir / "validation_selected_portfolio_paired_deltas.csv", index=False)


if __name__ == "__main__":
    main()
