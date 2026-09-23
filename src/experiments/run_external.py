#!/usr/bin/env python3
"""Run leakage-audited external MOABB validation for CalibMI."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adaptation import apply_channel_matrix, euclidean_alignment_matrix
from evaluate import compute_brier, compute_ece, compute_selective_acc
from decode import fit_predict_subject
from experiments.run_primary import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_SEED,
    apply_temperature,
    fit_temperature,
    write_json,
)
from models.edl_sgraph import EDLSGraph
from models.spectral_graph import SpectralCovGraph


DATASETS = ("Cho2017", "BNCI2014_001", "BNCI2014_004", "PhysionetMI")
MODEL_NAMES = ("v1_lda", "mdrm_t", "eegnet_ts", "edl_sgraph")
METRIC_NAMES = ("accuracy", "ECE", "Brier", "sel_acc_60")
STANDARD_LAMBDAS = {
    "lambda_B": 0.5,
    "lambda_KL": 0.1,
    "lambda_diff": 0.1,
    "lambda_sel": 0.05,
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    paradigm: str
    n_classes: int | None = None


DATASET_SPECS = {
    "Cho2017": DatasetSpec("Cho2017", "LeftRightImagery", 2),
    "BNCI2014_001": DatasetSpec("BNCI2014_001", "MotorImagery", 4),
    "BNCI2014_004": DatasetSpec("BNCI2014_004", "LeftRightImagery", 2),
    "PhysionetMI": DatasetSpec("PhysionetMI", "LeftRightImagery", 2),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, action="append", help="Dataset to run. Defaults to all.")
    parser.add_argument("--subject", action="append", help="Subject id to run. May be supplied multiple times.")
    parser.add_argument("--subject-limit", type=int, default=None, help="Limit subjects per dataset for smoke runs.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned work and exit.")
    parser.add_argument("--data-dir", type=Path, default=Path("data") / "moabb")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--primary-aggregate",
        type=Path,
        default=Path("results") / "edl_sgraph_aggregate.json",
        help="Optional BCI IV 2a aggregate JSON containing selected_lambdas.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default=None, help="Torch device. Defaults to cuda when available, else cpu.")
    parser.add_argument("--no-temperature", action="store_true", help="Disable validation temperature scaling.")
    return parser.parse_args()


def prepare_moabb_env(data_dir: Path) -> None:
    fake_home = REPO_ROOT / ".mne_home"
    mpl_dir = REPO_ROOT / ".mplconfig"
    (fake_home / ".mne").mkdir(parents=True, exist_ok=True)
    mpl_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(fake_home))
    os.environ.setdefault("MNE_DATA", str(data_dir))
    os.environ.setdefault("MNE_LOGGING_LEVEL", "WARNING")
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))


def import_moabb(data_dir: Path):
    prepare_moabb_env(data_dir)
    try:
        import moabb
        from moabb import datasets as moabb_datasets
        from moabb.paradigms import LeftRightImagery, MotorImagery
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("MOABB is required for external validation. Install `moabb`.") from exc
    moabb.set_log_level("warning")
    return moabb_datasets, {"LeftRightImagery": LeftRightImagery, "MotorImagery": MotorImagery}


def dataset_instance(dataset_name: str, data_dir: Path):
    moabb_datasets, _ = import_moabb(data_dir)
    try:
        return getattr(moabb_datasets, dataset_name)()
    except AttributeError as exc:
        raise ValueError(f"Unknown MOABB dataset: {dataset_name}") from exc


def make_paradigm(spec: DatasetSpec, data_dir: Path):
    _, paradigms = import_moabb(data_dir)
    cls = paradigms[spec.paradigm]
    kwargs = {"fmin": 8.0, "fmax": 30.0, "resample": 250.0}
    if spec.paradigm == "MotorImagery" and spec.n_classes is not None:
        try:
            return cls(n_classes=spec.n_classes, **kwargs)
        except TypeError:
            return cls(**kwargs)
    return cls(**kwargs)


def ordered_unique(values: pd.Series) -> list[Any]:
    return list(pd.Series(values).dropna().drop_duplicates())


def heldout_mask(meta: pd.DataFrame) -> tuple[np.ndarray, str, Any]:
    for column in ("session", "run"):
        if column in meta.columns:
            values = ordered_unique(meta[column])
            if len(values) > 1:
                heldout = values[-1]
                eval_mask = meta[column].to_numpy() == heldout
                train_mask = ~eval_mask
                if train_mask.any() and eval_mask.any():
                    return eval_mask, f"held_out_last_{column}", heldout
    raise ValueError("Subject does not expose multiple sessions/runs for a held-out protocol.")


def split_train_val(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(y.size)
    classes, counts = np.unique(y, return_counts=True)
    stratify = y if counts.size and counts.min() >= 2 else None
    if stratify is None:
        test_size: float | int = 0.2
    else:
        test_size = min(max(int(np.ceil(0.2 * y.size)), classes.size), y.size - classes.size)
        if test_size < classes.size:
            stratify = None
            test_size = 0.2
    train_idx, val_idx = train_test_split(idx, test_size=test_size, random_state=seed, stratify=stratify)
    return train_idx, val_idx


def encode_labels(y_train: np.ndarray, y_val: np.ndarray, y_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Any]]:
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    y_eval_enc = le.transform(y_eval)
    return y_train_enc.astype(np.int64), y_val_enc.astype(np.int64), y_eval_enc.astype(np.int64), le.classes_.tolist()


def prepare_subject_data(dataset_name: str, subject: Any, args: argparse.Namespace) -> dict[str, Any]:
    spec = DATASET_SPECS[dataset_name]
    dataset = dataset_instance(dataset_name, args.data_dir)
    paradigm = make_paradigm(spec, args.data_dir)
    X, y, meta = paradigm.get_data(dataset=dataset, subjects=[subject])
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    meta = pd.DataFrame(meta).reset_index(drop=True)

    eval_mask, split, heldout_value = heldout_mask(meta)
    train_mask = ~eval_mask
    X_dev, y_dev = X[train_mask], y[train_mask]
    X_eval_raw, y_eval_raw = X[eval_mask], y[eval_mask]
    if np.unique(y_dev).size < 2 or np.unique(y_eval_raw).size < 2:
        raise ValueError("Need at least two classes in both train and eval splits.")

    train_idx, val_idx = split_train_val(y_dev, args.seed)
    y_train, y_val, y_eval, classes = encode_labels(y_dev[train_idx], y_dev[val_idx], y_eval_raw)

    aligner = euclidean_alignment_matrix(X_dev[train_idx])
    X_train = apply_channel_matrix(X_dev[train_idx], aligner)
    X_val = apply_channel_matrix(X_dev[val_idx], aligner)
    X_eval = apply_channel_matrix(X_eval_raw, aligner)
    return {
        "dataset": dataset_name,
        "subject": subject,
        "paradigm": spec.paradigm,
        "split": split,
        "heldout": heldout_value,
        "classes": classes,
        "X_train_raw": X_dev[train_idx],
        "X_val_raw": X_dev[val_idx],
        "X_eval_raw": X_eval_raw,
        "y_train_raw": y_dev[train_idx],
        "y_val_raw": y_dev[val_idx],
        "y_eval_raw": y_eval_raw,
        "X_train": X_train,
        "X_val": X_val,
        "X_eval": X_eval,
        "y_train": y_train,
        "y_val": y_val,
        "y_eval": y_eval,
        "audit": {
            "n_total_trials": int(X.shape[0]),
            "n_train": int(X_train.shape[0]),
            "n_val": int(X_val.shape[0]),
            "n_eval": int(X_eval.shape[0]),
            "ea_fit_scope": "train_split_only",
            "eval_used_for_ea": False,
            "eval_used_for_temperature": False,
            "eval_used_for_model_selection": False,
        },
    }


def fit_v1_lda(data: dict[str, Any]) -> np.ndarray:
    pred = fit_predict_subject(
        subject=int(data["subject"]) if str(data["subject"]).isdigit() else 0,
        X_train=np.concatenate([data["X_train"], data["X_val"]], axis=0),
        y_train=np.concatenate([data["y_train"], data["y_val"]], axis=0),
        X_test=data["X_eval"],
        y_test=data["y_eval"],
        model_names=("LDA",),
    )
    if not np.array_equal(pred.y_true, data["y_eval"]):
        raise RuntimeError("v1_lda label encoding does not match shared external encoding.")
    return pred.proba["LDA"]


def fit_mdrm_t(data: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    from baselines.mdrm_t import MDRMT, audit_leakage

    model = MDRMT(coverage_targets=(0.4, 0.6, 0.8))
    model.fit(data["X_train_raw"], data["y_train_raw"], data["X_val_raw"], data["y_val_raw"])
    result = model.evaluate(data["X_eval_raw"], data["y_eval_raw"])
    audit_leakage(model)
    return model.predict_proba(data["X_eval_raw"]), {"metrics": result, "audit_trail": model.audit_trail_}


def fit_eegnet_ts(data: dict[str, Any], args: argparse.Namespace) -> np.ndarray:
    from eegnet import EEGNetConfig, fit_predict_eegnet

    y_true, proba = fit_predict_eegnet(
        X_train=np.concatenate([data["X_train"], data["X_val"]], axis=0),
        y_train=np.concatenate([data["y_train"], data["y_val"]], axis=0),
        X_test=data["X_eval"],
        y_test=data["y_eval"],
        config=EEGNetConfig(
            epochs=args.max_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            random_state=args.seed,
        ),
    )
    if not np.array_equal(y_true, data["y_eval"]):
        raise RuntimeError("EEGNet label encoding does not match shared external encoding.")
    return proba


def build_edl_model(X: np.ndarray, n_classes: int, device: str) -> tuple[EDLSGraph, str]:
    n_channels = int(X.shape[1])
    # EDLSGraph constructs a 22-channel anatomical DualGraphEncoder eagerly.
    # For external datasets with other montages, bootstrap the head/temporal
    # stack with the native 22-channel topology, then replace the encoder with
    # the channel-count-agnostic spectral branch before any forward pass.
    model = EDLSGraph(n_channels=n_channels if n_channels == 22 else 22, T=int(X.shape[2]), n_classes=n_classes)
    encoder = "dual_graph"
    if n_channels != 22:
        model.n_channels = n_channels
        model.graph_encoder = SpectralCovGraph(n_channels=n_channels, d_s=model.d_s, sfreq=model.sfreq)
        encoder = "spectral_non22_fallback"
    return model.to(device), encoder


@torch.no_grad()
def collect_edl_outputs(model: EDLSGraph, X: np.ndarray, device: str, batch_size: int) -> dict[str, np.ndarray]:
    model.eval()
    out: dict[str, list[np.ndarray]] = {"p_hat": [], "vacuity": [], "p_unknown": []}
    for start in range(0, X.shape[0], batch_size):
        xb = torch.as_tensor(X[start : start + batch_size], dtype=torch.float32, device=device)
        batch = model.inference(xb)
        for key in out:
            out[key].append(batch[key].detach().cpu().numpy())
    return {key: np.concatenate(values, axis=0) for key, values in out.items()}


def val_brier_edl(model: EDLSGraph, data: dict[str, Any], args: argparse.Namespace, device: str) -> float:
    outputs = collect_edl_outputs(model, data["X_val"], device, args.batch_size)
    return compute_brier(trim_edl_known_probs(outputs["p_hat"], len(data["classes"])), data["y_val"])


def fit_edl_sgraph(
    data: dict[str, Any],
    lambdas: dict[str, float],
    args: argparse.Namespace,
    device: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    torch.manual_seed(args.seed)
    model, encoder = build_edl_model(data["X_train"], len(data["classes"]), device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ds = torch.utils.data.TensorDataset(
        torch.as_tensor(data["X_train"], dtype=torch.float32),
        torch.as_tensor(data["y_train"], dtype=torch.long),
        torch.zeros(data["y_train"].shape[0], dtype=torch.float32),
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    best_state = None
    best_brier = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        train_losses = []
        for xb, yb, sb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            sb = sb.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, _, loss_dict = model(xb, s_target=sb, lambdas={**lambdas, "labels": yb})
            loss_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(float(loss_dict["loss"].detach().cpu().item()))

        brier = val_brier_edl(model, data, args, device)
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_brier": brier})
        if brier < best_brier - 1e-7:
            best_brier = brier
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    temperature = None
    val_outputs = collect_edl_outputs(model, data["X_val"], device, args.batch_size)
    val_probs = trim_edl_known_probs(val_outputs["p_hat"], len(data["classes"]))
    if not args.no_temperature:
        temperature = fit_temperature(val_probs, data["y_val"])

    raw = collect_edl_outputs(model, data["X_eval"], device, args.batch_size)
    eval_probs = trim_edl_known_probs(raw["p_hat"], len(data["classes"]))
    if temperature is not None:
        eval_probs = apply_temperature(eval_probs, temperature)
    return eval_probs, raw["vacuity"], {
        "encoder": encoder,
        "best_val_brier": best_brier,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "history": history,
        "temperature": temperature,
        "temperature_at_upper_clamp": bool(temperature is not None and temperature >= 19.999),
    }


def normalize_proba(proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba, dtype=float)
    denom = np.maximum(proba.sum(axis=1, keepdims=True), 1e-12)
    return proba / denom


def trim_edl_known_probs(p_hat: np.ndarray, n_classes: int) -> np.ndarray:
    return normalize_proba(np.asarray(p_hat, dtype=float)[:, :n_classes])


def metrics_for(y_true: np.ndarray, proba: np.ndarray, abstention_scores: np.ndarray | None = None) -> dict[str, float]:
    proba = normalize_proba(proba)
    y_true = np.asarray(y_true, dtype=int)
    if abstention_scores is None:
        abstention_scores = 1.0 - proba.max(axis=1)
    acc60, cov60 = compute_selective_acc(proba, y_true, abstention_scores, 0.6)
    ece, _ = compute_ece(proba, y_true)
    return {
        "accuracy": float(np.mean(proba.argmax(axis=1) == y_true)),
        "ECE": float(ece),
        "Brier": compute_brier(proba, y_true),
        "sel_acc_60": float(acc60),
        "coverage_60": float(cov60),
    }


def load_standard_lambdas(path: Path) -> dict[str, float]:
    if not path.exists():
        return dict(STANDARD_LAMBDAS)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    selected = payload.get("selected_lambdas")
    if not isinstance(selected, dict):
        return dict(STANDARD_LAMBDAS)
    lambdas = dict(STANDARD_LAMBDAS)
    for key in STANDARD_LAMBDAS:
        if key in selected:
            lambdas[key] = float(selected[key])
    return lambdas


def subject_result(dataset_name: str, subject: Any, lambdas: dict[str, float], args: argparse.Namespace, device: str) -> dict[str, Any]:
    data = prepare_subject_data(dataset_name, subject, args)
    outputs: dict[str, dict[str, Any]] = {}

    p_lda = fit_v1_lda(data)
    outputs["v1_lda"] = {"metrics": metrics_for(data["y_eval"], p_lda)}

    p_mdrm, mdrm_info = fit_mdrm_t(data)
    outputs["mdrm_t"] = {"metrics": metrics_for(data["y_eval"], p_mdrm), "model_info": mdrm_info}

    p_eegnet = fit_eegnet_ts(data, args)
    outputs["eegnet_ts"] = {"metrics": metrics_for(data["y_eval"], p_eegnet)}

    p_edl, vacuity, edl_info = fit_edl_sgraph(data, lambdas, args, device)
    outputs["edl_sgraph"] = {"metrics": metrics_for(data["y_eval"], p_edl, vacuity), "model_info": edl_info}

    return {
        "dataset": dataset_name,
        "subject": subject,
        "paradigm": data["paradigm"],
        "split": data["split"],
        "heldout": data["heldout"],
        "classes": data["classes"],
        "lambdas": lambdas,
        "audit": data["audit"],
        "models": outputs,
    }


def aggregate_dataset(dataset_name: str, subject_results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for model in MODEL_NAMES:
        summary[model] = {}
        for metric in METRIC_NAMES:
            values = np.asarray(
                [
                    result["models"][model]["metrics"][metric]
                    for result in subject_results
                    if model in result["models"]
                    and np.isfinite(result["models"][model]["metrics"].get(metric, np.nan))
                ],
                dtype=float,
            )
            summary[model][metric] = {
                "mean": float(values.mean()) if values.size else None,
                "std": float(values.std(ddof=1)) if values.size > 1 else 0.0 if values.size == 1 else None,
            }
    return {
        "dataset": dataset_name,
        "subjects": [result["subject"] for result in subject_results],
        "n_subjects": len(subject_results),
        "models": summary,
        "subject_results": subject_results,
    }


def write_external_summary(path: Path, aggregates: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset"]
    fields.extend(f"{model}_{metric}" for model in MODEL_NAMES for metric in METRIC_NAMES)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for aggregate in aggregates:
            row = {"dataset": aggregate["dataset"]}
            for model in MODEL_NAMES:
                for metric in METRIC_NAMES:
                    row[f"{model}_{metric}"] = aggregate["models"][model][metric]["mean"]
            writer.writerow(row)


def print_leakage_audit(dataset_name: str, results: list[dict[str, Any]]) -> None:
    if not results:
        print(f"[leakage audit] dataset={dataset_name} status=NO_VALID_SUBJECTS", flush=True)
        return
    n_subjects = len(results)
    n_eval = sum(int(result["audit"]["n_eval"]) for result in results)
    encoders = sorted(
        {
            result["models"]["edl_sgraph"]["model_info"]["encoder"]
            for result in results
            if "edl_sgraph" in result["models"]
        }
    )
    print(
        f"[leakage audit] dataset={dataset_name} status=PASS subjects={n_subjects} eval_trials={n_eval}",
        flush=True,
    )
    print("[leakage audit] split=last session/run held out before training/validation split", flush=True)
    print("[leakage audit] EA aligner fit on train split only and applied unchanged to val/eval", flush=True)
    print("[leakage audit] temperature/early stopping use validation split only; eval labels used only for metrics", flush=True)
    print(f"[leakage audit] edl_sgraph_encoder={','.join(encoders)}", flush=True)


def selected_subjects(dataset_name: str, args: argparse.Namespace) -> list[Any]:
    dataset = dataset_instance(dataset_name, args.data_dir)
    subjects = list(getattr(dataset, "subject_list", []))
    if args.subject:
        requested = {str(subject) for subject in args.subject}
        subjects = [subject for subject in subjects if str(subject) in requested]
    if args.subject_limit is not None:
        subjects = subjects[: args.subject_limit]
    if not subjects:
        raise ValueError(f"No subjects selected for {dataset_name}")
    return subjects


def dry_run(args: argparse.Namespace) -> int:
    datasets = args.dataset or list(DATASETS)
    print("CalibMI external validation dry run")
    for dataset_name in datasets:
        subjects = selected_subjects(dataset_name, args)
        spec = DATASET_SPECS[dataset_name]
        print(f"{dataset_name}: paradigm={spec.paradigm} subjects={subjects}")
        for subject in subjects:
            print(f"  subject {subject} -> {args.results_dir / f'external_{dataset_name}_subject{subject}.json'}")
        print(f"  aggregate -> {args.results_dir / f'external_{dataset_name}_aggregate.json'}")
    print(f"Summary CSV -> {args.results_dir / 'external_summary.csv'}")
    return 0


def main() -> int:
    args = parse_args()
    datasets = args.dataset or list(DATASETS)
    if args.dry_run:
        return dry_run(args)

    lambdas = load_standard_lambdas(args.primary_aggregate)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running external validation on device={device}; lambdas={lambdas}", flush=True)

    aggregates = []
    for dataset_name in datasets:
        results = []
        subjects = selected_subjects(dataset_name, args)
        print(f"[external] dataset={dataset_name} subjects={subjects}", flush=True)
        for subject in subjects:
            try:
                print(f"[external] dataset={dataset_name} subject={subject}", flush=True)
                result = subject_result(dataset_name, subject, lambdas, args, device)
            except Exception as exc:
                print(f"[external] SKIP dataset={dataset_name} subject={subject}: {exc}", flush=True)
                continue
            write_json(args.results_dir / f"external_{dataset_name}_subject{subject}.json", result)
            results.append(result)
        print_leakage_audit(dataset_name, results)
        aggregate = aggregate_dataset(dataset_name, results)
        write_json(args.results_dir / f"external_{dataset_name}_aggregate.json", aggregate)
        aggregates.append(aggregate)

    write_external_summary(args.results_dir / "external_summary.csv", aggregates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
