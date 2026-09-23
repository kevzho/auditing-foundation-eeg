#!/usr/bin/env python3
"""External competence-anchor validation for cropped CE MI decoders.

This runner deliberately avoids graph/EDL novelty claims.  It tests whether the
BCI IV-2a competence-anchor story generalizes to external MOABB datasets using
fixed, auditable session/run protocols.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import CALIBRATION_CV, CALIBRATION_METHOD, FBCSP_BANDS, INCLUDE_BANDPOWER, SFREQ
from evaluate import compute_brier, compute_ece, compute_selective_acc
from experiments.run_classical_anchor_competence import (
    apply_standardize,
    distill_loss,
    fit_temperature_from_logits,
    make_jsonable,
    proba_from_logits,
    standardize_fit,
    write_json,
)
from experiments.run_cropped_anchor_competence import (
    aggregate_trial_logits,
    augment_crops,
    make_multiscale_crops,
)
from experiments.run_primary import DEFAULT_BATCH_SIZE, DEFAULT_SEED
from fbcsp import FBCSPFeatures
from models.competence_graph_eegnet import CompetenceConfig, build_competence_model


MODEL_SET = (
    "lda_svm_vote",
    "eegnet_ts",
    "raw_cropped_shallow_convnet_ce",
    "anchor_cropped_shallow_convnet_ce",
)
OPTIONAL_MULTISCALE = "raw_multiscale_temporal_ce"
ANCHOR_TO_RAW = {"anchor_cropped_shallow_convnet_ce": "raw_cropped_shallow_convnet_ce"}
RAW_VARIANT = {
    "eegnet_ts": "raw_eegnet_ce",
    "raw_cropped_shallow_convnet_ce": "raw_shallow_convnet_ce",
    "anchor_cropped_shallow_convnet_ce": "raw_shallow_convnet_ce",
    "raw_multiscale_temporal_ce": "raw_multiscale_temporal_ce",
}
DATASETS = ("BNCI2014_001", "BNCI2014_004", "PhysionetMI")
DATASET_LABELS = {
    "BNCI2014_001": "BNCI2014_001",
    "BNCI2014_004": "BCI_IV_2b",
    "PhysionetMI": "PhysioNet_MI",
}
METRICS = ("accuracy", "ECE", "Brier", "sel_acc_60", "coverage_60")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    paradigm: str
    n_classes: int | None
    default_run: bool


DATASET_SPECS = {
    "BNCI2014_001": DatasetSpec("BNCI2014_001", "MotorImagery", 4, True),
    "BNCI2014_004": DatasetSpec("BNCI2014_004", "LeftRightImagery", 2, True),
    "PhysionetMI": DatasetSpec("PhysionetMI", "LeftRightImagery", 2, False),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, action="append", help="Dataset to run. Defaults to BNCI2014_001 only.")
    parser.add_argument("--subject", action="append", help="Subject id to run. May be supplied multiple times.")
    parser.add_argument("--subject-limit", type=int, default=None)
    parser.add_argument("--include-bci-iv-2b", action="store_true", help="Also run BNCI2014_004/BCI IV-2b when protocol inspection is clean.")
    parser.add_argument("--include-physionet", action="store_true", help="Inspect PhysioNet MI; run only if protocol is accepted.")
    parser.add_argument("--include-multiscale", action="store_true", help="Run raw_multiscale_temporal_ce as tested-but-not-selected ablation.")
    parser.add_argument("--inspect-only", action="store_true", help="Print protocol structures and write no model outputs.")
    parser.add_argument("--data-dir", type=Path, default=Path("data") / "moabb")
    parser.add_argument("--results-dir", type=Path, default=Path("results") / "external_competence_anchor")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lambda-distill", type=float, default=0.5)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--crop-sizes", type=int, nargs="+", default=[384, 512, 640])
    parser.add_argument("--crop-strides", type=int, nargs="+", default=[128])
    parser.add_argument("--balanced-sampler", dest="balanced_sampler", action="store_true", default=True)
    parser.add_argument("--no-balanced-sampler", dest="balanced_sampler", action="store_false")
    parser.add_argument("--augment", dest="augment", action="store_true", default=True)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--noise-std", type=float, default=0.02)
    parser.add_argument("--time-mask-frac", type=float, default=0.05)
    parser.add_argument("--temperature-max", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if len(args.crop_strides) not in (1, len(args.crop_sizes)):
        parser.error("--crop-strides must have length 1 or match --crop-sizes")
    return args


def selected_datasets(args: argparse.Namespace) -> list[str]:
    if args.dataset:
        return list(dict.fromkeys(args.dataset))
    datasets = ["BNCI2014_001"]
    if args.include_bci_iv_2b:
        datasets.append("BNCI2014_004")
    if args.include_physionet:
        datasets.append("PhysionetMI")
    return datasets


def prepare_moabb_env(data_dir: Path) -> None:
    fake_home = REPO_ROOT / ".mne_home"
    mpl_dir = REPO_ROOT / ".mplconfig"
    (fake_home / ".mne").mkdir(parents=True, exist_ok=True)
    mpl_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    os.environ["_MNE_FAKE_HOME_DIR"] = str(fake_home)
    os.environ["MNE_DATA"] = str(data_dir)
    os.environ["MNE_LOGGING_LEVEL"] = "WARNING"
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)


def import_moabb(data_dir: Path):
    prepare_moabb_env(data_dir)
    import moabb
    from moabb import datasets as moabb_datasets
    from moabb.paradigms import LeftRightImagery, MotorImagery

    moabb.set_log_level("warning")
    return moabb_datasets, {"LeftRightImagery": LeftRightImagery, "MotorImagery": MotorImagery}


def dataset_instance(dataset_name: str, data_dir: Path):
    moabb_datasets, _ = import_moabb(data_dir)
    return getattr(moabb_datasets, dataset_name)()


def make_paradigm(spec: DatasetSpec, data_dir: Path):
    _, paradigms = import_moabb(data_dir)
    kwargs = {"fmin": 8.0, "fmax": 30.0, "resample": 250.0}
    cls = paradigms[spec.paradigm]
    if spec.paradigm == "MotorImagery" and spec.n_classes is not None:
        try:
            return cls(n_classes=spec.n_classes, **kwargs)
        except TypeError:
            return cls(**kwargs)
    return cls(**kwargs)


def ordered_unique(values: pd.Series) -> list[Any]:
    return list(pd.Series(values).dropna().drop_duplicates())


def structure_summary(meta: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {"columns": list(meta.columns), "n_trials": int(len(meta))}
    for column in ("session", "run"):
        if column in meta.columns:
            counts = meta.groupby(column, dropna=False).size().to_dict()
            out[column] = {"values": [str(v) for v in ordered_unique(meta[column])], "counts": {str(k): int(v) for k, v in counts.items()}}
    if {"session", "run"}.issubset(meta.columns):
        counts = meta.groupby(["session", "run"], dropna=False).size().reset_index(name="n")
        out["session_run_counts"] = [
            {"session": str(row["session"]), "run": str(row["run"]), "n": int(row["n"])}
            for _, row in counts.iterrows()
        ]
    return out


def session_sort_key(value: Any) -> tuple[int, str]:
    text = str(value)
    digits = "".join(ch for ch in text if ch.isdigit())
    return (int(digits) if digits else 10_000, text)


def protocol_masks(meta: pd.DataFrame, dataset_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if dataset_name == "PhysionetMI":
        raise ValueError("PhysioNet MI is skipped: no predeclared leakage-safe cross-session/cross-run protocol in this runner.")
    if "session" not in meta.columns:
        raise ValueError("No session column; refusing ambiguous external protocol.")
    session_text = meta["session"].astype(str).str.lower()
    train_sessions = ordered_unique(meta.loc[session_text.str.contains("train"), "session"])
    eval_sessions = ordered_unique(meta.loc[session_text.str.contains("test"), "session"])
    if not train_sessions or not eval_sessions:
        raise ValueError("Expected explicit train/test session labels.")

    train_eval_mask = meta["session"].isin(train_sessions).to_numpy()
    eval_mask = meta["session"].isin(eval_sessions).to_numpy()
    train_meta = meta.loc[train_eval_mask].copy()

    if len(train_sessions) > 1:
        val_session = sorted(train_sessions, key=session_sort_key)[-1]
        val_mask = (meta["session"] == val_session).to_numpy()
        train_mask = train_eval_mask & ~val_mask
        validation_rule = "held_out_last_training_session"
        validation_value = str(val_session)
    elif "run" in meta.columns and len(ordered_unique(train_meta["run"])) > 1:
        train_runs = ordered_unique(train_meta["run"])
        val_run = sorted(train_runs, key=session_sort_key)[-1]
        val_mask = train_eval_mask & (meta["run"] == val_run).to_numpy()
        train_mask = train_eval_mask & ~val_mask
        validation_rule = "held_out_last_training_run"
        validation_value = str(val_run)
    else:
        raise ValueError("Training split has no unambiguous session/run holdout for validation.")

    if not train_mask.any() or not val_mask.any() or not eval_mask.any():
        raise ValueError("Empty train/validation/eval mask after protocol selection.")
    protocol = {
        "dataset": dataset_name,
        "dataset_label": DATASET_LABELS.get(dataset_name, dataset_name),
        "train_sessions": [str(v) for v in train_sessions],
        "eval_sessions": [str(v) for v in eval_sessions],
        "validation_rule": validation_rule,
        "validation_value": validation_value,
        "train_eval_protocol": "explicit MOABB train sessions to explicit MOABB test sessions",
        "eval_labels_used_for": "final_metrics_only",
    }
    return train_mask, val_mask, eval_mask, protocol


def encode_labels(y_train: np.ndarray, y_val: np.ndarray, y_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    le = LabelEncoder()
    y_train_e = le.fit_transform(y_train).astype(np.int64)
    y_val_e = le.transform(y_val).astype(np.int64)
    y_eval_e = le.transform(y_eval).astype(np.int64)
    return y_train_e, y_val_e, y_eval_e, [str(c) for c in le.classes_]


def class_counts(y: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(y, return_counts=True)
    return {str(v): int(c) for v, c in zip(values, counts)}


def prepare_subject_data(dataset_name: str, subject: Any, args: argparse.Namespace) -> dict[str, Any]:
    spec = DATASET_SPECS[dataset_name]
    dataset = dataset_instance(dataset_name, args.data_dir)
    paradigm = make_paradigm(spec, args.data_dir)
    X, y, meta = paradigm.get_data(dataset=dataset, subjects=[subject])
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    meta = pd.DataFrame(meta).reset_index(drop=True)
    train_mask, val_mask, eval_mask, protocol = protocol_masks(meta, dataset_name)
    y_train, y_val, y_eval, classes = encode_labels(y[train_mask], y[val_mask], y[eval_mask])
    n_classes = len(classes)
    for split_name, labels in (("train", y_train), ("validation", y_val), ("eval", y_eval)):
        if np.unique(labels).size != n_classes:
            raise ValueError(f"{split_name} split lacks all {n_classes} classes.")
    return {
        "dataset": dataset_name,
        "dataset_label": DATASET_LABELS.get(dataset_name, dataset_name),
        "subject": subject,
        "classes": classes,
        "n_classes": n_classes,
        "chance_accuracy": 1.0 / n_classes,
        "chance_brier": (n_classes - 1.0) / n_classes,
        "X_train_raw": X[train_mask],
        "X_val_raw": X[val_mask],
        "X_eval_raw": X[eval_mask],
        "y_train": y_train,
        "y_val": y_val,
        "y_eval": y_eval,
        "y_train_raw": y[train_mask],
        "y_val_raw": y[val_mask],
        "y_eval_raw": y[eval_mask],
        "meta_structure": structure_summary(meta),
        "protocol": {
            **protocol,
            "n_total_trials": int(X.shape[0]),
            "n_train": int(train_mask.sum()),
            "n_validation": int(val_mask.sum()),
            "n_eval": int(eval_mask.sum()),
            "train_class_counts": class_counts(y[train_mask]),
            "validation_class_counts": class_counts(y[val_mask]),
            "eval_class_counts": class_counts(y[eval_mask]),
            "train_mask_rule": "session/run metadata only",
            "validation_mask_rule": protocol["validation_rule"],
            "standardization_fit_split": "train",
            "teacher_fit_split": "train",
            "temperature_fit_split": "validation_trial_logits",
            "eval_used_for_standardization": False,
            "eval_used_for_teacher_fit": False,
            "eval_used_for_temperature": False,
            "eval_used_for_model_selection": False,
        },
    }


def align_proba(p: np.ndarray, classes: np.ndarray, n_rows: int, n_classes: int) -> np.ndarray:
    aligned = np.zeros((n_rows, n_classes), dtype=np.float64)
    for j, cls in enumerate(classes):
        aligned[:, int(cls)] = p[:, j]
    denom = np.maximum(aligned.sum(axis=1, keepdims=True), 1e-12)
    return aligned / denom


def build_external_decoder(name: str, n_channels: int) -> CalibratedClassifierCV:
    if name == "lda":
        clf = LinearDiscriminantAnalysis()
    elif name == "svm":
        clf = SVC(kernel="rbf", C=1.0, gamma="scale")
    else:
        raise ValueError(name)
    feat = FBCSPFeatures(
        sfreq=SFREQ,
        bands=FBCSP_BANDS,
        n_components=min(4, int(n_channels)),
        include_bandpower=INCLUDE_BANDPOWER,
    )
    return CalibratedClassifierCV(Pipeline([("feat", feat), ("clf", clf)]), method=CALIBRATION_METHOD, cv=CALIBRATION_CV)


def fit_teacher(data: dict[str, Any]) -> dict[str, Any]:
    train_probas = {}
    val_probas = {}
    eval_probas = {}
    audits = {}
    n_channels = int(data["X_train_raw"].shape[1])
    for name in ("lda", "svm"):
        decoder = build_external_decoder(name, n_channels)
        decoder.fit(data["X_train_raw"], data["y_train"])
        classes = getattr(decoder, "classes_", np.arange(data["n_classes"]))
        train_probas[name] = align_proba(decoder.predict_proba(data["X_train_raw"]), classes, data["X_train_raw"].shape[0], data["n_classes"])
        val_probas[name] = align_proba(decoder.predict_proba(data["X_val_raw"]), classes, data["X_val_raw"].shape[0], data["n_classes"])
        eval_probas[name] = align_proba(decoder.predict_proba(data["X_eval_raw"]), classes, data["X_eval_raw"].shape[0], data["n_classes"])
        audits[name] = {
            "fit_split": "train",
            "eval_used_for_fit": False,
            "n_train": int(data["X_train_raw"].shape[0]),
            "n_validation_predicted": int(data["X_val_raw"].shape[0]),
            "n_eval_predicted": int(data["X_eval_raw"].shape[0]),
            "classes": [int(c) for c in classes],
            "decoder": name,
            "feature_n_components": min(4, n_channels),
            "calibration": f"CalibratedClassifierCV(method={CALIBRATION_METHOD}, cv={CALIBRATION_CV}) inside train split",
        }
    train_vote = 0.5 * (train_probas["lda"] + train_probas["svm"])
    val_vote = 0.5 * (val_probas["lda"] + val_probas["svm"])
    eval_vote = 0.5 * (eval_probas["lda"] + eval_probas["svm"])
    return {
        "name": "lda_svm_vote",
        "train_proba": train_vote.astype(np.float32),
        "val_proba": val_vote.astype(np.float32),
        "eval_proba": eval_vote.astype(np.float32),
        "audit": {
            "teacher": "lda_svm_vote",
            "fit_split": "train",
            "validation_predictions_for_anchor_selection": True,
            "eval_predictions_used_as_targets": False,
            "eval_labels_used": False,
            "members": audits,
        },
    }


def normalize_proba(proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba, dtype=float)
    return proba / np.maximum(proba.sum(axis=1, keepdims=True), 1e-12)


def metrics_for(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    proba = normalize_proba(proba)
    scores = 1.0 - proba.max(axis=1)
    ece, _ = compute_ece(proba, y_true)
    sel_acc, coverage = compute_selective_acc(proba, y_true, scores, 0.6)
    return {
        "accuracy": float(np.mean(proba.argmax(axis=1) == y_true)),
        "ECE": float(ece),
        "Brier": compute_brier(proba, y_true),
        "sel_acc_60": float(sel_acc),
        "coverage_60": float(coverage),
    }


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    teacher_proba: np.ndarray | None,
    batch_size: int,
    balanced_sampler: bool,
    seed: int,
) -> DataLoader:
    tensors = [torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long)]
    if teacher_proba is not None:
        tensors.append(torch.as_tensor(teacher_proba, dtype=torch.float32))
    sampler = None
    shuffle = True
    if balanced_sampler:
        classes, counts = np.unique(y, return_counts=True)
        weights_by_class = {int(cls): 1.0 / float(count) for cls, count in zip(classes, counts)}
        weights = torch.as_tensor([weights_by_class[int(label)] for label in y], dtype=torch.double)
        generator = torch.Generator().manual_seed(seed)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=generator)
        shuffle = False
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle, sampler=sampler)


@torch.no_grad()
def collect_logits(model: torch.nn.Module, X: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    model.eval()
    chunks = []
    for start in range(0, X.shape[0], batch_size):
        xb = torch.as_tensor(X[start : start + batch_size], dtype=torch.float32, device=device)
        chunks.append(model(xb).detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def build_neural_model(X: np.ndarray, n_classes: int, variant: str, args: argparse.Namespace) -> torch.nn.Module:
    cfg = CompetenceConfig(
        n_channels=int(X.shape[1]),
        n_times=int(X.shape[2]),
        n_classes=int(n_classes),
        variant=RAW_VARIANT[variant],
        dropout=float(args.dropout),
    )
    return build_competence_model(cfg)


def neural_view(data: dict[str, Any], args: argparse.Namespace, teacher: dict[str, Any], variant: str) -> dict[str, Any]:
    mean, std = standardize_fit(data["X_train_raw"])
    X_train = apply_standardize(data["X_train_raw"], mean, std)
    X_val = apply_standardize(data["X_val_raw"], mean, std)
    X_eval = apply_standardize(data["X_eval_raw"], mean, std)
    anchored = variant in ANCHOR_TO_RAW
    if variant == "eegnet_ts":
        return {
            "X_train": X_train,
            "y_train": data["y_train"],
            "train_teacher": None,
            "X_val": X_val,
            "val_trial_ids": None,
            "X_eval": X_eval,
            "eval_trial_ids": None,
            "crop_audit": None,
            "preprocessing": {"standardization": {"fit_split": "train", "per_channel": True, "eval_used_for_fit": False}},
        }
    train_teacher = teacher["train_proba"] if anchored else None
    X_train_c, y_train_c, train_trial_ids, train_teacher_c, train_crop_audit = make_multiscale_crops(
        X_train, data["y_train"], args.crop_sizes, args.crop_strides, train_teacher
    )
    X_val_c, _, val_trial_ids, _, val_crop_audit = make_multiscale_crops(X_val, None, args.crop_sizes, args.crop_strides, None)
    X_eval_c, _, eval_trial_ids, _, eval_crop_audit = make_multiscale_crops(X_eval, None, args.crop_sizes, args.crop_strides, None)
    return {
        "X_train": X_train_c,
        "y_train": y_train_c,
        "train_trial_ids": train_trial_ids,
        "train_teacher": train_teacher_c,
        "X_val": X_val_c,
        "val_trial_ids": val_trial_ids,
        "X_eval": X_eval_c,
        "eval_trial_ids": eval_trial_ids,
        "crop_audit": {
            **train_crop_audit,
            "train_crops": int(X_train_c.shape[0]),
            "validation_crops": int(X_val_c.shape[0]),
            "eval_crops": int(X_eval_c.shape[0]),
            "validation_crop_audit": val_crop_audit,
            "eval_crop_audit": eval_crop_audit,
            "train_source": "external_train_split_only",
            "validation_aggregation": "mean_crop_logits_to_trial",
            "eval_aggregation": "mean_crop_logits_to_trial",
            "eval_labels_used_for_crop_selection": False,
        },
        "preprocessing": {"standardization": {"fit_split": "train", "per_channel": True, "eval_used_for_fit": False}},
    }


def trial_logits_from_model(model: torch.nn.Module, view: dict[str, Any], split: str, n_trials: int, device: str, args: argparse.Namespace) -> np.ndarray:
    logits = collect_logits(model, view[f"X_{split}"], device, args.batch_size)
    trial_ids = view.get(f"{split}_trial_ids")
    if trial_ids is None:
        return logits
    return aggregate_trial_logits(logits, trial_ids, n_trials)


def train_neural(data: dict[str, Any], args: argparse.Namespace, variant: str, teacher: dict[str, Any], device: str) -> dict[str, Any]:
    anchored = variant in ANCHOR_TO_RAW
    view = neural_view(data, args, teacher, variant)
    seed = int(args.seed + int(data["subject"]) * 997 + sum((i + 1) * ord(ch) for i, ch in enumerate(variant)) % 1000)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_neural_model(view["X_train"], data["n_classes"], variant, args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = make_loader(view["X_train"], view["y_train"], view["train_teacher"], args.batch_size, args.balanced_sampler, seed)
    loss_fn = torch.nn.CrossEntropyLoss()
    best_state = None
    best_val_loss = float("inf")
    best_val_brier = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        train_ce = []
        train_kd = []
        for batch in loader:
            xb = batch[0].to(device)
            if variant != "eegnet_ts":
                xb = augment_crops(xb, args)
            yb = batch[1].to(device)
            tb = batch[2].to(device) if anchored else None
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            ce = loss_fn(logits, yb)
            kd = distill_loss(logits, tb, args.distill_temperature) if tb is not None and args.lambda_distill > 0 else torch.zeros((), device=device)
            loss = ce + float(args.lambda_distill) * kd
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))
            train_ce.append(float(ce.detach().cpu().item()))
            train_kd.append(float(kd.detach().cpu().item()))

        val_logits = trial_logits_from_model(model, view, "val", data["y_val"].shape[0], device, args)
        logits_t = torch.as_tensor(val_logits, dtype=torch.float32)
        y_t = torch.as_tensor(data["y_val"], dtype=torch.long)
        val_loss_t = F.cross_entropy(logits_t, y_t)
        if anchored:
            val_loss_t = val_loss_t + float(args.lambda_distill) * distill_loss(
                logits_t,
                torch.as_tensor(teacher["val_proba"], dtype=torch.float32),
                args.distill_temperature,
            )
        val_metrics_raw = metrics_for(data["y_val"], proba_from_logits(val_logits))
        val_loss = float(val_loss_t.item())
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(np.mean(train_losses)),
                "train_ce_loss": float(np.mean(train_ce)),
                "train_distill_loss": float(np.mean(train_kd)),
                "val_loss": val_loss,
                "val_accuracy": float(val_metrics_raw["accuracy"]),
                "val_brier": float(val_metrics_raw["Brier"]),
            }
        )
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_val_brier = val_metrics_raw["Brier"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    val_logits = trial_logits_from_model(model, view, "val", data["y_val"].shape[0], device, args)
    eval_logits = trial_logits_from_model(model, view, "eval", data["y_eval"].shape[0], device, args)
    temperature, temp_at_upper = fit_temperature_from_logits(val_logits, data["y_val"], args.temperature_max)
    val_proba = proba_from_logits(val_logits, temperature)
    eval_proba = proba_from_logits(eval_logits, temperature)
    return {
        "dataset": data["dataset"],
        "dataset_label": data["dataset_label"],
        "subject": data["subject"],
        "model": variant,
        "anchored": bool(anchored),
        "teacher": "lda_svm_vote" if anchored else None,
        "tested_but_not_selected": bool(variant == OPTIONAL_MULTISCALE),
        "metrics": metrics_for(data["y_eval"], eval_proba),
        "validation_metrics": metrics_for(data["y_val"], val_proba),
        "model_info": {
            **view["preprocessing"],
            "crop_training": view["crop_audit"],
            "temperature_scaling": {
                "enabled": True,
                "fit_split": "validation_trial_logits",
                "eval_used_for_fit": False,
                "temperature": float(temperature),
                "temperature_at_upper_clamp": bool(temp_at_upper),
            },
            "distillation": {
                "enabled": bool(anchored),
                "teacher": "lda_svm_vote" if anchored else None,
                "lambda_distill": float(args.lambda_distill) if anchored else 0.0,
                "distill_temperature": float(args.distill_temperature) if anchored else None,
                "teacher_fit_split": "train" if anchored else None,
                "teacher_targets": "train_crops_repeated_from_train_trials" if anchored else None,
                "eval_teacher_targets_used": False,
            },
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss),
            "best_val_brier": float(best_val_brier),
            "epochs_ran": int(len(history)),
            "history": history,
            "chance_accuracy": float(data["chance_accuracy"]),
            "chance_brier": float(data["chance_brier"]),
        },
    }


def classical_row(data: dict[str, Any], teacher: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": data["dataset"],
        "dataset_label": data["dataset_label"],
        "subject": data["subject"],
        "model": "lda_svm_vote",
        "anchored": False,
        "teacher": None,
        "tested_but_not_selected": False,
        "metrics": metrics_for(data["y_eval"], teacher["eval_proba"]),
        "validation_metrics": metrics_for(data["y_val"], teacher["val_proba"]),
        "model_info": {
            "best_epoch": None,
            "best_val_loss": None,
            "best_val_brier": float(metrics_for(data["y_val"], teacher["val_proba"])["Brier"]),
            "temperature_scaling": {"enabled": False, "temperature": 1.0, "temperature_at_upper_clamp": False},
            "history": [],
            "chance_accuracy": float(data["chance_accuracy"]),
            "chance_brier": float(data["chance_brier"]),
            "teacher_audit": teacher["audit"],
        },
    }


def compute_sanity_flags(row: dict[str, Any], rows_by_key: dict[tuple[str, Any, str], dict[str, Any]]) -> list[str]:
    info = row["model_info"]
    chance_acc = float(info["chance_accuracy"])
    chance_brier = float(info["chance_brier"])
    flags = []
    if info.get("best_epoch") == 1:
        flags.append("best_epoch=1")
    history = info.get("history") or []
    if len(history) >= 2 and history[-1]["train_loss"] >= history[0]["train_loss"]:
        flags.append("train_loss_not_decreasing")
    if row["validation_metrics"]["accuracy"] <= chance_acc + 0.05:
        flags.append("validation_accuracy_at_chance")
    if row["validation_metrics"]["Brier"] >= chance_brier:
        flags.append("validation_brier_not_below_chance")
    if info.get("temperature_scaling", {}).get("temperature_at_upper_clamp"):
        flags.append("temperature_upper_clamp")
    key_prefix = (row["dataset"], row["subject"])
    eegnet = rows_by_key.get((*key_prefix, "eegnet_ts"))
    if row["model"] in {"raw_cropped_shallow_convnet_ce", "anchor_cropped_shallow_convnet_ce", OPTIONAL_MULTISCALE} and eegnet:
        if row["metrics"]["accuracy"] < eegnet["metrics"]["accuracy"]:
            flags.append("eval_accuracy_below_eegnet_ts")
    if row["model"] == "anchor_cropped_shallow_convnet_ce":
        raw = rows_by_key.get((*key_prefix, "raw_cropped_shallow_convnet_ce"))
        if raw and row["metrics"]["accuracy"] < raw["metrics"]["accuracy"] and row["metrics"]["Brier"] > raw["metrics"]["Brier"]:
            flags.append("anchor_worse_than_matched_raw_accuracy_and_brier")
    return flags


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for dataset in sorted({r["dataset"] for r in rows}):
        for model in sorted({r["model"] for r in rows if r["dataset"] == dataset}):
            selected = [r for r in rows if r["dataset"] == dataset and r["model"] == model]
            row: dict[str, Any] = {
                "dataset": dataset,
                "dataset_label": DATASET_LABELS.get(dataset, dataset),
                "model": model,
                "n": len(selected),
                "anchored": bool(selected[0]["anchored"]),
                "tested_but_not_selected": bool(selected[0].get("tested_but_not_selected", False)),
                "sanity_flag_count": int(sum(len(r.get("sanity_flags", [])) for r in selected)),
            }
            for metric in METRICS:
                values = np.asarray([r["metrics"][metric] for r in selected], dtype=float)
                row[metric] = {"mean": float(values.mean()), "std": float(values.std(ddof=0))}
            for metric, source in (("val_accuracy", "accuracy"), ("val_Brier", "Brier")):
                values = np.asarray([r["validation_metrics"][source] for r in selected], dtype=float)
                row[metric] = {"mean": float(values.mean()), "std": float(values.std(ddof=0))}
            for metric in ("best_epoch", "best_val_loss", "best_val_brier", "temperature"):
                vals = []
                for r in selected:
                    if metric == "temperature":
                        vals.append(float(r["model_info"]["temperature_scaling"]["temperature"]))
                    elif r["model_info"].get(metric) is not None:
                        vals.append(float(r["model_info"][metric]))
                arr = np.asarray(vals, dtype=float)
                row[metric] = {"mean": float(arr.mean()) if arr.size else None, "std": float(arr.std(ddof=0)) if arr.size else None}
            out.append(row)
    return out


def paired_values(rows: list[dict[str, Any]], dataset: str, model: str, control: str, metric: str) -> np.ndarray:
    by = {(r["dataset"], r["subject"], r["model"]): r for r in rows}
    values = []
    for r in rows:
        if r["dataset"] != dataset or r["model"] != model:
            continue
        c = by.get((dataset, r["subject"], control))
        if c:
            values.append(float(r["metrics"][metric] - c["metrics"][metric]))
    return np.asarray(values, dtype=float)


def sign_test_pvalue(values: np.ndarray, beneficial_positive: bool) -> float | None:
    nonzero = values[values != 0]
    n = int(nonzero.size)
    if n == 0:
        return None
    successes = int(np.sum(nonzero > 0)) if beneficial_positive else int(np.sum(nonzero < 0))
    tail = sum(math.comb(n, k) for k in range(0, min(successes, n - successes) + 1)) / (2**n)
    return float(min(1.0, 2.0 * tail))


def paired_test(values: np.ndarray, beneficial_positive: bool) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    out = {
        "n": int(values.size),
        "mean_delta": float(values.mean()) if values.size else None,
        "median_delta": float(np.median(values)) if values.size else None,
        "subjects_improved": int(np.sum(values > 0)) if beneficial_positive else int(np.sum(values < 0)),
        "pilot_only": bool(values.size < 9),
        "test": None,
        "p_value": None,
    }
    if values.size < 9:
        return out
    nonzero = values[values != 0]
    if nonzero.size < 2:
        return out
    try:
        from scipy.stats import wilcoxon

        result = wilcoxon(nonzero, alternative="greater" if beneficial_positive else "less", zero_method="wilcox")
        out["test"] = "wilcoxon_signed_rank"
        out["p_value"] = float(result.pvalue)
    except Exception:
        out["test"] = "sign_test"
        out["p_value"] = sign_test_pvalue(values, beneficial_positive)
    return out


def comparison_tables(rows: list[dict[str, Any]], include_multiscale: bool) -> dict[str, Any]:
    datasets = sorted({r["dataset"] for r in rows})
    models = ["raw_cropped_shallow_convnet_ce", "anchor_cropped_shallow_convnet_ce"]
    if include_multiscale:
        models.append(OPTIONAL_MULTISCALE)
    paired_vs_eegnet = []
    paired_vs_vote = []
    anchor_rows = []
    multiscale_rows = []
    evidence = []
    for dataset in datasets:
        for model in models:
            for metric, beneficial_positive in (("accuracy", True), ("Brier", False), ("ECE", False), ("sel_acc_60", True)):
                values = paired_values(rows, dataset, model, "eegnet_ts", metric)
                if values.size:
                    paired_vs_eegnet.append({"dataset": dataset, "model": model, "metric": metric, **paired_test(values, beneficial_positive)})
                values = paired_values(rows, dataset, model, "lda_svm_vote", metric)
                if values.size:
                    paired_vs_vote.append({"dataset": dataset, "model": model, "metric": metric, **paired_test(values, beneficial_positive)})
            if model == "anchor_cropped_shallow_convnet_ce":
                acc = paired_values(rows, dataset, model, "raw_cropped_shallow_convnet_ce", "accuracy")
                brier = paired_values(rows, dataset, model, "raw_cropped_shallow_convnet_ce", "Brier")
                ece = paired_values(rows, dataset, model, "raw_cropped_shallow_convnet_ce", "ECE")
                if acc.size:
                    anchor_rows.append(
                        {
                            "dataset": dataset,
                            "anchor_model": model,
                            "matched_raw": "raw_cropped_shallow_convnet_ce",
                            "n": int(acc.size),
                            "accuracy_delta_mean": float(acc.mean()),
                            "accuracy_delta_median": float(np.median(acc)),
                            "brier_delta_mean": float(brier.mean()),
                            "brier_delta_median": float(np.median(brier)),
                            "ece_delta_mean": float(ece.mean()),
                            "subjects_accuracy_improved": int(np.sum(acc > 0)),
                            "subjects_brier_improved": int(np.sum(brier < 0)),
                        }
                    )
                    evidence.append({"dataset": dataset, "comparison": "anchor_vs_raw", "model": model, "metric": "accuracy", **paired_test(acc, True)})
                    evidence.append({"dataset": dataset, "comparison": "anchor_vs_raw", "model": model, "metric": "Brier", **paired_test(brier, False)})
            if model == OPTIONAL_MULTISCALE:
                acc = paired_values(rows, dataset, model, "raw_cropped_shallow_convnet_ce", "accuracy")
                brier = paired_values(rows, dataset, model, "raw_cropped_shallow_convnet_ce", "Brier")
                if acc.size:
                    multiscale_rows.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "control": "raw_cropped_shallow_convnet_ce",
                            "status": "tested_but_not_selected",
                            "n": int(acc.size),
                            "accuracy_delta_mean": float(acc.mean()),
                            "accuracy_delta_median": float(np.median(acc)),
                            "brier_delta_mean": float(brier.mean()),
                            "brier_delta_median": float(np.median(brier)),
                            "subjects_accuracy_improved": int(np.sum(acc > 0)),
                            "subjects_brier_improved": int(np.sum(brier < 0)),
                        }
                    )
                    evidence.append({"dataset": dataset, "comparison": "multiscale_vs_raw_cropped", "model": model, "metric": "accuracy", **paired_test(acc, True)})
                    evidence.append({"dataset": dataset, "comparison": "multiscale_vs_raw_cropped", "model": model, "metric": "Brier", **paired_test(brier, False)})
    return {
        "paired_deltas_vs_eegnet_ts": paired_vs_eegnet,
        "paired_deltas_vs_lda_svm_vote": paired_vs_vote,
        "anchor_usefulness": anchor_rows,
        "multiscale_ablation": multiscale_rows,
        "significance_evidence": evidence,
    }


def decide_verdict(summary: list[dict[str, Any]], anchor_rows: list[dict[str, Any]]) -> tuple[str, str]:
    datasets = sorted({row["dataset"] for row in summary})
    dataset_support = []
    for dataset in datasets:
        rows = {row["model"]: row for row in summary if row["dataset"] == dataset}
        raw = rows.get("raw_cropped_shallow_convnet_ce")
        eegnet = rows.get("eegnet_ts")
        vote = rows.get("lda_svm_vote")
        anchor = rows.get("anchor_cropped_shallow_convnet_ce")
        usefulness = next((r for r in anchor_rows if r["dataset"] == dataset), None)
        if not raw or not eegnet or not vote or not anchor:
            dataset_support.append(False)
            continue
        ce_competent = raw["accuracy"]["mean"] >= eegnet["accuracy"]["mean"] - 0.02
        anchor_calibrates = bool(usefulness and (usefulness["brier_delta_mean"] < 0 or usefulness["ece_delta_mean"] < 0))
        not_systemic = anchor["sanity_flag_count"] < max(1, anchor["n"] * 2)
        dataset_support.append(bool(ce_competent and anchor_calibrates and not_systemic))
    if dataset_support and all(dataset_support):
        verdict = "externally_supported" if len(dataset_support) > 1 else "mixed_external_support"
    elif any(dataset_support):
        verdict = "mixed_external_support"
    elif datasets == ["BNCI2014_001"]:
        verdict = "bci2a_only_for_now"
    else:
        verdict = "failed_external"
    edl_verdict = "justified_later" if verdict == "externally_supported" else "not_yet"
    return verdict, edl_verdict


def write_csvs(results_dir: Path, rows: list[dict[str, Any]], summary: list[dict[str, Any]]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    subject_fields = [
        "dataset",
        "dataset_label",
        "subject",
        "model",
        "anchored",
        "tested_but_not_selected",
        "accuracy",
        "ECE",
        "Brier",
        "sel_acc_60",
        "coverage_60",
        "val_accuracy",
        "val_Brier",
        "best_epoch",
        "best_val_loss",
        "best_val_brier",
        "temperature",
        "temperature_at_upper_clamp",
        "sanity_flags",
    ]
    with (results_dir / "external_competence_anchor_subject_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=subject_fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["dataset"], str(r["subject"]), r["model"])):
            info = row["model_info"]
            writer.writerow(
                {
                    "dataset": row["dataset"],
                    "dataset_label": row["dataset_label"],
                    "subject": row["subject"],
                    "model": row["model"],
                    "anchored": row["anchored"],
                    "tested_but_not_selected": row.get("tested_but_not_selected", False),
                    **row["metrics"],
                    "val_accuracy": row["validation_metrics"]["accuracy"],
                    "val_Brier": row["validation_metrics"]["Brier"],
                    "best_epoch": "" if info.get("best_epoch") is None else info["best_epoch"],
                    "best_val_loss": "" if info.get("best_val_loss") is None else info["best_val_loss"],
                    "best_val_brier": "" if info.get("best_val_brier") is None else info["best_val_brier"],
                    "temperature": info["temperature_scaling"]["temperature"],
                    "temperature_at_upper_clamp": info["temperature_scaling"]["temperature_at_upper_clamp"],
                    "sanity_flags": ";".join(row.get("sanity_flags", [])),
                }
            )
    fields = ["dataset", "dataset_label", "model", "n", "anchored", "tested_but_not_selected"]
    for metric in (*METRICS, "val_accuracy", "val_Brier", "best_epoch", "best_val_loss", "best_val_brier", "temperature"):
        fields.extend([f"{metric}_mean", f"{metric}_std"])
    fields.append("sanity_flag_count")
    with (results_dir / "external_competence_anchor_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sorted(summary, key=lambda r: (r["dataset"], r["model"])):
            flat = {key: row[key] for key in ("dataset", "dataset_label", "model", "n", "anchored", "tested_but_not_selected")}
            for metric in (*METRICS, "val_accuracy", "val_Brier", "best_epoch", "best_val_loss", "best_val_brier", "temperature"):
                flat[f"{metric}_mean"] = row[metric]["mean"]
                flat[f"{metric}_std"] = row[metric]["std"]
            flat["sanity_flag_count"] = row["sanity_flag_count"]
            writer.writerow(flat)


def fmt_mean_std(row: dict[str, Any], metric: str) -> str:
    val = row[metric]["mean"]
    std = row[metric]["std"]
    if val is None:
        return ""
    return f"{val:.4f} ({std:.4f})"


def add_protocol_table(lines: list[str], protocols: list[dict[str, Any]]) -> None:
    lines.extend(["", "## Dataset Protocols", "", "| dataset | subject | train | validation | eval | rule |", "|---|---:|---:|---:|---:|---|"])
    for p in protocols:
        lines.append(
            f"| {p['dataset_label']} | {p['subject']} | {p['n_train']} | {p['n_validation']} | {p['n_eval']} | {p['validation_mask_rule']}={p['validation_value']} |"
        )


def add_delta_table(lines: list[str], title: str, rows: list[dict[str, Any]]) -> None:
    lines.extend(["", f"## {title}", "", "| dataset | model | metric | mean delta | median delta | improved | n | note |", "|---|---|---|---:|---:|---:|---:|---|"])
    if not rows:
        lines.append("| none | none | none |  |  |  |  | no paired rows |")
        return
    for row in rows:
        note = "pilot/external stress" if row["pilot_only"] else "paired external"
        lines.append(
            f"| {DATASET_LABELS.get(row['dataset'], row['dataset'])} | {row['model']} | {row['metric']} | "
            f"{row['mean_delta']:+.4f} | {row['median_delta']:+.4f} | {row['subjects_improved']} | {row['n']} | {note} |"
        )


def add_evidence_table(lines: list[str], rows: list[dict[str, Any]]) -> None:
    lines.extend(["", "## Significance / Evidence", "", "| dataset | comparison | model | metric | mean delta | median delta | improved | n | test | p | note |", "|---|---|---|---|---:|---:|---:|---:|---|---:|---|"])
    if not rows:
        lines.append("| none | none | none | none |  |  |  |  |  |  | no paired evidence |")
        return
    for row in rows:
        p_text = "" if row["p_value"] is None else f"{row['p_value']:.4f}"
        note = "pilot/external stress evidence" if row["pilot_only"] else "full paired evidence"
        lines.append(
            f"| {DATASET_LABELS.get(row['dataset'], row['dataset'])} | {row['comparison']} | {row['model']} | {row['metric']} | "
            f"{row['mean_delta']:+.4f} | {row['median_delta']:+.4f} | {row['subjects_improved']} | {row['n']} | {row['test'] or ''} | {p_text} | {note} |"
        )


def write_readout(results_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# External Competence-Anchor Readout",
        "",
        f"Verdict: **{payload['verdict']}**",
        f"EDL verdict: **{payload['edl_verdict']}**",
        "",
        "This is a competence/calibration-anchor validation, not a new-neural-architecture or SOTA accuracy claim.",
    ]
    cfg = payload.get("run_config", {})
    if cfg:
        lines.extend(
            [
                "",
                "## Run Configuration",
                "",
                f"- epochs={cfg.get('epochs')}, patience={cfg.get('patience')}, batch_size={cfg.get('batch_size')}",
                f"- crop_sizes={cfg.get('crop_sizes')}, crop_strides={cfg.get('crop_strides')}",
                f"- augment={cfg.get('augment')}, balanced_sampler={cfg.get('balanced_sampler')}",
                "- Bounded CPU run: treat as external stress evidence for the competence-anchor story, not a full-budget architecture result.",
            ]
        )
    add_protocol_table(lines, payload["dataset_protocols"])
    lines.extend(["", "## Aggregate Metrics", "", "| dataset | model | accuracy | ECE | Brier | sel_acc_60 | val acc | val Brier | flags |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in sorted(payload["metrics"], key=lambda r: (r["dataset"], r["model"])):
        model = row["model"] + (" (tested but not selected)" if row.get("tested_but_not_selected") else "")
        lines.append(
            f"| {row['dataset_label']} | {model} | {fmt_mean_std(row, 'accuracy')} | {fmt_mean_std(row, 'ECE')} | "
            f"{fmt_mean_std(row, 'Brier')} | {fmt_mean_std(row, 'sel_acc_60')} | {fmt_mean_std(row, 'val_accuracy')} | "
            f"{fmt_mean_std(row, 'val_Brier')} | {row['sanity_flag_count']} |"
        )
    add_delta_table(lines, "Paired Deltas vs eegnet_ts", payload["paired_deltas_vs_eegnet_ts"])
    add_delta_table(lines, "Paired Deltas vs lda_svm_vote", payload["paired_deltas_vs_lda_svm_vote"])
    lines.extend(["", "## Anchor Usefulness vs Raw Cropped Control", "", "| dataset | anchor | raw | acc delta | Brier delta | ECE delta | acc improved | Brier improved | n |", "|---|---|---|---:|---:|---:|---:|---:|---:|"])
    for row in payload["anchor_usefulness"]:
        lines.append(
            f"| {DATASET_LABELS.get(row['dataset'], row['dataset'])} | {row['anchor_model']} | {row['matched_raw']} | "
            f"{row['accuracy_delta_mean']:+.4f} | {row['brier_delta_mean']:+.4f} | {row['ece_delta_mean']:+.4f} | "
            f"{row['subjects_accuracy_improved']} | {row['subjects_brier_improved']} | {row['n']} |"
        )
    if payload["multiscale_ablation"]:
        lines.extend(["", "## Optional Multiscale Ablation", "", "| dataset | model | control | status | acc delta | Brier delta | acc improved | Brier improved | n |", "|---|---|---|---|---:|---:|---:|---:|---:|"])
        for row in payload["multiscale_ablation"]:
            lines.append(
                f"| {DATASET_LABELS.get(row['dataset'], row['dataset'])} | {row['model']} | {row['control']} | {row['status']} | "
                f"{row['accuracy_delta_mean']:+.4f} | {row['brier_delta_mean']:+.4f} | {row['subjects_accuracy_improved']} | {row['subjects_brier_improved']} | {row['n']} |"
            )
    add_evidence_table(lines, payload["significance_evidence"])
    lines.extend(["", "## Per-Subject Sanity Flags", ""])
    for key, flags in sorted(payload["sanity_flags_by_subject_model"].items()):
        lines.append(f"- {key}: {', '.join(flags) if flags else 'none'}")
    lines.extend(
        [
            "",
            "## Explicit Verdicts",
            "",
            f"- external_verdict: {payload['verdict']}",
            f"- edl_verdict: {payload['edl_verdict']}",
            "",
            "## Leakage Audit",
            "",
            "- Session/run structure is inspected before training.",
            "- Train/validation split is derived only inside explicit training sessions/runs.",
            "- Standardization is fit only on the train split.",
            "- Classical teachers are fit only on the train split.",
            "- Temperature scaling is fit only on validation trial-level logits.",
            "- Eval labels are used only for final metrics and sanity comparisons.",
            "- PhysioNet MI is skipped unless a leakage-safe protocol is explicitly added.",
        ]
    )
    (results_dir / "external_competence_anchor_readout.md").write_text("\n".join(lines), encoding="utf-8")


def selected_subjects(dataset_name: str, args: argparse.Namespace) -> list[Any]:
    dataset = dataset_instance(dataset_name, args.data_dir)
    subjects = list(getattr(dataset, "subject_list", []))
    if args.subject:
        requested = {str(s) for s in args.subject}
        subjects = [s for s in subjects if str(s) in requested]
    if args.subject_limit is not None:
        subjects = subjects[: args.subject_limit]
    return subjects


def print_structure(data: dict[str, Any]) -> None:
    p = data["protocol"]
    print(
        f"[protocol] dataset={data['dataset']} subject={data['subject']} "
        f"train={p['n_train']} validation={p['n_validation']} eval={p['n_eval']} "
        f"train_sessions={p['train_sessions']} eval_sessions={p['eval_sessions']} "
        f"validation={p['validation_mask_rule']}:{p['validation_value']}",
        flush=True,
    )
    structure = data["meta_structure"]
    print(f"[structure] sessions={structure.get('session', {}).get('values')} runs={structure.get('run', {}).get('values')}", flush=True)


def main() -> int:
    args = parse_args()
    datasets = selected_datasets(args)
    if "PhysionetMI" in datasets and args.include_physionet:
        print("[external_competence_anchor] PhysioNet MI requested but skipped: no predeclared leakage-safe cross-session/cross-run protocol.", flush=True)
        datasets = [d for d in datasets if d != "PhysionetMI"]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_names = list(MODEL_SET) + ([OPTIONAL_MULTISCALE] if args.include_multiscale else [])
    print(f"[external_competence_anchor] device={device} datasets={datasets} models={model_names}", flush=True)

    subject_rows: list[dict[str, Any]] = []
    subject_payloads = []
    protocols = []
    skipped = []
    for dataset_name in datasets:
        subjects = selected_subjects(dataset_name, args)
        print(f"[external_competence_anchor] dataset={dataset_name} subjects={subjects}", flush=True)
        for subject in subjects:
            try:
                data = prepare_subject_data(dataset_name, subject, args)
                print_structure(data)
                protocols.append({"dataset": dataset_name, "dataset_label": data["dataset_label"], "subject": subject, **data["protocol"]})
                if args.inspect_only:
                    continue
                teacher = fit_teacher(data)
                models: dict[str, Any] = {}
                row = classical_row(data, teacher)
                subject_rows.append(row)
                models["lda_svm_vote"] = row
                for model_name in model_names:
                    if model_name == "lda_svm_vote":
                        continue
                    print(f"[external_competence_anchor] dataset={dataset_name} subject={subject} model={model_name}", flush=True)
                    row = train_neural(data, args, model_name, teacher, device)
                    subject_rows.append(row)
                    models[model_name] = row
                subject_payloads.append(
                    {
                        "dataset": dataset_name,
                        "dataset_label": data["dataset_label"],
                        "subject": subject,
                        "classes": data["classes"],
                        "meta_structure": data["meta_structure"],
                        "protocol": data["protocol"],
                        "teacher_audit": teacher["audit"],
                        "models": models,
                    }
                )
            except Exception as exc:
                print(f"[external_competence_anchor] SKIP dataset={dataset_name} subject={subject}: {exc}", flush=True)
                skipped.append({"dataset": dataset_name, "subject": subject, "reason": str(exc)})

    if args.inspect_only:
        return 0

    rows_by_key = {(r["dataset"], r["subject"], r["model"]): r for r in subject_rows}
    for row in subject_rows:
        row["sanity_flags"] = compute_sanity_flags(row, rows_by_key)
    for payload in subject_payloads:
        for model_name, row in payload["models"].items():
            row["sanity_flags"] = rows_by_key[(row["dataset"], row["subject"], model_name)].get("sanity_flags", [])
        dataset_dir = args.results_dir / payload["dataset"]
        write_json(dataset_dir / f"subject_{payload['subject']}_audit.json", payload)

    summary = aggregate(subject_rows)
    comparisons = comparison_tables(subject_rows, args.include_multiscale)
    verdict, edl_verdict = decide_verdict(summary, comparisons["anchor_usefulness"])
    payload = {
        "protocol": "external_competence_anchor",
        "core_thesis": "Classical decoders as competence and calibration anchors for reliability-gated neural uncertainty in small-sample motor imagery decoding.",
        "models": model_names,
        "run_config": {
            "epochs": int(args.epochs),
            "patience": int(args.patience),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "dropout": float(args.dropout),
            "lambda_distill": float(args.lambda_distill),
            "distill_temperature": float(args.distill_temperature),
            "crop_sizes": [int(v) for v in args.crop_sizes],
            "crop_strides": [int(v) for v in args.crop_strides],
            "balanced_sampler": bool(args.balanced_sampler),
            "augment": bool(args.augment),
            "noise_std": float(args.noise_std) if args.augment else 0.0,
            "time_mask_frac": float(args.time_mask_frac) if args.augment else 0.0,
            "temperature_max": float(args.temperature_max),
            "seed": int(args.seed),
        },
        "datasets_requested": datasets,
        "datasets_run": sorted({r["dataset"] for r in subject_rows}),
        "skipped": skipped,
        "dataset_protocols": protocols,
        "metrics": summary,
        **comparisons,
        "sanity_flags_by_subject_model": {
            f"{r['dataset']}:S{r['subject']}:{r['model']}": r.get("sanity_flags", [])
            for r in subject_rows
        },
        "verdict": verdict,
        "edl_verdict": edl_verdict,
        "sanity_rules": {
            "best_epoch": "flag when == 1",
            "train_loss": "flag when final train loss is not below first train loss",
            "validation_accuracy": "flag when <= chance + 0.05",
            "validation_brier": "flag when >= chance Brier",
            "temperature": "flag when temperature hits upper clamp",
            "neural_vs_eegnet": "flag neural models below eegnet_ts eval accuracy",
            "anchor": "flag anchor if worse than matched raw cropped control on both accuracy and Brier",
        },
    }
    write_csvs(args.results_dir, subject_rows, summary)
    write_json(args.results_dir / "external_competence_anchor_summary.json", payload)
    write_readout(args.results_dir, payload)
    print(f"[external_competence_anchor] wrote {args.results_dir}", flush=True)
    print(f"[external_competence_anchor] verdict={verdict} edl_verdict={edl_verdict}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
