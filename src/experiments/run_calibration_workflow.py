#!/usr/bin/env python3
"""Validation-locked calibration/accuracy experiment suite for BCI IV-2a.

All model, aggregation, calibration, mixing, teacher, checkpoint, augmentation,
and architecture choices are made from the A0xT train/validation split only.
A0xE labels are read only after choices are frozen, for final held-out metrics.
Result filenames include ``validation_only_selection`` to make that discipline
auditable from artifacts as well as code.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import platform
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import CALIBRATION_CV, CALIBRATION_METHOD, CSP_N_COMPONENTS, FBCSP_BANDS, INCLUDE_BANDPOWER, SFREQ
from decode import build_decoder
from evaluate import compute_brier, compute_ece
from experiments.run_classical_anchor_competence import (
    N_SUBJECTS,
    apply_standardize,
    fit_temperature_from_logits,
    make_jsonable,
    prepare_subject,
    proba_from_logits,
    standardize_fit,
    write_json,
)
from experiments.run_cropped_anchor_competence import (
    alpha_grid,
    normalize_crop_strides,
    normalized_proba,
)
from experiments.run_primary import DEFAULT_BATCH_SIZE, DEFAULT_SEED
from fbcsp import FBCSPFeatures


EXPERIMENT_ORDER = (
    "baseline",
    "mdrm_t_ea",
    "ea_baseline",
    "ea_seed_ensemble",
    "ea_augmentation",
    "seed_ensemble",
    "crop_aggregation",
    "brier_loss",
    "calibration_grid",
    "teacher_student_mixture",
    "stronger_teacher",
    "swa_ema",
    "calibration_checkpoint",
    "augmentation",
    "architecture",
)
DATASETS = ("bci4_2a", "bci_iiia", "BNCI2014_001", "BNCI2014_004", "PhysionetMI")
EXTERNAL_CONFIRMATORY_EXPERIMENTS = ("baseline", "mdrm_t_ea", "seed_ensemble", "augmentation", "teacher_student_mixture")
SELECTION_SPLIT = "A0xT_validation"
HELDOUT_SPLIT = "A0xE_final_report_only"
BCI_IIIA_EVENT_ID = {
    "left_hand": 769,
    "right_hand": 770,
    "feet": 771,
    "tongue": 772,
}


@dataclass(frozen=True)
class ArchConfig:
    temporal_kernel: int = 64
    dropout: float = 0.5
    max_norm: float = 0.0
    temporal_filters: int = 16


@dataclass(frozen=True)
class AugmentConfig:
    time_shift: int = 0
    channel_dropout: float = 0.0
    noise_std: float = 0.0
    freq_mask_frac: float = 0.0


@dataclass(frozen=True)
class TrainConfig:
    seed: int
    epochs: int
    patience: int
    batch_size: int
    lr: float
    weight_decay: float
    balanced_sampler: bool
    cosine_lr: bool
    lambda_brier: float = 0.0
    arch: ArchConfig = ArchConfig()
    augment: AugmentConfig = AugmentConfig()
    checkpoint_rule: str = "val_loss"
    average_checkpoint: str = "none"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="bci4_2a")
    parser.add_argument("--subject", action="append", help="Subject id to run. May be supplied multiple times.")
    parser.add_argument("--subjects", nargs="+", help="Alternative multi-subject form.")
    parser.add_argument("--subject-limit", type=int, default=None, help="Limit discovered subjects for smoke/external runs.")
    parser.add_argument("--experiments", nargs="+", default=["baseline"], choices=EXPERIMENT_ORDER + ("all",))
    parser.add_argument(
        "--external-confirmatory",
        action="store_true",
        help="Run only the predeclared promising methods for external validation.",
    )
    parser.add_argument(
        "--allow-stratified-external-split",
        action="store_true",
        help="Allow deterministic stratified train/validation/eval splits when no leakage-safe session/run protocol is available.",
    )
    parser.add_argument("--inspect-only", action="store_true", help="Inspect dataset split protocols and write no model metrics.")
    parser.add_argument(
        "--merge-summaries",
        action="store_true",
        help="Merge validation_only_selection_*_subject_metrics.csv files under --results-dir and exit.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results") / "calibration_workflow")
    parser.add_argument("--crop-sizes", type=int, nargs="+", default=[384, 512, 640])
    parser.add_argument("--crop-strides", type=int, nargs="+", default=[128])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--device",
        default=None,
        help="Force a device. Default: cuda if available, else cpu. Pass 'mps' explicitly to use Apple GPU.",
    )
    parser.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_true",
        default=True,
        help="Pin kernel selection so a rerun on the same platform is exact. Default on.",
    )
    parser.add_argument(
        "--no-deterministic",
        dest="deterministic",
        action="store_false",
        help="Allow nondeterministic fast kernels. Faster, but not exactly reproducible.",
    )
    parser.add_argument("--temperature-max", type=float, default=20.0)
    parser.add_argument("--alpha-grid-step", type=float, default=0.05)
    parser.add_argument("--balanced-sampler", dest="balanced_sampler", action="store_true", default=True)
    parser.add_argument("--no-balanced-sampler", dest="balanced_sampler", action="store_false")
    parser.add_argument("--cosine-lr", dest="cosine_lr", action="store_true", default=True)
    parser.add_argument("--no-cosine-lr", dest="cosine_lr", action="store_false")
    parser.add_argument("--include-confusion", action="store_true")
    parser.add_argument(
        "--neural-euclidean-alignment",
        action="store_true",
        help="Enable train-only Euclidean Alignment preprocessing for ea_* neural experiment keys.",
    )
    parser.add_argument(
        "--save-probabilities",
        action="store_true",
        help="Save validation/heldout per-trial probabilities for reliability diagrams.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress logs except final artifact path.")
    parser.add_argument("--log-epochs", type=int, default=0, help="If positive, print train/validation progress every N epochs.")
    parser.add_argument("--dry-run", action="store_true", help="Run one subject, one seed, one epoch, and tiny grids for smoke testing.")
    args = parser.parse_args()
    if len(args.crop_strides) not in (1, len(args.crop_sizes)):
        parser.error("--crop-strides must have length 1 or match --crop-sizes")
    if "all" in args.experiments:
        args.experiments = list(EXPERIMENT_ORDER)
    if args.external_confirmatory:
        args.experiments = list(EXTERNAL_CONFIRMATORY_EXPERIMENTS)
    if args.neural_euclidean_alignment and not any(exp.startswith("ea_") for exp in args.experiments):
        args.experiments = [*args.experiments, "ea_seed_ensemble"]
    if args.dry_run:
        args.experiments = ["baseline"] if args.experiments == ["baseline"] else args.experiments
        args.epochs = min(args.epochs, 1)
        args.patience = 1
        args.seeds = args.seeds[:1]
        args.crop_sizes = args.crop_sizes[:1]
        args.crop_strides = args.crop_strides[:1]
        if not args.subject and not args.subjects:
            args.subject_limit = 1
    if not 0.0 < float(args.alpha_grid_step) <= 1.0:
        parser.error("--alpha-grid-step must be in (0, 1]")
    return args


def progress(args: argparse.Namespace, message: str) -> None:
    if not getattr(args, "quiet", False):
        print(message, flush=True)


def _coerce_subject(dataset: str, value: Any) -> Any:
    text = str(value)
    if dataset in {"bci4_2a", "BNCI2014_001", "BNCI2014_004", "PhysionetMI"} and text.isdigit():
        return int(text)
    return text


def _discover_bci_iiia_subjects(args: argparse.Namespace) -> list[str]:
    data_dir = args.data_dir / "BCICIV_3a_gdf" if args.data_dir.name != "BCICIV_3a_gdf" else args.data_dir
    subjects = sorted(path.stem for path in data_dir.glob("*.gdf"))
    if not subjects:
        raise FileNotFoundError(f"No BCI IIIa .gdf files found in {data_dir}")
    return subjects


def _prepare_moabb_env(data_dir: Path) -> None:
    fake_home = REPO_ROOT / ".mne_home"
    mpl_dir = REPO_ROOT / ".mplconfig"
    (fake_home / ".mne").mkdir(parents=True, exist_ok=True)
    mpl_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(fake_home))
    os.environ.setdefault("MNE_HOME", str(fake_home))
    os.environ.setdefault("MNE_DATA", str(data_dir))
    os.environ.setdefault("MNE_LOGGING_LEVEL", "WARNING")
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))


def _import_moabb(data_dir: Path):
    _prepare_moabb_env(data_dir)
    try:
        import moabb
        from moabb import datasets as moabb_datasets
        from moabb.paradigms import LeftRightImagery, MotorImagery
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("MOABB is required for MOABB external validation. Install `moabb`.") from exc
    moabb.set_log_level("warning")
    return moabb_datasets, {"LeftRightImagery": LeftRightImagery, "MotorImagery": MotorImagery}


def _moabb_dataset_instance(dataset_name: str, data_dir: Path):
    moabb_datasets, _ = _import_moabb(data_dir)
    try:
        return getattr(moabb_datasets, dataset_name)()
    except AttributeError as exc:
        raise ValueError(f"Unknown MOABB dataset: {dataset_name}") from exc


def _discover_moabb_subjects(args: argparse.Namespace) -> list[Any]:
    dataset = _moabb_dataset_instance(args.dataset, args.data_dir / "moabb")
    return list(getattr(dataset, "subject_list", []))


def resolve_subjects(args: argparse.Namespace) -> list[Any]:
    subjects: list[Any] = []
    if args.subject:
        subjects.extend(args.subject)
    if args.subjects:
        subjects.extend(args.subjects)
    if subjects:
        out = [_coerce_subject(args.dataset, subject) for subject in subjects]
    elif args.dataset == "bci4_2a":
        out = list(range(1, N_SUBJECTS + 1))
    elif args.dataset == "bci_iiia":
        out = _discover_bci_iiia_subjects(args)
    else:
        out = _discover_moabb_subjects(args)
    if args.subject_limit is not None:
        out = out[: int(args.subject_limit)]
    if args.dataset == "bci4_2a":
        invalid = [subject for subject in out if not isinstance(subject, int) or subject < 1 or subject > N_SUBJECTS]
        if invalid:
            raise ValueError(f"BCI IV-2a subjects must be integers 1..{N_SUBJECTS}; got {invalid}")
    return list(dict.fromkeys(out))


def set_all_seeds(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def configure_determinism(enabled: bool) -> None:
    """Pin kernel selection so a rerun on the same platform is exact.

    Bit-identical results across different backends are not achievable and are
    not the goal: floating-point addition is not associative, and CPU, CUDA and
    MPS use different kernels with different reduction orders. What this buys is
    exact reproducibility for anyone rerunning on the same platform and build,
    which is the claim ``reproducibility_manifest.json`` actually makes. Report
    cross-platform differences as a measured quantity instead
    (``src/scripts/compare_platform_runs.py``).
    """
    if not enabled:
        torch.backends.cudnn.benchmark = True
        return
    # cuBLAS reads this when it creates its workspace, on the first CUDA matmul.
    # Setting it here is early enough; setting it after that point is not.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as exc:  # pragma: no cover - backend dependent
        # Some ops have no deterministic kernel on some builds. Degrade loudly
        # rather than silently reporting a determinism guarantee we do not have.
        print(f"warning: deterministic algorithms unavailable ({exc})", flush=True)


def platform_fingerprint(device: str) -> dict[str, Any]:
    """Environment facts needed to reproduce a run exactly.

    Recorded in every summary JSON so results produced on different machines
    are comparable rather than silently conflated.
    """
    out: dict[str, Any] = {
        "device": str(device),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    if torch.cuda.is_available():
        out["cuda"] = torch.version.cuda
        out["cudnn"] = torch.backends.cudnn.version()
        out["gpu"] = torch.cuda.get_device_name(0)
    return out


def dataset_slug(dataset: str) -> str:
    return dataset.lower().replace("-", "_")


def safe_subject_id(subject: Any) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(subject))


def chance_brier(n_classes: int) -> float:
    return float((int(n_classes) - 1.0) / int(n_classes))


def class_counts(y: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(y, return_counts=True)
    return {str(v): int(c) for v, c in zip(values, counts)}


def _split_train_val_indices(y: np.ndarray, seed: int, val_size: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(y.size)
    classes, counts = np.unique(y, return_counts=True)
    stratify = y if classes.size > 1 and counts.min() >= 2 else None
    test_size: float | int = float(val_size)
    if stratify is not None:
        proposed = int(np.ceil(float(val_size) * y.size))
        test_size = min(max(proposed, classes.size), y.size - classes.size)
        if test_size < classes.size:
            stratify = None
            test_size = float(val_size)
    return train_test_split(idx, test_size=test_size, random_state=int(seed), stratify=stratify)


def _split_stratified_three_way(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(y.size)
    train_val_idx, eval_idx = train_test_split(idx, test_size=0.2, random_state=int(seed), stratify=y)
    rel_train_idx, rel_val_idx = _split_train_val_indices(y[train_val_idx], int(seed) + 17, val_size=0.25)
    return train_val_idx[rel_train_idx], train_val_idx[rel_val_idx], eval_idx


def encode_split_labels(
    y_train_raw: np.ndarray,
    y_val_raw: np.ndarray,
    y_eval_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train_raw).astype(np.int64)
    y_val = encoder.transform(y_val_raw).astype(np.int64)
    y_eval = encoder.transform(y_eval_raw).astype(np.int64)
    return y_train, y_val, y_eval, [str(c) for c in encoder.classes_]


def _complete_class_splits(data: dict[str, Any]) -> None:
    n_classes = int(data["n_classes"])
    for split_name, key in (("train", "y_train"), ("validation", "y_val"), ("eval", "y_eval")):
        if np.unique(data[key]).size != n_classes:
            raise ValueError(f"{data['dataset']} subject {data['subject']}: {split_name} split lacks all {n_classes} classes.")


def _enrich_prepared_data(data: dict[str, Any]) -> dict[str, Any]:
    n_classes = int(data.get("n_classes", len(data["classes"])))
    data["n_classes"] = n_classes
    data["chance_accuracy"] = float(1.0 / n_classes)
    data["chance_Brier"] = chance_brier(n_classes)
    data.setdefault("selection_split", SELECTION_SPLIT)
    data.setdefault("heldout_split", HELDOUT_SPLIT)
    data.setdefault("train_split", "A0xT_train")
    data.setdefault("protocol", {})
    data["protocol"] = {
        "n_classes": n_classes,
        "classes": data["classes"],
        "chance_accuracy": data["chance_accuracy"],
        "chance_Brier": data["chance_Brier"],
        "n_train": int(data["X_train_raw"].shape[0]),
        "n_validation": int(data["X_val_raw"].shape[0]),
        "n_eval": int(data["X_eval_raw"].shape[0]),
        "train_class_counts": class_counts(data["y_train"]),
        "validation_class_counts": class_counts(data["y_val"]),
        "eval_class_counts": class_counts(data["y_eval"]),
        "eval_labels_used_for_selection": False,
        **data["protocol"],
    }
    return data


def prepare_bci4_2a_subject(subject: int, args: argparse.Namespace) -> dict[str, Any]:
    data = prepare_subject(int(subject), args.data_dir, args.seed)
    data.update(
        {
            "dataset": "bci4_2a",
            "dataset_label": "BCI_IV_2a",
            "n_classes": len(data["classes"]),
            "selection_split": SELECTION_SPLIT,
            "heldout_split": HELDOUT_SPLIT,
            "train_split": "A0xT_train",
            "protocol": {
                "protocol_name": "A0xT_train_validation_to_A0xE",
                "fit_split": "A0xT_train",
                "validation_split": SELECTION_SPLIT,
                "heldout_split": HELDOUT_SPLIT,
                "eval_labels_used_for": "final_metrics_only",
                "eval_labels_used_for_selection": False,
                "stratified_validation_split": True,
            },
        }
    )
    return _enrich_prepared_data(data)


def prepare_bci_iiia_subject(subject: str, args: argparse.Namespace) -> dict[str, Any]:
    _prepare_moabb_env(args.data_dir / "moabb")
    try:
        import mne
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("MNE is required to load BCI IIIa GDF files.") from exc

    data_dir = args.data_dir / "BCICIV_3a_gdf" if args.data_dir.name != "BCICIV_3a_gdf" else args.data_dir
    gdf_path = data_dir / f"{subject}.gdf"
    if not gdf_path.exists():
        raise FileNotFoundError(f"Missing BCI IIIa GDF file: {gdf_path}")

    raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose=False)
    raw.pick("eeg")
    raw.filter(
        8.0,
        30.0,
        method="iir",
        iir_params={"order": 5, "ftype": "butter"},
        verbose=False,
    )
    events, annotation_ids = mne.events_from_annotations(raw, verbose=False)
    event_id = {
        name: annotation_ids[str(code)]
        for name, code in BCI_IIIA_EVENT_ID.items()
        if str(code) in annotation_ids
    }
    if len(event_id) < 2:
        raise ValueError(f"{subject}: expected at least two motor-imagery event classes, found {event_id}")

    cue_events = events[np.isin(events[:, 2], list(event_id.values()))]
    name_by_event = {value: name for name, value in event_id.items()}
    labels = np.asarray(
        [BCI_IIIA_EVENT_ID[name_by_event[int(event)]] for event in cue_events[:, 2]],
        dtype=int,
    )
    epochs = mne.Epochs(
        raw,
        cue_events,
        event_id,
        tmin=0.0,
        tmax=4.5,
        baseline=None,
        preload=True,
        event_repeated="drop",
        verbose=False,
    )
    X = epochs.get_data(copy=True).astype(np.float32)
    y = labels[epochs.selection]
    dev_idx, eval_idx = train_test_split(
        np.arange(y.size),
        test_size=0.35,
        random_state=int(args.seed),
        stratify=y,
    )
    rel_train_idx, rel_val_idx = _split_train_val_indices(y[dev_idx], int(args.seed) + 31, val_size=0.2)
    train_idx = dev_idx[rel_train_idx]
    val_idx = dev_idx[rel_val_idx]

    y_train, y_val, y_eval, classes = encode_split_labels(y[train_idx], y[val_idx], y[eval_idx])
    data = {
        "dataset": "bci_iiia",
        "dataset_label": "BCI_IIIa",
        "subject": str(subject),
        "classes": classes,
        "X_train_raw": X[train_idx],
        "y_train": y_train,
        "X_val_raw": X[val_idx],
        "y_val": y_val,
        "X_eval_raw": X[eval_idx],
        "y_eval": y_eval,
        "paths": {"gdf": str(gdf_path)},
        "split": {
            "fit_split": "stratified_trial_split_external_train",
            "validation_split": "stratified_trial_split_external_validation",
            "eval_split": "stratified_trial_split_external_eval",
            "eval_labels_used_for": "final_metrics_only",
            "stratified": True,
        },
        "selection_split": "stratified_trial_split_external_validation",
        "heldout_split": "stratified_trial_split_external_eval_final_report_only",
        "train_split": "stratified_trial_split_external_train",
        "protocol": {
            "protocol_name": "stratified_trial_split_external",
            "source_split": "single_labeled_gdf_stratified_trial_split",
            "split_seed": int(args.seed),
            "validation_rule": "deterministic stratified split from non-eval development trials",
            "heldout_rule": "deterministic stratified held-out trials from the single labeled GDF",
            "eval_labels_used_for": "final_metrics_only",
            "eval_labels_used_for_selection": False,
            "allow_stratified_external_split": True,
        },
    }
    data["n_classes"] = len(classes)
    _complete_class_splits(data)
    return _enrich_prepared_data(data)


def _make_moabb_paradigm(dataset_name: str, data_dir: Path):
    _, paradigms = _import_moabb(data_dir)
    if dataset_name == "BNCI2014_001":
        cls = paradigms["MotorImagery"]
        try:
            return cls(n_classes=4, fmin=8.0, fmax=30.0, resample=250.0)
        except TypeError:
            return cls(fmin=8.0, fmax=30.0, resample=250.0)
    cls = paradigms["LeftRightImagery"]
    return cls(fmin=8.0, fmax=30.0, resample=250.0)


def ordered_unique(values: pd.Series) -> list[Any]:
    return list(pd.Series(values).dropna().drop_duplicates())


def session_sort_key(value: Any) -> tuple[int, str]:
    text = str(value)
    digits = "".join(ch for ch in text if ch.isdigit())
    return (int(digits) if digits else 10_000, text)


def structure_summary(meta: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {"columns": list(meta.columns), "n_trials": int(len(meta))}
    for column in ("session", "run"):
        if column in meta.columns:
            counts = meta.groupby(column, dropna=False).size().to_dict()
            out[column] = {
                "values": [str(v) for v in ordered_unique(meta[column])],
                "counts": {str(k): int(v) for k, v in counts.items()},
            }
    if {"session", "run"}.issubset(meta.columns):
        counts = meta.groupby(["session", "run"], dropna=False).size().reset_index(name="n")
        out["session_run_counts"] = [
            {"session": str(row["session"]), "run": str(row["run"]), "n": int(row["n"])}
            for _, row in counts.iterrows()
        ]
    return out


def moabb_protocol_masks(
    meta: pd.DataFrame,
    y: np.ndarray,
    dataset_name: str,
    seed: int,
    allow_stratified: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if "session" in meta.columns:
        session_text = meta["session"].astype(str).str.lower()
        train_sessions = ordered_unique(meta.loc[session_text.str.contains("train"), "session"])
        eval_sessions = ordered_unique(meta.loc[session_text.str.contains("test"), "session"])
        if train_sessions and eval_sessions:
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
            elif allow_stratified:
                dev_idx = np.flatnonzero(train_eval_mask)
                rel_train, rel_val = _split_train_val_indices(y[dev_idx], int(seed) + 43, val_size=0.2)
                train_mask = np.zeros(len(meta), dtype=bool)
                val_mask = np.zeros(len(meta), dtype=bool)
                train_mask[dev_idx[rel_train]] = True
                val_mask[dev_idx[rel_val]] = True
                validation_rule = "stratified_trial_split_external_validation_from_explicit_train_session"
                validation_value = "deterministic_seeded_split"
            else:
                raise ValueError(
                    f"{dataset_name}: explicit train/test sessions exist, but the train session has no separate session/run "
                    "to hold out for validation. Re-run with --allow-stratified-external-split to use a deterministic "
                    "stratified validation split inside the training session."
                )
            if not train_mask.any() or not val_mask.any() or not eval_mask.any():
                raise ValueError(f"{dataset_name}: empty train/validation/eval mask after protocol selection.")
            return train_mask, val_mask, eval_mask, {
                "protocol_name": "explicit_moabb_train_validation_eval",
                "train_sessions": [str(v) for v in train_sessions],
                "eval_sessions": [str(v) for v in eval_sessions],
                "validation_rule": validation_rule,
                "validation_value": validation_value,
                "train_eval_protocol": "explicit MOABB train sessions to explicit MOABB test sessions",
                "eval_labels_used_for": "final_metrics_only",
                "eval_labels_used_for_selection": False,
                "allow_stratified_external_split": bool(allow_stratified),
            }

    if not allow_stratified:
        extra = " PhysionetMI is skipped by default because its MOABB metadata does not provide a predeclared train/test protocol here." if dataset_name == "PhysionetMI" else ""
        raise ValueError(
            f"{dataset_name}: no leakage-safe explicit train/test session protocol was found.{extra} "
            "Use --inspect-only to review metadata, or --allow-stratified-external-split to force a deterministic "
            "stratified external split."
        )

    idx = np.arange(len(meta))
    train_idx, val_idx, eval_idx = _split_stratified_three_way(y[idx], int(seed))
    train_mask = np.zeros(len(meta), dtype=bool)
    val_mask = np.zeros(len(meta), dtype=bool)
    eval_mask = np.zeros(len(meta), dtype=bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    eval_mask[eval_idx] = True
    return train_mask, val_mask, eval_mask, {
        "protocol_name": "stratified_trial_split_external",
        "validation_rule": "deterministic stratified validation split",
        "heldout_rule": "deterministic stratified held-out eval split",
        "split_seed": int(seed),
        "eval_labels_used_for": "final_metrics_only",
        "eval_labels_used_for_selection": False,
        "allow_stratified_external_split": True,
    }


def prepare_moabb_subject(subject: Any, args: argparse.Namespace) -> dict[str, Any]:
    data_dir = args.data_dir / "moabb"
    dataset = _moabb_dataset_instance(args.dataset, data_dir)
    paradigm = _make_moabb_paradigm(args.dataset, data_dir)
    try:
        X, y, meta = paradigm.get_data(dataset=dataset, subjects=[subject])
    except Exception as exc:
        raise RuntimeError(
            f"{args.dataset} subject {subject}: MOABB data could not be loaded. If the dataset is not already cached "
            f"under {data_dir}, a network download may be required. Original error: {exc!r}"
        ) from exc
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    meta = pd.DataFrame(meta).reset_index(drop=True)
    train_mask, val_mask, eval_mask, protocol = moabb_protocol_masks(
        meta,
        y,
        args.dataset,
        int(args.seed),
        bool(args.allow_stratified_external_split),
    )
    y_train, y_val, y_eval, classes = encode_split_labels(y[train_mask], y[val_mask], y[eval_mask])
    data = {
        "dataset": args.dataset,
        "dataset_label": args.dataset,
        "subject": subject,
        "classes": classes,
        "X_train_raw": X[train_mask],
        "y_train": y_train,
        "X_val_raw": X[val_mask],
        "y_val": y_val,
        "X_eval_raw": X[eval_mask],
        "y_eval": y_eval,
        "paths": {"moabb_data_dir": str(data_dir)},
        "split": {
            "fit_split": "external_train",
            "validation_split": "external_validation",
            "eval_split": "external_eval_final_report_only",
            "eval_labels_used_for": "final_metrics_only",
        },
        "selection_split": "external_validation",
        "heldout_split": "external_eval_final_report_only",
        "train_split": "external_train",
        "meta_structure": structure_summary(meta),
        "protocol": {
            **protocol,
            "meta_structure": structure_summary(meta),
        },
    }
    data["n_classes"] = len(classes)
    _complete_class_splits(data)
    return _enrich_prepared_data(data)


def prepare_dataset_subject(subject: Any, args: argparse.Namespace) -> dict[str, Any]:
    if args.dataset == "bci4_2a":
        return prepare_bci4_2a_subject(int(subject), args)
    if args.dataset == "bci_iiia":
        return prepare_bci_iiia_subject(str(subject), args)
    return prepare_moabb_subject(subject, args)


def split_name(data: dict[str, Any], key: str, default: str) -> str:
    return str(data.get(key, default))


def align_proba_dynamic(p: np.ndarray, classes: np.ndarray, n_rows: int, n_classes: int) -> np.ndarray:
    aligned = np.zeros((n_rows, int(n_classes)), dtype=np.float64)
    for j, cls in enumerate(classes):
        idx = int(cls)
        if 0 <= idx < int(n_classes):
            aligned[:, idx] = p[:, j]
    denom = aligned.sum(axis=1, keepdims=True)
    denom = np.where(denom <= 0.0, 1.0, denom)
    return aligned / denom


class CroppedShallowConvNet(nn.Module):
    """Shallow cropped ConvNet with a tiny predeclared architecture grid."""

    def __init__(self, n_channels: int, n_times: int, n_classes: int, arch: ArchConfig):
        super().__init__()
        filters = max(4, int(arch.temporal_filters))
        kernel = max(3, min(int(arch.temporal_kernel), int(n_times)))
        if kernel % 2 == 0:
            kernel += 1
        self.max_norm = float(arch.max_norm)
        self.features = nn.Sequential(
            nn.Conv2d(1, filters, kernel_size=(1, kernel), padding=(0, kernel // 2), bias=False),
            nn.Conv2d(filters, filters, kernel_size=(n_channels, 1), bias=False),
            nn.BatchNorm2d(filters),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(float(arch.dropout)),
            nn.Conv2d(filters, filters, kernel_size=(1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(filters),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(float(arch.dropout)),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(nn.Dropout(float(arch.dropout)), nn.Linear(filters, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        return self.classifier(self.features(x))

    @torch.no_grad()
    def apply_max_norm(self) -> None:
        if self.max_norm <= 0:
            return
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                w = module.weight.data
                flat = w.view(w.shape[0], -1)
                norms = flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
                desired = torch.clamp(norms, max=self.max_norm)
                flat.mul_(desired / norms)


def make_loader(X: np.ndarray, y: np.ndarray, cfg: TrainConfig) -> DataLoader:
    tensors = [torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long)]
    sampler = None
    shuffle = True
    if cfg.balanced_sampler:
        classes, counts = np.unique(y, return_counts=True)
        weights_by_class = {int(cls): 1.0 / float(count) for cls, count in zip(classes, counts)}
        weights = torch.as_tensor([weights_by_class[int(label)] for label in y], dtype=torch.double)
        generator = torch.Generator().manual_seed(int(cfg.seed))
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=generator)
        shuffle = False
    return DataLoader(TensorDataset(*tensors), batch_size=int(cfg.batch_size), shuffle=shuffle, sampler=sampler)


def crop_starts_from_reference(n_times: int, crop_size: int, crop_stride: int) -> list[int]:
    if crop_size <= 0 or crop_size >= n_times:
        return [0]
    last = n_times - crop_size
    starts = list(range(0, last + 1, int(crop_stride)))
    if starts[-1] != last:
        starts.append(last)
    return starts


def pad_or_trim_crop(crop: np.ndarray, target_size: int) -> np.ndarray:
    out = np.zeros((crop.shape[0], target_size), dtype=np.float32)
    width = min(int(crop.shape[1]), int(target_size))
    out[:, :width] = crop[:, :width]
    return out


def make_fixed_multiscale_crops(
    X: np.ndarray,
    y: np.ndarray | None,
    crop_specs: list[dict[str, Any]],
    target_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create crops using A0xT-derived starts/effective sizes for every split."""

    X = np.asarray(X, dtype=np.float32)
    labels_source = np.zeros(X.shape[0], dtype=np.int64) if y is None else np.asarray(y, dtype=np.int64)
    crops = []
    labels = []
    trial_ids = []
    for trial_id in range(X.shape[0]):
        for spec in crop_specs:
            size = int(spec["effective_crop_size"])
            for start in spec["starts"]:
                stop = min(int(start) + size, X.shape[2])
                crops.append(pad_or_trim_crop(X[trial_id, :, int(start) : stop], target_size))
                labels.append(labels_source[trial_id])
                trial_ids.append(trial_id)
    return (
        np.stack(crops, axis=0).astype(np.float32, copy=False),
        np.asarray(labels, dtype=np.int64),
        np.asarray(trial_ids, dtype=np.int64),
    )


def preprocess_and_crop(data: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    mean, std = standardize_fit(data["X_train_raw"])
    X_train = apply_standardize(data["X_train_raw"], mean, std)
    X_val = apply_standardize(data["X_val_raw"], mean, std)
    X_eval = apply_standardize(data["X_eval_raw"], mean, std)
    strides = normalize_crop_strides(args.crop_sizes, args.crop_strides)
    reference_n_times = int(X_train.shape[2])
    crop_specs = []
    for crop_size, stride in zip(args.crop_sizes, strides):
        effective_size = reference_n_times if int(crop_size) >= reference_n_times else int(crop_size)
        starts = crop_starts_from_reference(reference_n_times, effective_size, int(stride))
        crop_specs.append(
            {
                "requested_crop_size": int(crop_size),
                "effective_crop_size": int(effective_size),
                "crop_stride": int(stride),
                "starts": [int(s) for s in starts],
                "crops_per_trial": int(len(starts)),
            }
        )
    target_size = max(int(spec["effective_crop_size"]) for spec in crop_specs)
    X_train_c, y_train_c, train_trial_ids = make_fixed_multiscale_crops(X_train, data["y_train"], crop_specs, target_size)
    X_val_c, _, val_trial_ids = make_fixed_multiscale_crops(X_val, None, crop_specs, target_size)
    X_eval_c, _, eval_trial_ids = make_fixed_multiscale_crops(X_eval, None, crop_specs, target_size)
    return {
        "X_train_crops": X_train_c,
        "y_train_crops": y_train_c,
        "train_trial_ids": train_trial_ids,
        "X_val_crops": X_val_c,
        "val_trial_ids": val_trial_ids,
        "X_eval_crops": X_eval_c,
        "eval_trial_ids": eval_trial_ids,
        "audit": {
            "crop_sizes": [int(s) for s in args.crop_sizes],
            "crop_strides": [int(s) for s in strides],
            "scales": crop_specs,
            "padded_to_n_times": int(target_size),
            "crops_per_trial": int(sum(spec["crops_per_trial"] for spec in crop_specs)),
            "reference_split": split_name(data, "train_split", "A0xT_train"),
            "reference_n_times": reference_n_times,
            "validation_n_times": int(X_val.shape[2]),
            "eval_n_times_report_only": int(X_eval.shape[2]),
            "crop_dimension_source": f"{split_name(data, 'train_split', 'A0xT_train')}_dimensions_and_predeclared_cli_grid_only",
            "crop_strides_normalized": [int(s) for s in strides],
            "eval_labels_used_for_crop_or_aggregation_selection": False,
        },
    }


def _symmetrize(mat: np.ndarray) -> np.ndarray:
    return 0.5 * (mat + mat.T)


def _regularize_spd(mat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mat = _symmetrize(np.asarray(mat, dtype=float))
    scale = float(np.trace(mat)) / max(mat.shape[0], 1)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return _symmetrize(mat + eps * scale * np.eye(mat.shape[0]))


def _matrix_inv_sqrt_spd(mat: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    evals, evecs = np.linalg.eigh(_symmetrize(mat))
    evals = np.maximum(evals, eps)
    return _symmetrize((evecs * (1.0 / np.sqrt(evals))) @ evecs.T)


def _trial_covariances(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return np.matmul(X, np.swapaxes(X, -1, -2)) / max(int(X.shape[-1]) - 1, 1)


def _apply_ea_matrix(X: np.ndarray, ea_inv_sqrt: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return (ea_inv_sqrt[None, :, :] @ X).astype(np.float32, copy=False)


def apply_train_only_euclidean_alignment(data: dict[str, Any]) -> dict[str, Any]:
    train_cov = np.asarray([_regularize_spd(cov) for cov in _trial_covariances(data["X_train_raw"])])
    mean_cov = _regularize_spd(train_cov.mean(axis=0))
    ea_inv_sqrt = _matrix_inv_sqrt_spd(mean_cov)
    out = copy.deepcopy(data)
    out["X_train_raw"] = _apply_ea_matrix(data["X_train_raw"], ea_inv_sqrt)
    out["X_val_raw"] = _apply_ea_matrix(data["X_val_raw"], ea_inv_sqrt)
    out["X_eval_raw"] = _apply_ea_matrix(data["X_eval_raw"], ea_inv_sqrt)
    out["euclidean_alignment"] = {
        "used": True,
        "fit_split": split_name(data, "train_split", "A0xT_train"),
        "eval_used_for_fit": False,
        "transform_splits": [
            split_name(data, "train_split", "A0xT_train"),
            split_name(data, "selection_split", SELECTION_SPLIT),
            split_name(data, "heldout_split", HELDOUT_SPLIT),
        ],
        "mean_covariance_shape": [int(v) for v in mean_cov.shape],
    }
    out["protocol"] = {
        **out.get("protocol", {}),
        "euclidean_alignment": out["euclidean_alignment"],
    }
    return out


def augment_batch(x: torch.Tensor, cfg: AugmentConfig) -> torch.Tensor:
    out = x
    if cfg.time_shift > 0:
        out = out.clone()
        shifts = torch.randint(-int(cfg.time_shift), int(cfg.time_shift) + 1, (out.shape[0],), device=out.device)
        for i, shift in enumerate(shifts.tolist()):
            out[i] = torch.roll(out[i], shifts=int(shift), dims=-1)
    if cfg.channel_dropout > 0:
        keep = (torch.rand(out.shape[0], out.shape[1], 1, device=out.device) >= float(cfg.channel_dropout)).to(out.dtype)
        out = out * keep
    if cfg.noise_std > 0:
        out = out + torch.randn_like(out) * float(cfg.noise_std)
    if cfg.freq_mask_frac > 0:
        fft = torch.fft.rfft(out, dim=-1)
        n_freq = fft.shape[-1]
        width = max(1, int(round(n_freq * min(float(cfg.freq_mask_frac), 0.5))))
        if width < n_freq:
            starts = torch.randint(0, n_freq - width + 1, (out.shape[0],), device=out.device)
            fft = fft.clone()
            for i, start in enumerate(starts.tolist()):
                fft[i, :, start : start + width] = 0
            out = torch.fft.irfft(fft, n=out.shape[-1], dim=-1).to(out.dtype)
    return out


@torch.no_grad()
def collect_crop_logits(model: nn.Module, X: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    model.eval()
    chunks = []
    for start in range(0, X.shape[0], batch_size):
        xb = torch.as_tensor(X[start : start + batch_size], dtype=torch.float32, device=device)
        chunks.append(model(xb).detach().cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z -= z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)


def nll_from_proba(proba: np.ndarray, y: np.ndarray) -> float:
    p = normalized_proba(proba)
    y = np.asarray(y, dtype=np.int64)
    return float(-np.mean(np.log(np.clip(p[np.arange(y.size), y], 1e-12, 1.0)))) if y.size else float("nan")


def per_trial_brier(proba: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = normalized_proba(proba)
    labels = np.asarray(y, dtype=np.int64)
    oh = np.zeros_like(p)
    oh[np.arange(labels.size), labels] = 1.0
    return np.sum((p - oh) ** 2, axis=1)


def metric_bundle(y: np.ndarray, proba: np.ndarray, include_confusion: bool = False) -> dict[str, Any]:
    p = normalized_proba(proba)
    pred = p.argmax(axis=1)
    ece, _ = compute_ece(p, y)
    n_classes = int(p.shape[1])
    out: dict[str, Any] = {
        "accuracy": float(np.mean(pred == y)),
        "Brier": compute_brier(p, y),
        "ECE": float(ece),
        "NLL": nll_from_proba(p, y),
        "chance_accuracy": float(1.0 / n_classes),
        "chance_Brier": chance_brier(n_classes),
    }
    if include_confusion:
        confusion = np.zeros((n_classes, n_classes), dtype=int)
        for true, guessed in zip(np.asarray(y, dtype=int), pred):
            confusion[int(true), int(guessed)] += 1
        out["confusion_matrix"] = confusion.tolist()
    return out


def metrics_from_logits(y: np.ndarray, logits: np.ndarray, temperature: float = 1.0, include_confusion: bool = False) -> dict[str, Any]:
    return metric_bundle(y, proba_from_logits(logits, temperature), include_confusion=include_confusion)


def aggregate_crop_logits(crop_logits: np.ndarray, trial_ids: np.ndarray, n_trials: int, method: str) -> np.ndarray:
    out = np.zeros((n_trials, crop_logits.shape[1]), dtype=np.float64)
    for trial in range(n_trials):
        rows = np.asarray(crop_logits[trial_ids == trial], dtype=np.float64)
        if rows.size == 0:
            continue
        if method == "mean_logits":
            agg = rows.mean(axis=0)
        elif method == "median_logits":
            agg = np.median(rows, axis=0)
        elif method == "trimmed_mean_logits":
            if rows.shape[0] >= 5:
                sorted_rows = np.sort(rows, axis=0)
                trim = max(1, int(math.floor(0.1 * rows.shape[0])))
                agg = sorted_rows[trim:-trim].mean(axis=0) if sorted_rows.shape[0] > 2 * trim else rows.mean(axis=0)
            else:
                agg = rows.mean(axis=0)
        elif method == "confidence_weighted_logits":
            crop_p = softmax_np(rows)
            weights = crop_p.max(axis=1)
            agg = np.average(rows, axis=0, weights=np.clip(weights, 1e-8, None))
        else:
            raise ValueError(f"Unknown crop aggregation method: {method}")
        out[trial] = agg
    return out.astype(np.float32)


def brier_loss_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    proba = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(labels, num_classes=logits.shape[1]).to(dtype=proba.dtype)
    return torch.sum((proba - one_hot) ** 2, dim=1).mean()


def validation_checkpoint_choice(history: list[dict[str, Any]], rule: str) -> dict[str, Any]:
    if rule == "val_loss":
        return min(history, key=lambda row: (row["val_loss"], -row["val_accuracy"], row["epoch"]))
    best_brier = min(float(row["val_brier"]) for row in history)
    best_row = min(history, key=lambda row: float(row["val_brier"]))
    threshold = best_brier + float(best_row["val_brier_se"])
    candidates = [row for row in history if float(row["val_brier"]) <= threshold]
    return max(candidates, key=lambda row: (row["val_accuracy"], -row["val_brier"], -row["val_loss"], -row["epoch"]))


def update_ema(ema: dict[str, torch.Tensor] | None, model: nn.Module, decay: float) -> dict[str, torch.Tensor]:
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if ema is None:
        return state
    for key, value in state.items():
        if torch.is_floating_point(value):
            ema[key].mul_(float(decay)).add_(value, alpha=1.0 - float(decay))
        else:
            ema[key] = value
    return ema


def train_student(
    view: dict[str, Any],
    data: dict[str, Any],
    cfg: TrainConfig,
    device: str,
    log_fn: Callable[[str], None] | None = None,
    log_epochs: int = 0,
) -> dict[str, Any]:
    set_all_seeds(cfg.seed)
    model = CroppedShallowConvNet(
        n_channels=int(view["X_train_crops"].shape[1]),
        n_times=int(view["X_train_crops"].shape[2]),
        n_classes=int(data["n_classes"]),
        arch=cfg.arch,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(cfg.epochs))) if cfg.cosine_lr else None
    loader = make_loader(view["X_train_crops"], view["y_train_crops"], cfg)
    best_state: dict[str, torch.Tensor] | None = None
    checkpoint_states: dict[int, dict[str, torch.Tensor]] = {}
    best_val_loss = float("inf")
    stale = 0
    history: list[dict[str, Any]] = []
    ema_state: dict[str, torch.Tensor] | None = None
    ema_start = max(1, int(round(cfg.epochs * 0.75)))

    for epoch in range(1, int(cfg.epochs) + 1):
        model.train()
        losses = []
        ce_losses = []
        brier_losses = []
        for xb, yb in loader:
            xb = augment_batch(xb.to(device), cfg.augment)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            ce = F.cross_entropy(logits, yb)
            brier = brier_loss_from_logits(logits, yb)
            loss = ce + float(cfg.lambda_brier) * brier
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            model.apply_max_norm()
            losses.append(float(loss.detach().cpu().item()))
            ce_losses.append(float(ce.detach().cpu().item()))
            brier_losses.append(float(brier.detach().cpu().item()))
        if scheduler is not None:
            scheduler.step()
        if cfg.average_checkpoint == "ema" and epoch >= ema_start:
            ema_state = update_ema(ema_state, model, decay=0.95)

        val_crop_logits = collect_crop_logits(model, view["X_val_crops"], device, cfg.batch_size)
        val_logits = aggregate_crop_logits(val_crop_logits, view["val_trial_ids"], data["y_val"].shape[0], "mean_logits")
        val_proba = proba_from_logits(val_logits)
        val_loss = float(F.cross_entropy(torch.as_tensor(val_logits), torch.as_tensor(data["y_val"], dtype=torch.long)).item())
        val_metrics = metric_bundle(data["y_val"], val_proba)
        trial_briers = per_trial_brier(val_proba, data["y_val"])
        row = {
            "epoch": int(epoch),
            "train_loss": float(np.mean(losses)),
            "train_ce_loss": float(np.mean(ce_losses)),
            "train_brier_loss": float(np.mean(brier_losses)),
            "val_loss": val_loss,
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_brier": float(val_metrics["Brier"]),
            "val_brier_se": float(np.std(trial_briers, ddof=1) / math.sqrt(max(1, trial_briers.size))) if trial_briers.size > 1 else 0.0,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        if log_fn is not None and int(log_epochs) > 0 and cfg.epochs > 1:
            interval = int(log_epochs)
            if epoch == 1 or epoch % interval == 0 or epoch == int(cfg.epochs):
                log_fn(
                    f"    epoch {epoch:03d}/{int(cfg.epochs):03d} "
                    f"loss={row['train_loss']:.4f} val_acc={row['val_accuracy']:.3f} val_brier={row['val_brier']:.4f}"
                )
        checkpoint_states[epoch] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = checkpoint_states[epoch]
            stale = 0
        else:
            stale += 1
            if stale >= int(cfg.patience):
                break

    chosen = validation_checkpoint_choice(history, cfg.checkpoint_rule)
    if cfg.checkpoint_rule == "calibration_aware":
        model.load_state_dict(checkpoint_states[int(chosen["epoch"])])
    elif best_state is not None:
        model.load_state_dict(best_state)
    if cfg.average_checkpoint == "ema" and ema_state is not None:
        model.load_state_dict(ema_state)
        chosen = {**chosen, "average_checkpoint_applied": "ema", "ema_start_epoch": int(ema_start)}

    val_crop_logits = collect_crop_logits(model, view["X_val_crops"], device, cfg.batch_size)
    eval_crop_logits = collect_crop_logits(model, view["X_eval_crops"], device, cfg.batch_size)
    return {
        "seed": int(cfg.seed),
        "cfg": cfg,
        "val_crop_logits": val_crop_logits,
        "eval_crop_logits": eval_crop_logits,
        "history": history,
        "chosen_checkpoint": chosen,
        "audit": {
            "random_seed": int(cfg.seed),
            "loss": {"ce": True, "lambda_brier": float(cfg.lambda_brier)},
            "checkpoint_rule": cfg.checkpoint_rule,
            "checkpoint_selection_split": split_name(data, "selection_split", SELECTION_SPLIT),
            "checkpoint_eval_labels_used": False,
            "average_checkpoint": cfg.average_checkpoint,
            "augmentation": cfg.augment.__dict__,
            "architecture": cfg.arch.__dict__,
        },
    }


class VectorTemperature:
    def __init__(self, upper: float):
        self.upper = float(upper)
        self.temperatures: np.ndarray | None = None

    def fit(self, logits: np.ndarray, y: np.ndarray) -> "VectorTemperature":
        x = torch.as_tensor(logits, dtype=torch.float32)
        labels = torch.as_tensor(y, dtype=torch.long)
        log_t = torch.zeros(x.shape[1], requires_grad=True)
        optimizer = torch.optim.LBFGS([log_t], lr=0.05, max_iter=300, line_search_fn="strong_wolfe")

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            temp = torch.exp(log_t).clamp(0.05, self.upper)
            loss = F.cross_entropy(x / temp, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperatures = torch.exp(log_t).detach().clamp(0.05, self.upper).cpu().numpy()
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        if self.temperatures is None:
            raise RuntimeError("VectorTemperature is not fit")
        return softmax_np(np.asarray(logits, dtype=np.float64) / self.temperatures[None, :])


class DirichletCalibration:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, multi_class="auto", C=1.0)

    def fit(self, logits: np.ndarray, y: np.ndarray) -> "DirichletCalibration":
        features = np.log(np.clip(softmax_np(logits), 1e-8, 1.0))
        self.model.fit(features, y)
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        features = np.log(np.clip(softmax_np(logits), 1e-8, 1.0))
        return normalized_proba(self.model.predict_proba(features))

    def params(self) -> dict[str, Any]:
        return {"coef": self.model.coef_.tolist(), "intercept": self.model.intercept_.tolist()}


def calibration_grid(val_logits: np.ndarray, eval_logits: np.ndarray, data: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    scalar_t, scalar_at_upper = fit_temperature_from_logits(val_logits, data["y_val"], args.temperature_max)
    scalar_val = proba_from_logits(val_logits, scalar_t)
    scalar_eval = proba_from_logits(eval_logits, scalar_t)
    candidates.append(
        {
            "method": "scalar_temperature",
            "val_proba": scalar_val,
            "eval_proba": scalar_eval,
            "params": {"temperature": float(scalar_t), "temperature_at_upper_clamp": bool(scalar_at_upper)},
        }
    )
    vector = VectorTemperature(args.temperature_max).fit(val_logits, data["y_val"])
    candidates.append(
        {
            "method": "vector_temperature",
            "val_proba": vector.predict_proba(val_logits),
            "eval_proba": vector.predict_proba(eval_logits),
            "params": {"temperatures": vector.temperatures.tolist() if vector.temperatures is not None else []},
        }
    )
    try:
        diri = DirichletCalibration().fit(val_logits, data["y_val"])
        candidates.append(
            {
                "method": "dirichlet_calibration",
                "val_proba": diri.predict_proba(val_logits),
                "eval_proba": diri.predict_proba(eval_logits),
                "params": diri.params(),
            }
        )
    except Exception as exc:
        candidates.append({"method": "dirichlet_calibration", "skipped": True, "reason": repr(exc)})

    records = []
    scalar_record = None
    for candidate in candidates:
        if candidate.get("skipped"):
            records.append(candidate)
            continue
        val_metrics = metric_bundle(data["y_val"], candidate["val_proba"], include_confusion=False)
        eval_metrics = metric_bundle(data["y_eval"], candidate["eval_proba"], include_confusion=args.include_confusion)
        record = {
            "method": candidate["method"],
            "params": candidate["params"],
            "validation_metrics": val_metrics,
            "heldout_metrics": eval_metrics,
            "fit_split": split_name(data, "selection_split", SELECTION_SPLIT),
            "eval_labels_used_for_selection": False,
        }
        records.append(record)
        if candidate["method"] == "scalar_temperature":
            scalar_record = record
    assert scalar_record is not None
    selected = scalar_record
    for record in records:
        if record.get("skipped"):
            continue
        if (
            float(record["validation_metrics"]["Brier"]) < float(selected["validation_metrics"]["Brier"]) - 1e-12
            and float(record["validation_metrics"]["accuracy"]) >= float(scalar_record["validation_metrics"]["accuracy"])
        ):
            selected = record
    selected_candidate = next(
        candidate
        for candidate in candidates
        if not candidate.get("skipped") and candidate["method"] == selected["method"]
    )
    return {
        "selected": selected,
        "selected_proba": {
            "val_proba": selected_candidate["val_proba"],
            "eval_proba": selected_candidate["eval_proba"],
        },
        "candidates": records,
        "default": "scalar_temperature",
    }


def fit_pipeline_teacher(name: str, data: dict[str, Any]) -> dict[str, Any]:
    n_classes = int(data["n_classes"])
    train_split = split_name(data, "train_split", "A0xT_train")
    selection_split = split_name(data, "selection_split", SELECTION_SPLIT)
    if name == "lda_svm_vote":
        lda = fit_pipeline_teacher("fbcsp_lda", data)
        svm = build_decoder("SVM")
        svm.fit(data["X_train_raw"], data["y_train"])
        svm_classes = getattr(svm, "classes_", np.arange(n_classes))
        svm_train = align_proba_dynamic(svm.predict_proba(data["X_train_raw"]), svm_classes, data["X_train_raw"].shape[0], n_classes)
        svm_val = align_proba_dynamic(svm.predict_proba(data["X_val_raw"]), svm_classes, data["X_val_raw"].shape[0], n_classes)
        svm_eval = align_proba_dynamic(svm.predict_proba(data["X_eval_raw"]), svm_classes, data["X_eval_raw"].shape[0], n_classes)
        return {
            "name": name,
            "train_proba": normalized_proba(0.5 * (lda["train_proba"] + svm_train)),
            "val_proba": normalized_proba(0.5 * (lda["val_proba"] + svm_val)),
            "eval_proba": normalized_proba(0.5 * (lda["eval_proba"] + svm_eval)),
            "audit": {"fit_split": train_split, "selection_split": selection_split, "eval_labels_used_for_selection": False},
        }
    if name == "fbcsp_lda":
        model = build_decoder("LDA")
    elif name == "shrinkage_lda":
        model = CalibratedClassifierCV(
            Pipeline(
                [
                    ("feat", FBCSPFeatures(sfreq=SFREQ, bands=FBCSP_BANDS, n_components=CSP_N_COMPONENTS, include_bandpower=INCLUDE_BANDPOWER)),
                    ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
                ]
            ),
            method=CALIBRATION_METHOD,
            cv=CALIBRATION_CV,
        )
    elif name == "tangent_logreg":
        try:
            from pyriemann.estimation import Covariances
            from pyriemann.tangentspace import TangentSpace
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("pyRiemann is required for tangent-space logistic regression.") from exc
        model = Pipeline(
            [
                ("cov", Covariances(estimator="oas")),
                ("ts", TangentSpace(metric="riemann")),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, C=1.0)),
            ]
        )
    elif name == "mdrm_t":
        from baselines.mdrm_t import MDRMT

        model = MDRMT()
        model.fit(data["X_train_raw"], data["y_train"], data["X_val_raw"], data["y_val"])
        return {
            "name": name,
            "train_proba": normalized_proba(model.predict_proba(data["X_train_raw"])),
            "val_proba": normalized_proba(model.predict_proba(data["X_val_raw"])),
            "eval_proba": normalized_proba(model.predict_proba(data["X_eval_raw"])),
            "audit": {"fit_split": train_split, "temperature_split": selection_split, "eval_labels_used_for_selection": False},
        }
    else:
        raise ValueError(f"Unknown teacher candidate: {name}")

    model.fit(data["X_train_raw"], data["y_train"])
    fitted_classes = getattr(model, "classes_", np.arange(n_classes))
    return {
        "name": name,
        "train_proba": align_proba_dynamic(model.predict_proba(data["X_train_raw"]), fitted_classes, data["X_train_raw"].shape[0], n_classes),
        "val_proba": align_proba_dynamic(model.predict_proba(data["X_val_raw"]), fitted_classes, data["X_val_raw"].shape[0], n_classes),
        "eval_proba": align_proba_dynamic(model.predict_proba(data["X_eval_raw"]), fitted_classes, data["X_eval_raw"].shape[0], n_classes),
        "audit": {"fit_split": train_split, "selection_split": selection_split, "eval_labels_used_for_selection": False},
    }


def select_teacher(data: dict[str, Any], candidates: list[str]) -> dict[str, Any]:
    records = []
    for name in candidates:
        try:
            teacher = fit_pipeline_teacher(name, data)
            record = {
                "name": name,
                "teacher": teacher,
                "validation_metrics": metric_bundle(data["y_val"], teacher["val_proba"]),
                "heldout_metrics_for_report_only": metric_bundle(data["y_eval"], teacher["eval_proba"]),
                "skipped": False,
            }
        except Exception as exc:
            record = {"name": name, "skipped": True, "reason": repr(exc)}
        records.append(record)
    valid = [row for row in records if not row.get("skipped")]
    if not valid:
        raise RuntimeError("No teacher candidates fit successfully")
    selected = min(valid, key=lambda row: (row["validation_metrics"]["Brier"], -row["validation_metrics"]["accuracy"], row["name"]))
    return {"selected": selected, "candidates": records}


def fit_alpha(student_val: np.ndarray, teacher_val: np.ndarray, y_val: np.ndarray, step: float) -> tuple[float, list[dict[str, float]]]:
    student_acc = float(metric_bundle(y_val, student_val)["accuracy"])
    rows = []
    for alpha in alpha_grid(step):
        mix = normalized_proba(float(alpha) * student_val + (1.0 - float(alpha)) * teacher_val)
        metrics = metric_bundle(y_val, mix)
        rows.append({"alpha": float(alpha), "validation_Brier": metrics["Brier"], "validation_accuracy": metrics["accuracy"]})
    guarded = [row for row in rows if row["validation_accuracy"] >= student_acc]
    pool = guarded or rows
    best = min(pool, key=lambda row: (row["validation_Brier"], -row["validation_accuracy"], abs(1.0 - row["alpha"])))
    return float(best["alpha"]), rows


def run_mdrm_t_ea(data: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    from baselines.mdrm_t import MDRMT, audit_leakage

    model = MDRMT(temperature_bounds=(0.05, float(args.temperature_max)), coverage_targets=(0.40, 0.60, 0.80, 0.90))
    model.fit(data["X_train_raw"], data["y_train"], data["X_val_raw"], data["y_val"])
    val_proba = align_proba_dynamic(
        model.predict_proba(data["X_val_raw"]),
        np.asarray(model.classes_),
        data["X_val_raw"].shape[0],
        int(data["n_classes"]),
    )
    eval_proba = align_proba_dynamic(
        model.predict_proba(data["X_eval_raw"]),
        np.asarray(model.classes_),
        data["X_eval_raw"].shape[0],
        int(data["n_classes"]),
    )
    eval_report = model.evaluate(data["X_eval_raw"], data["y_eval"])
    audit_leakage(model)
    threshold_rows = [
        {
            "target_coverage": float(target),
            "threshold": float(threshold),
            "source_split": split_name(data, "selection_split", SELECTION_SPLIT),
            "eval_labels_used_for_threshold": False,
        }
        for target, threshold in sorted(model.abstention_thresholds_.items())
    ]
    return {
        "run": {
            "experiment": "mdrm_t_train_only_euclidean_alignment",
            "validation_metrics": metric_bundle(data["y_val"], val_proba),
            "heldout_metrics": metric_bundle(data["y_eval"], eval_proba, include_confusion=args.include_confusion),
            "selection_audit": {
                "fit_split": split_name(data, "train_split", "A0xT_train"),
                "selection_split": split_name(data, "selection_split", SELECTION_SPLIT),
                "validation_split": split_name(data, "selection_split", SELECTION_SPLIT),
                "heldout_split": split_name(data, "heldout_split", HELDOUT_SPLIT),
                "eval_labels_used_for_selection": False,
            },
            "calibration": {
                "method": "scalar_temperature",
                "temperature": float(model.temperature_),
                "fit_split": split_name(data, "selection_split", SELECTION_SPLIT),
                "temperature_fit_source": split_name(data, "selection_split", SELECTION_SPLIT),
                "eval_labels_used_for_fit": False,
            },
            "metadata": {
                "model": "MDRM-T",
                "pyriemann_classifier": "MDM",
                "metric": model.metric,
                "classes": [int(c) if isinstance(c, (np.integer, int)) else str(c) for c in np.asarray(model.classes_)],
                "euclidean_alignment": {
                    "used": True,
                    "fit_split": split_name(data, "train_split", "A0xT_train"),
                    "fit_source": split_name(data, "train_split", "A0xT_train"),
                    "eval_used_for_fit": False,
                },
                "abstention_thresholds": threshold_rows,
                "abstention_threshold_source": split_name(data, "selection_split", SELECTION_SPLIT),
                "eval_report": {k: v for k, v in eval_report.items() if k != "risk_coverage_data"},
                "risk_coverage_data": eval_report.get("risk_coverage_data", []),
                "audit_fields": {
                    "fit_split": split_name(data, "train_split", "A0xT_train"),
                    "validation_split": split_name(data, "selection_split", SELECTION_SPLIT),
                    "heldout_split": split_name(data, "heldout_split", HELDOUT_SPLIT),
                    "eval_labels_used_for_selection": False,
                    "ea_fit_source": split_name(data, "train_split", "A0xT_train"),
                    "temperature_fit_source": split_name(data, "selection_split", SELECTION_SPLIT),
                    "abstention_threshold_source": split_name(data, "selection_split", SELECTION_SPLIT),
                },
            },
        },
        "val_proba": val_proba,
        "eval_proba": eval_proba,
    }


def run_seed_model(
    view: dict[str, Any],
    data: dict[str, Any],
    args: argparse.Namespace,
    seed: int,
    log_fn: Callable[[str], None] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    base_arch = ArchConfig(dropout=float(args.dropout))
    cfg = TrainConfig(
        seed=int(seed),
        epochs=int(args.epochs),
        patience=int(args.patience),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        balanced_sampler=bool(args.balanced_sampler),
        cosine_lr=bool(args.cosine_lr),
        arch=base_arch,
    )
    for key, value in overrides.items():
        cfg = replace(cfg, **{key: value})
    return train_student(view, data, cfg, args.device, log_fn=log_fn, log_epochs=int(args.log_epochs))


def freeze_eval(
    name: str,
    val_logits: np.ndarray,
    eval_logits: np.ndarray,
    data: dict[str, Any],
    args: argparse.Namespace,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw_val_metrics = metrics_from_logits(data["y_val"], val_logits, include_confusion=False)
    raw_eval_metrics = metrics_from_logits(data["y_eval"], eval_logits, include_confusion=args.include_confusion)
    temperature, at_upper = fit_temperature_from_logits(val_logits, data["y_val"], args.temperature_max)
    ts_val_metrics = metrics_from_logits(data["y_val"], val_logits, temperature, include_confusion=False)
    ts_eval_metrics = metrics_from_logits(data["y_eval"], eval_logits, temperature, include_confusion=args.include_confusion)
    return {
        "experiment": name,
        "validation_metrics_raw": raw_val_metrics,
        "heldout_metrics_raw": raw_eval_metrics,
        "validation_metrics": ts_val_metrics,
        "heldout_metrics": ts_eval_metrics,
        "calibration": {
            "method": "scalar_temperature",
            "temperature": float(temperature),
            "temperature_at_upper_clamp": bool(at_upper),
            "fit_split": split_name(data, "selection_split", SELECTION_SPLIT),
            "eval_labels_used_for_fit": False,
        },
        "selection_audit": {
            "selection_split": split_name(data, "selection_split", SELECTION_SPLIT),
            "heldout_split": split_name(data, "heldout_split", HELDOUT_SPLIT),
            "eval_labels_used_for_selection": False,
        },
        "metadata": metadata or {},
    }


def _probability_output_path(args: argparse.Namespace, dataset: str, subject: Any) -> Path:
    slug = dataset_slug(dataset)
    subject_id = safe_subject_id(subject)
    return args.results_dir / "probabilities" / f"validation_only_selection_{slug}_subject{subject_id}_probabilities.npz"


def save_probability_artifact(
    args: argparse.Namespace,
    data: dict[str, Any],
    runs: dict[str, Any],
    probabilities: dict[str, dict[str, np.ndarray]],
) -> None:
    if not probabilities:
        return
    path = _probability_output_path(args, data["dataset"], data["subject"])
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "y_val": np.asarray(data["y_val"], dtype=np.int64),
        "y_eval": np.asarray(data["y_eval"], dtype=np.int64),
        "classes": np.asarray(data["classes"], dtype=str),
        "metadata_json": np.asarray(
            json.dumps(
                make_jsonable(
                    {
                        "dataset": data["dataset"],
                        "dataset_label": data.get("dataset_label", data["dataset"]),
                        "subject": data["subject"],
                        "n_classes": int(data["n_classes"]),
                        "split": data["split"],
                        "protocol": data["protocol"],
                        "validation_discipline": {
                            "selection_split": split_name(data, "selection_split", SELECTION_SPLIT),
                            "heldout_split": split_name(data, "heldout_split", HELDOUT_SPLIT),
                            "eval_labels_used_for_selection": False,
                        },
                        "experiments": {
                            key: {
                                "experiment": value.get("experiment", key),
                                "validation_metrics": value.get("validation_metrics", {}),
                                "heldout_metrics": value.get("heldout_metrics", {}),
                                "selection_audit": value.get("selection_audit", {}),
                            }
                            for key, value in runs.items()
                            if key in probabilities
                        },
                    }
                )
            )
        ),
    }
    for key, record in probabilities.items():
        safe_key = safe_subject_id(key)
        arrays[f"{safe_key}__val_proba"] = np.asarray(record["val_proba"], dtype=np.float32)
        arrays[f"{safe_key}__eval_proba"] = np.asarray(record["eval_proba"], dtype=np.float32)
    np.savez_compressed(path, **arrays)
    progress(args, f"[subject {data['subject']}] wrote probability artifact {path}")


def run_subject(subject: Any, args: argparse.Namespace) -> dict[str, Any]:
    def log(message: str) -> None:
        progress(args, f"[subject {subject}] {message}")

    log("start")
    data = prepare_dataset_subject(subject, args)
    view = preprocess_and_crop(data, args)
    seed0 = int(args.seeds[0])
    runs: dict[str, Any] = {}
    probability_records: dict[str, dict[str, np.ndarray]] = {}
    seed_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    ea_data: dict[str, Any] | None = None
    ea_view: dict[str, Any] | None = None
    ea_seed_cache: dict[tuple[Any, ...], dict[str, Any]] = {}

    def remember_logits(key: str, val_logits: np.ndarray, eval_logits: np.ndarray, run: dict[str, Any]) -> None:
        if not bool(args.save_probabilities):
            return
        temperature = float(run["calibration"]["temperature"])
        probability_records[key] = {
            "val_proba": proba_from_logits(val_logits, temperature),
            "eval_proba": proba_from_logits(eval_logits, temperature),
        }

    def remember_proba(key: str, val_proba: np.ndarray, eval_proba: np.ndarray) -> None:
        if not bool(args.save_probabilities):
            return
        probability_records[key] = {
            "val_proba": normalized_proba(val_proba),
            "eval_proba": normalized_proba(eval_proba),
        }

    def cached_train(seed: int, **overrides: Any) -> dict[str, Any]:
        key = (seed, tuple(sorted((k, repr(v)) for k, v in overrides.items())))
        if key not in seed_cache:
            suffix = ", ".join(f"{k}={v}" for k, v in sorted(overrides.items())) or "baseline settings"
            log(f"train seed={int(seed)} {suffix}")
            seed_cache[key] = run_seed_model(view, data, args, seed, log_fn=log, **overrides)
        return seed_cache[key]

    def ensure_ea_view() -> tuple[dict[str, Any], dict[str, Any]]:
        nonlocal ea_data, ea_view
        if ea_data is None or ea_view is None:
            log("fit train-only Euclidean Alignment for neural experiments")
            ea_data = apply_train_only_euclidean_alignment(data)
            ea_view = preprocess_and_crop(ea_data, args)
        return ea_data, ea_view

    def cached_ea_train(seed: int, **overrides: Any) -> dict[str, Any]:
        current_data, current_view = ensure_ea_view()
        key = (seed, tuple(sorted((k, repr(v)) for k, v in overrides.items())))
        if key not in ea_seed_cache:
            suffix = ", ".join(f"{k}={v}" for k, v in sorted(overrides.items())) or "ea baseline settings"
            log(f"train EA seed={int(seed)} {suffix}")
            ea_seed_cache[key] = run_seed_model(current_view, current_data, args, seed, log_fn=log, **overrides)
        return ea_seed_cache[key]

    log("experiment baseline: start")
    baseline_model = cached_train(seed0)
    baseline_val_logits = aggregate_crop_logits(
        baseline_model["val_crop_logits"], view["val_trial_ids"], data["y_val"].shape[0], "mean_logits"
    )
    baseline_eval_logits = aggregate_crop_logits(
        baseline_model["eval_crop_logits"], view["eval_trial_ids"], data["y_eval"].shape[0], "mean_logits"
    )
    baseline = freeze_eval(
        "baseline_ce_cropped_shallow_convnet",
        baseline_val_logits,
        baseline_eval_logits,
        data,
        args,
        {"model": baseline_model["audit"], "crop_audit": view["audit"], "headline_control": True},
    )
    runs["baseline"] = baseline
    remember_logits("baseline", baseline_val_logits, baseline_eval_logits, baseline)
    log(
        "experiment baseline: done "
        f"val_acc={baseline['validation_metrics']['accuracy']:.3f} val_brier={baseline['validation_metrics']['Brier']:.4f} "
        f"heldout_acc={baseline['heldout_metrics']['accuracy']:.3f} heldout_brier={baseline['heldout_metrics']['Brier']:.4f}"
    )

    if "mdrm_t_ea" in args.experiments:
        log("experiment mdrm_t_ea: start")
        mdrm = run_mdrm_t_ea(data, args)
        runs["mdrm_t_ea"] = mdrm["run"]
        remember_proba("mdrm_t_ea", mdrm["val_proba"], mdrm["eval_proba"])
        log(
            "experiment mdrm_t_ea: done "
            f"val_acc={runs['mdrm_t_ea']['validation_metrics']['accuracy']:.3f} "
            f"val_brier={runs['mdrm_t_ea']['validation_metrics']['Brier']:.4f} "
            f"heldout_acc={runs['mdrm_t_ea']['heldout_metrics']['accuracy']:.3f} "
            f"heldout_brier={runs['mdrm_t_ea']['heldout_metrics']['Brier']:.4f}"
        )

    if "ea_baseline" in args.experiments:
        log("experiment ea_baseline: start")
        current_data, current_view = ensure_ea_view()
        ea_model = cached_ea_train(seed0)
        val_logits = aggregate_crop_logits(
            ea_model["val_crop_logits"], current_view["val_trial_ids"], current_data["y_val"].shape[0], "mean_logits"
        )
        eval_logits = aggregate_crop_logits(
            ea_model["eval_crop_logits"], current_view["eval_trial_ids"], current_data["y_eval"].shape[0], "mean_logits"
        )
        runs["ea_baseline"] = freeze_eval(
            "ea_baseline_ce_cropped_shallow_convnet",
            val_logits,
            eval_logits,
            current_data,
            args,
            {"model": ea_model["audit"], "crop_audit": current_view["audit"], "euclidean_alignment": current_data["euclidean_alignment"]},
        )
        remember_logits("ea_baseline", val_logits, eval_logits, runs["ea_baseline"])
        log("experiment ea_baseline: done")

    if "ea_seed_ensemble" in args.experiments:
        log("experiment ea_seed_ensemble: start")
        current_data, current_view = ensure_ea_view()
        seed_runs = [cached_ea_train(int(seed)) for seed in args.seeds]
        val_logits = np.mean(
            [
                aggregate_crop_logits(run["val_crop_logits"], current_view["val_trial_ids"], current_data["y_val"].shape[0], "mean_logits")
                for run in seed_runs
            ],
            axis=0,
        )
        eval_logits = np.mean(
            [
                aggregate_crop_logits(run["eval_crop_logits"], current_view["eval_trial_ids"], current_data["y_eval"].shape[0], "mean_logits")
                for run in seed_runs
            ],
            axis=0,
        )
        runs["ea_seed_ensemble"] = freeze_eval(
            "ea_seed_ensemble",
            val_logits,
            eval_logits,
            current_data,
            args,
            {
                "seeds": [int(seed) for seed in args.seeds],
                "aggregation": "average_trial_level_logits_across_ea_seeds",
                "euclidean_alignment": current_data["euclidean_alignment"],
            },
        )
        remember_logits("ea_seed_ensemble", val_logits, eval_logits, runs["ea_seed_ensemble"])
        log("experiment ea_seed_ensemble: done")

    if "ea_augmentation" in args.experiments:
        log("experiment ea_augmentation: start")
        current_data, current_view = ensure_ea_view()
        aug_grid = [
            AugmentConfig(),
            AugmentConfig(time_shift=8),
            AugmentConfig(channel_dropout=0.05),
            AugmentConfig(noise_std=0.01),
            AugmentConfig(freq_mask_frac=0.03),
        ]
        ea_base_run = cached_ea_train(seed0)
        base_val_logits = aggregate_crop_logits(
            ea_base_run["val_crop_logits"], current_view["val_trial_ids"], current_data["y_val"].shape[0], "mean_logits"
        )
        base_acc = float(metrics_from_logits(current_data["y_val"], base_val_logits)["accuracy"])
        candidates = []
        for aug in aug_grid:
            run = cached_ea_train(seed0, augment=aug)
            val_logits = aggregate_crop_logits(run["val_crop_logits"], current_view["val_trial_ids"], current_data["y_val"].shape[0], "mean_logits")
            eval_logits = aggregate_crop_logits(run["eval_crop_logits"], current_view["eval_trial_ids"], current_data["y_eval"].shape[0], "mean_logits")
            candidates.append({"augmentation": aug.__dict__, "validation_metrics_raw": metrics_from_logits(current_data["y_val"], val_logits), "val_logits": val_logits, "eval_logits": eval_logits})
        guarded = [row for row in candidates if row["validation_metrics_raw"]["accuracy"] >= base_acc] or candidates
        selected = min(guarded, key=lambda row: (row["validation_metrics_raw"]["Brier"], -row["validation_metrics_raw"]["accuracy"], repr(row["augmentation"])))
        runs["ea_augmentation"] = freeze_eval(
            "ea_best_light_augmentation",
            selected["val_logits"],
            selected["eval_logits"],
            current_data,
            args,
            {
                "selected_augmentation": selected["augmentation"],
                "selection_rule": "validation_Brier_with_validation_accuracy_guardrail",
                "candidates": [{k: v for k, v in row.items() if k not in ("val_logits", "eval_logits")} for row in candidates],
                "euclidean_alignment": current_data["euclidean_alignment"],
            },
        )
        remember_logits("ea_augmentation", selected["val_logits"], selected["eval_logits"], runs["ea_augmentation"])
        log(f"experiment ea_augmentation: done selected={selected['augmentation']}")

    if "seed_ensemble" in args.experiments:
        log("experiment seed_ensemble: start")
        seed_runs = [cached_train(int(seed)) for seed in args.seeds]
        val_logits = np.mean(
            [
                aggregate_crop_logits(run["val_crop_logits"], view["val_trial_ids"], data["y_val"].shape[0], "mean_logits")
                for run in seed_runs
            ],
            axis=0,
        )
        eval_logits = np.mean(
            [
                aggregate_crop_logits(run["eval_crop_logits"], view["eval_trial_ids"], data["y_eval"].shape[0], "mean_logits")
                for run in seed_runs
            ],
            axis=0,
        )
        runs["seed_ensemble"] = freeze_eval(
            "seed_ensemble",
            val_logits,
            eval_logits,
            data,
            args,
            {"seeds": [int(seed) for seed in args.seeds], "aggregation": "average_trial_level_logits_across_seeds"},
        )
        remember_logits("seed_ensemble", val_logits, eval_logits, runs["seed_ensemble"])
        log("experiment seed_ensemble: done")

    if "crop_aggregation" in args.experiments:
        log("experiment crop_aggregation: start")
        methods = ["mean_logits", "median_logits", "trimmed_mean_logits", "confidence_weighted_logits"]
        candidates = []
        for method in methods:
            val_logits = aggregate_crop_logits(baseline_model["val_crop_logits"], view["val_trial_ids"], data["y_val"].shape[0], method)
            eval_logits = aggregate_crop_logits(baseline_model["eval_crop_logits"], view["eval_trial_ids"], data["y_eval"].shape[0], method)
            candidates.append(
                {
                    "method": method,
                    "validation_metrics_raw": metrics_from_logits(data["y_val"], val_logits),
                    "heldout_metrics_raw_report_only": metrics_from_logits(data["y_eval"], eval_logits, include_confusion=args.include_confusion),
                    "val_logits": val_logits,
                    "eval_logits": eval_logits,
                }
            )
        selected = min(candidates, key=lambda row: (row["validation_metrics_raw"]["Brier"], -row["validation_metrics_raw"]["accuracy"], row["method"]))
        runs["crop_aggregation"] = freeze_eval(
            "validation_selected_crop_aggregation",
            selected["val_logits"],
            selected["eval_logits"],
            data,
            args,
            {
                "selected_method": selected["method"],
                "selection_objective": "validation_Brier_tie_break_validation_accuracy",
                "candidates": [{k: v for k, v in row.items() if k not in ("val_logits", "eval_logits")} for row in candidates],
            },
        )
        remember_logits("crop_aggregation", selected["val_logits"], selected["eval_logits"], runs["crop_aggregation"])
        log(f"experiment crop_aggregation: done selected={selected['method']}")

    if "brier_loss" in args.experiments:
        log("experiment brier_loss: start")
        lambdas = [0.0, 0.02, 0.05, 0.1]
        candidates = []
        base_acc = float(baseline["validation_metrics_raw"]["accuracy"])
        for lam in lambdas:
            run = cached_train(seed0, lambda_brier=float(lam))
            val_logits = aggregate_crop_logits(run["val_crop_logits"], view["val_trial_ids"], data["y_val"].shape[0], "mean_logits")
            eval_logits = aggregate_crop_logits(run["eval_crop_logits"], view["eval_trial_ids"], data["y_eval"].shape[0], "mean_logits")
            metrics = metrics_from_logits(data["y_val"], val_logits)
            candidates.append({"lambda_brier": float(lam), "validation_metrics_raw": metrics, "val_logits": val_logits, "eval_logits": eval_logits, "audit": run["audit"]})
        guarded = [row for row in candidates if row["validation_metrics_raw"]["accuracy"] >= base_acc or row["lambda_brier"] == 0.0]
        selected = min(guarded, key=lambda row: (row["validation_metrics_raw"]["Brier"], -row["validation_metrics_raw"]["accuracy"], row["lambda_brier"]))
        runs["brier_loss"] = freeze_eval(
            "ce_plus_brier_regularization",
            selected["val_logits"],
            selected["eval_logits"],
            data,
            args,
            {
                "selected_lambda": selected["lambda_brier"],
                "selection_rule": "lowest_A0xT_validation_Brier_with_validation_accuracy_not_below_pure_CE_when_available",
                "candidates": [{k: v for k, v in row.items() if k not in ("val_logits", "eval_logits")} for row in candidates],
            },
        )
        remember_logits("brier_loss", selected["val_logits"], selected["eval_logits"], runs["brier_loss"])
        log(f"experiment brier_loss: done selected_lambda={selected['lambda_brier']}")

    if "calibration_grid" in args.experiments:
        log("experiment calibration_grid: start")
        grid = calibration_grid(baseline_val_logits, baseline_eval_logits, data, args)
        runs["calibration_grid"] = {
            "experiment": "posthoc_calibration_grid",
            "validation_metrics": grid["selected"]["validation_metrics"],
            "heldout_metrics": grid["selected"]["heldout_metrics"],
            "selection_audit": {
                "selection_split": split_name(data, "selection_split", SELECTION_SPLIT),
                "heldout_split": split_name(data, "heldout_split", HELDOUT_SPLIT),
                "eval_labels_used_for_selection": False,
            },
            "metadata": {k: v for k, v in grid.items() if k != "selected_proba"},
        }
        if bool(args.save_probabilities):
            selected_proba = grid["selected_proba"]
            remember_proba("calibration_grid", selected_proba["val_proba"], selected_proba["eval_proba"])
        log(f"experiment calibration_grid: done selected={grid['selected']['method']}")

    teacher_selection: dict[str, Any] | None = None
    if any(exp in args.experiments for exp in ("teacher_student_mixture", "stronger_teacher")):
        log("experiment stronger_teacher: start")
        teacher_selection = select_teacher(data, ["lda_svm_vote", "fbcsp_lda", "shrinkage_lda", "tangent_logreg", "mdrm_t"])
        teacher = teacher_selection["selected"]["teacher"]
        if "stronger_teacher" in args.experiments:
            runs["stronger_teacher"] = {
                "experiment": "stronger_classical_teacher",
                "validation_metrics": teacher_selection["selected"]["validation_metrics"],
                "heldout_metrics": teacher_selection["selected"]["heldout_metrics_for_report_only"],
                "selection_audit": {
                    "selection_split": split_name(data, "selection_split", SELECTION_SPLIT),
                    "heldout_split": split_name(data, "heldout_split", HELDOUT_SPLIT),
                    "eval_labels_used_for_selection": False,
                },
                "metadata": {
                    "selected_teacher": teacher_selection["selected"]["name"],
                    "selection_objective": "validation_Brier_tie_break_validation_accuracy",
                    "candidates": [
                        {k: v for k, v in row.items() if k != "teacher"} for row in teacher_selection["candidates"]
                    ],
                },
            }
            remember_proba("stronger_teacher", teacher["val_proba"], teacher["eval_proba"])
        log(f"experiment stronger_teacher: done selected={teacher_selection['selected']['name']}")
        if "teacher_student_mixture" in args.experiments:
            log("experiment teacher_student_mixture: start")
            scalar_t = float(baseline["calibration"]["temperature"])
            student_raw_val = proba_from_logits(baseline_val_logits)
            student_raw_eval = proba_from_logits(baseline_eval_logits)
            student_cal_val = proba_from_logits(baseline_val_logits, scalar_t)
            student_cal_eval = proba_from_logits(baseline_eval_logits, scalar_t)
            alpha, alpha_records = fit_alpha(student_cal_val, teacher["val_proba"], data["y_val"], args.alpha_grid_step)
            mix_val = normalized_proba(alpha * student_cal_val + (1.0 - alpha) * teacher["val_proba"])
            mix_eval = normalized_proba(alpha * student_cal_eval + (1.0 - alpha) * teacher["eval_proba"])
            runs["teacher_student_mixture"] = {
                "experiment": "teacher_student_probability_mixing",
                "validation_metrics": metric_bundle(data["y_val"], mix_val),
                "heldout_metrics": metric_bundle(data["y_eval"], mix_eval, include_confusion=args.include_confusion),
                "selection_audit": {
                    "selection_split": split_name(data, "selection_split", SELECTION_SPLIT),
                    "heldout_split": split_name(data, "heldout_split", HELDOUT_SPLIT),
                    "eval_labels_used_for_selection": False,
                },
                "metadata": {
                    "selected_alpha": float(alpha),
                    "teacher": teacher_selection["selected"]["name"],
                    "alpha_grid": alpha_records,
                    "comparators": {
                        "matched_raw_student": {
                            "validation": metric_bundle(data["y_val"], student_raw_val),
                            "heldout": metric_bundle(data["y_eval"], student_raw_eval, include_confusion=args.include_confusion),
                        },
                        "calibrated_student": {
                            "validation": metric_bundle(data["y_val"], student_cal_val),
                            "heldout": metric_bundle(data["y_eval"], student_cal_eval, include_confusion=args.include_confusion),
                        },
                        "teacher_alone": {
                            "validation": metric_bundle(data["y_val"], teacher["val_proba"]),
                            "heldout": metric_bundle(data["y_eval"], teacher["eval_proba"], include_confusion=args.include_confusion),
                        },
                    },
                },
            }
            remember_proba("teacher_student_mixture", mix_val, mix_eval)
            log(f"experiment teacher_student_mixture: done alpha={alpha:.2f}")

    if "swa_ema" in args.experiments:
        log("experiment swa_ema: start")
        ema_run = cached_train(seed0, average_checkpoint="ema")
        val_logits = aggregate_crop_logits(ema_run["val_crop_logits"], view["val_trial_ids"], data["y_val"].shape[0], "mean_logits")
        eval_logits = aggregate_crop_logits(ema_run["eval_crop_logits"], view["eval_trial_ids"], data["y_eval"].shape[0], "mean_logits")
        runs["swa_ema"] = freeze_eval("ema_late_checkpoint", val_logits, eval_logits, data, args, {"checkpoint_average": "ema", "model": ema_run["audit"]})
        remember_logits("swa_ema", val_logits, eval_logits, runs["swa_ema"])
        log("experiment swa_ema: done")

    if "calibration_checkpoint" in args.experiments:
        log("experiment calibration_checkpoint: start")
        ckpt_run = cached_train(seed0, checkpoint_rule="calibration_aware")
        val_logits = aggregate_crop_logits(ckpt_run["val_crop_logits"], view["val_trial_ids"], data["y_val"].shape[0], "mean_logits")
        eval_logits = aggregate_crop_logits(ckpt_run["eval_crop_logits"], view["eval_trial_ids"], data["y_eval"].shape[0], "mean_logits")
        runs["calibration_checkpoint"] = freeze_eval(
            "calibration_aware_checkpointing",
            val_logits,
            eval_logits,
            data,
            args,
            {"checkpoint_selection": ckpt_run["chosen_checkpoint"], "model": ckpt_run["audit"]},
        )
        remember_logits("calibration_checkpoint", val_logits, eval_logits, runs["calibration_checkpoint"])
        log(f"experiment calibration_checkpoint: done epoch={ckpt_run['chosen_checkpoint']['epoch']}")

    if "augmentation" in args.experiments:
        log("experiment augmentation: start")
        aug_grid = [
            AugmentConfig(),
            AugmentConfig(time_shift=8),
            AugmentConfig(channel_dropout=0.05),
            AugmentConfig(noise_std=0.01),
            AugmentConfig(freq_mask_frac=0.03),
        ]
        candidates = []
        base_acc = float(baseline["validation_metrics_raw"]["accuracy"])
        for aug in aug_grid:
            run = cached_train(seed0, augment=aug)
            val_logits = aggregate_crop_logits(run["val_crop_logits"], view["val_trial_ids"], data["y_val"].shape[0], "mean_logits")
            eval_logits = aggregate_crop_logits(run["eval_crop_logits"], view["eval_trial_ids"], data["y_eval"].shape[0], "mean_logits")
            candidates.append({"augmentation": aug.__dict__, "validation_metrics_raw": metrics_from_logits(data["y_val"], val_logits), "val_logits": val_logits, "eval_logits": eval_logits})
        guarded = [row for row in candidates if row["validation_metrics_raw"]["accuracy"] >= base_acc] or candidates
        selected = min(guarded, key=lambda row: (row["validation_metrics_raw"]["Brier"], -row["validation_metrics_raw"]["accuracy"], repr(row["augmentation"])))
        runs["augmentation"] = freeze_eval(
            "best_light_augmentation",
            selected["val_logits"],
            selected["eval_logits"],
            data,
            args,
            {
                "selected_augmentation": selected["augmentation"],
                "selection_rule": "validation_Brier_with_validation_accuracy_guardrail",
                "candidates": [{k: v for k, v in row.items() if k not in ("val_logits", "eval_logits")} for row in candidates],
            },
        )
        remember_logits("augmentation", selected["val_logits"], selected["eval_logits"], runs["augmentation"])
        log(f"experiment augmentation: done selected={selected['augmentation']}")

    if "architecture" in args.experiments:
        log("experiment architecture: start")
        arch_grid = [
            ArchConfig(dropout=float(args.dropout)),
            ArchConfig(temporal_kernel=32, dropout=float(args.dropout), temporal_filters=16),
            ArchConfig(temporal_kernel=64, dropout=0.35, temporal_filters=16),
            ArchConfig(temporal_kernel=64, dropout=float(args.dropout), max_norm=2.0, temporal_filters=16),
            ArchConfig(temporal_kernel=96, dropout=float(args.dropout), temporal_filters=24),
        ]
        candidates = []
        for arch in arch_grid:
            run = cached_train(seed0, arch=arch)
            val_logits = aggregate_crop_logits(run["val_crop_logits"], view["val_trial_ids"], data["y_val"].shape[0], "mean_logits")
            eval_logits = aggregate_crop_logits(run["eval_crop_logits"], view["eval_trial_ids"], data["y_eval"].shape[0], "mean_logits")
            candidates.append({"architecture": arch.__dict__, "validation_metrics_raw": metrics_from_logits(data["y_val"], val_logits), "val_logits": val_logits, "eval_logits": eval_logits})
        selected = min(candidates, key=lambda row: (row["validation_metrics_raw"]["Brier"], -row["validation_metrics_raw"]["accuracy"], repr(row["architecture"])))
        runs["architecture"] = freeze_eval(
            "best_shallow_architecture_variant",
            selected["val_logits"],
            selected["eval_logits"],
            data,
            args,
            {
                "selected_architecture": selected["architecture"],
                "selection_rule": "per_subject_A0xT_validation_Brier_tie_break_validation_accuracy",
                "candidates": [{k: v for k, v in row.items() if k not in ("val_logits", "eval_logits")} for row in candidates],
            },
        )
        remember_logits("architecture", selected["val_logits"], selected["eval_logits"], runs["architecture"])
        log(f"experiment architecture: done selected={selected['architecture']}")

    log("done")
    if bool(args.save_probabilities):
        save_probability_artifact(args, data, runs, probability_records)
    return {
        "dataset": data["dataset"],
        "dataset_label": data.get("dataset_label", data["dataset"]),
        "subject": data["subject"],
        "classes": data["classes"],
        "n_classes": int(data["n_classes"]),
        "chance_accuracy": float(data["chance_accuracy"]),
        "chance_Brier": float(data["chance_Brier"]),
        "paths": data["paths"],
        "split": data["split"],
        "protocol": data["protocol"],
        "validation_discipline": {
            "train_split": split_name(data, "train_split", "A0xT_train"),
            "selection_split": split_name(data, "selection_split", SELECTION_SPLIT),
            "heldout_split": split_name(data, "heldout_split", HELDOUT_SPLIT),
            "heldout_labels_used_for": "final_heldout_metrics_only",
            "eval_labels_used_for_selection": False,
        },
        "seeds": [int(s) for s in args.seeds],
        "experiments": runs,
    }


def flatten_subject_rows(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for payload in payloads:
        subject = payload["subject"]
        for key, exp in payload["experiments"].items():
            val = exp.get("validation_metrics", {})
            held = exp.get("heldout_metrics", {})
            rows.append(
                {
                    "dataset": payload.get("dataset", "bci4_2a"),
                    "dataset_label": payload.get("dataset_label", payload.get("dataset", "bci4_2a")),
                    "subject": subject,
                    "n_classes": payload.get("n_classes"),
                    "experiment_key": key,
                    "experiment": exp.get("experiment", key),
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
                    "selection_split": exp.get("selection_audit", {}).get("selection_split", SELECTION_SPLIT),
                    "eval_labels_used_for_selection": exp.get("selection_audit", {}).get("eval_labels_used_for_selection", False),
                }
            )
    return rows


def aggregate_rows(subject_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for key in EXPERIMENT_ORDER:
        rows = [row for row in subject_rows if row["experiment_key"] == key]
        if not rows:
            continue
        out: dict[str, Any] = {
            "dataset": rows[0].get("dataset"),
            "experiment_key": key,
            "experiment": rows[0]["experiment"],
            "n_subjects": len(rows),
            "n_classes": rows[0].get("n_classes"),
            "chance_accuracy": rows[0].get("chance_accuracy"),
            "chance_Brier": rows[0].get("chance_Brier"),
        }
        for metric in ("accuracy", "Brier", "ECE", "NLL"):
            for prefix in ("val", "heldout"):
                values = np.asarray([float(row[f"{prefix}_{metric}"]) for row in rows], dtype=float)
                out[f"{prefix}_{metric}_mean"] = float(np.nanmean(values))
                out[f"{prefix}_{metric}_std"] = float(np.nanstd(values, ddof=0))
        summary.append(out)
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_dataset_artifacts(args: argparse.Namespace, out_dir: Path, payloads: list[dict[str, Any]], subjects: list[Any]) -> None:
    slug = dataset_slug(args.dataset)
    subject_rows = flatten_subject_rows(payloads)
    summary = aggregate_rows(subject_rows)
    dataset_metrics = out_dir / f"validation_only_selection_{slug}_subject_metrics.csv"
    dataset_summary_csv = out_dir / f"validation_only_selection_{slug}_summary.csv"
    dataset_summary_json = out_dir / f"validation_only_selection_{slug}_summary.json"
    write_csv(dataset_metrics, subject_rows)
    write_csv(dataset_summary_csv, summary)
    summary_payload = {
        "dataset": args.dataset,
        "validation_discipline": {
            "selection_split": payloads[0]["validation_discipline"]["selection_split"] if payloads else SELECTION_SPLIT,
            "heldout_split": payloads[0]["validation_discipline"]["heldout_split"] if payloads else HELDOUT_SPLIT,
            "eval_labels_used_for_selection": False,
        },
        "subjects": subjects,
        "experiments": args.experiments,
        "external_confirmatory": bool(args.external_confirmatory),
        "platform": platform_fingerprint(args.device),
        "summary": summary,
    }
    write_json(dataset_summary_json, summary_payload)

    if args.dataset == "bci4_2a":
        write_csv(out_dir / "validation_only_selection_subject_metrics.csv", subject_rows)
        write_csv(out_dir / "validation_only_selection_summary.csv", summary)
        write_json(
            out_dir / "validation_only_selection_summary.json",
            {
                **summary_payload,
                "validation_discipline": {
                    "selection_split": SELECTION_SPLIT,
                    "heldout_split": HELDOUT_SPLIT,
                    "a0xe_used_for_selection": False,
                    "eval_labels_used_for_selection": False,
                },
            },
        )
    progress(args, f"Wrote {dataset_summary_csv}")


def inspect_subjects(args: argparse.Namespace, subjects: list[Any], out_dir: Path) -> None:
    slug = dataset_slug(args.dataset)
    rows = []
    for subject in subjects:
        if args.dataset in {"BNCI2014_001", "BNCI2014_004", "PhysionetMI"}:
            rows.append(inspect_moabb_subject(subject, args))
            row = rows[-1]
            if row["ok"]:
                progress(args, f"[subject {subject}] inspect ok protocol={row['protocol'].get('protocol_name')}")
            else:
                progress(args, f"[subject {subject}] inspect skipped: {row['error']}")
            continue
        try:
            data = prepare_dataset_subject(subject, args)
            rows.append(
                {
                    "dataset": data["dataset"],
                    "subject": data["subject"],
                    "ok": True,
                    "n_classes": data["n_classes"],
                    "classes": data["classes"],
                    "protocol": data["protocol"],
                    "validation_discipline": {
                        "train_split": data["train_split"],
                        "selection_split": data["selection_split"],
                        "heldout_split": data["heldout_split"],
                        "eval_labels_used_for_selection": False,
                    },
                }
            )
            progress(args, f"[subject {subject}] inspect ok protocol={data['protocol'].get('protocol_name')}")
        except Exception as exc:
            rows.append({"dataset": args.dataset, "subject": subject, "ok": False, "error": repr(exc)})
            progress(args, f"[subject {subject}] inspect skipped: {exc}")
    path = out_dir / f"validation_only_selection_{slug}_protocol_inspection.json"
    write_json(path, {"dataset": args.dataset, "subjects": subjects, "inspection": rows})
    print(f"Wrote {path}", flush=True)


def inspect_moabb_subject(subject: Any, args: argparse.Namespace) -> dict[str, Any]:
    data_dir = args.data_dir / "moabb"
    dataset = _moabb_dataset_instance(args.dataset, data_dir)
    paradigm = _make_moabb_paradigm(args.dataset, data_dir)
    try:
        X, y, meta = paradigm.get_data(dataset=dataset, subjects=[subject])
    except Exception as exc:
        return {
            "dataset": args.dataset,
            "subject": subject,
            "ok": False,
            "error": (
                f"MOABB data could not be loaded. If the dataset is not cached under {data_dir}, "
                f"a network download may be required. Original error: {exc!r}"
            ),
        }
    y = np.asarray(y)
    meta = pd.DataFrame(meta).reset_index(drop=True)
    base = {
        "dataset": args.dataset,
        "subject": subject,
        "meta_structure": structure_summary(meta),
        "n_trials": int(np.asarray(X).shape[0]),
        "raw_class_counts": class_counts(y),
    }
    try:
        train_mask, val_mask, eval_mask, protocol = moabb_protocol_masks(
            meta,
            y,
            args.dataset,
            int(args.seed),
            bool(args.allow_stratified_external_split),
        )
        y_train, y_val, y_eval, classes = encode_split_labels(y[train_mask], y[val_mask], y[eval_mask])
        return {
            **base,
            "ok": True,
            "n_classes": len(classes),
            "classes": classes,
            "protocol": {
                **protocol,
                "n_train": int(train_mask.sum()),
                "n_validation": int(val_mask.sum()),
                "n_eval": int(eval_mask.sum()),
                "train_class_counts": class_counts(y_train),
                "validation_class_counts": class_counts(y_val),
                "eval_class_counts": class_counts(y_eval),
                "eval_labels_used_for_selection": False,
            },
            "validation_discipline": {
                "train_split": "external_train",
                "selection_split": "external_validation",
                "heldout_split": "external_eval_final_report_only",
                "eval_labels_used_for_selection": False,
            },
        }
    except Exception as exc:
        return {**base, "ok": False, "error": str(exc)}


def merge_subject_metric_csvs(out_dir: Path) -> Path:
    rows: list[dict[str, Any]] = []
    for path in sorted(out_dir.rglob("validation_only_selection_*_subject_metrics.csv")):
        if path.name == "validation_only_selection_subject_metrics.csv":
            continue
        if any(part.endswith("_smoke") for part in path.parts):
            continue
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row = dict(row)
                row.setdefault("source_file", str(path))
                rows.append(row)
    path = out_dir / "validation_only_selection_dataset_comparison.csv"
    write_csv(path, rows)
    return path


def main() -> None:
    args = parse_args()
    # Default stays cuda-else-cpu. MPS is opt-in via --device mps: switching
    # backends changes results, and every retained result predating this flag
    # was produced on cpu.
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    configure_determinism(bool(args.deterministic))
    out_dir = args.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.merge_summaries:
        path = merge_subject_metric_csvs(out_dir)
        print(f"Wrote {path}", flush=True)
        return
    subjects = resolve_subjects(args)
    if args.inspect_only:
        inspect_subjects(args, subjects, out_dir)
        return
    payloads = []
    for subject in subjects:
        payload = run_subject(subject, args)
        payloads.append(payload)
        slug = dataset_slug(args.dataset)
        subject_id = safe_subject_id(subject)
        write_json(out_dir / f"validation_only_selection_{slug}_subject{subject_id}.json", payload)
        if args.dataset == "bci4_2a":
            write_json(out_dir / f"validation_only_selection_subject{subject_id}.json", payload)
        progress(args, f"[subject {subject}] wrote validation-only selection results")
    write_dataset_artifacts(args, out_dir, payloads, subjects)


if __name__ == "__main__":
    main()
