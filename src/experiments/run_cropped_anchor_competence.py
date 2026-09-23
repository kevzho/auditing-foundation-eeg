#!/usr/bin/env python3
"""Cropped classical-anchored neural competence for BCI IV-2a A0xT -> A0xE.

This runner tests stronger no-graph CE students under crop-aggregated MI
training.  Classical decoders remain leakage-safe competence/calibration
anchors, and EDL/graph claims remain gated downstream.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate import compute_brier, compute_ece, compute_selective_acc
from experiments.run_classical_anchor_competence import (
    CHANCE_ACC,
    CHANCE_BRIER,
    N_CLASSES,
    N_SUBJECTS,
    TEACHERS,
    apply_standardize,
    distill_loss,
    fit_teacher,
    fit_temperature_from_logits,
    load_minimal_baselines,
    load_variant_metrics,
    make_jsonable,
    paired_deltas,
    prepare_subject,
    proba_from_logits,
    standardize_fit,
    write_json,
)
from experiments.run_primary import DEFAULT_BATCH_SIZE, DEFAULT_SEED
from models.competence_graph_eegnet import CompetenceConfig, build_competence_model


CROPPED_TO_RAW = {
    "anchor_multiscale_temporal_ce": "raw_multiscale_temporal_ce",
    "raw_multiscale_temporal_ce": "raw_multiscale_temporal_ce",
    "anchor_cropped_eegnet_ce": "raw_eegnet_ce",
    "raw_cropped_eegnet_ce": "raw_eegnet_ce",
    "anchor_cropped_shallow_convnet_ce": "raw_shallow_convnet_ce",
    "raw_cropped_shallow_convnet_ce": "raw_shallow_convnet_ce",
    "anchor_cropped_filterbank_shallow_ce": "raw_filterbank_shallow_ce",
    "raw_cropped_filterbank_shallow_ce": "raw_filterbank_shallow_ce",
}
ANCHOR_TO_RAW = {
    "anchor_multiscale_temporal_ce": "raw_multiscale_temporal_ce",
    "anchor_cropped_eegnet_ce": "raw_cropped_eegnet_ce",
    "anchor_cropped_shallow_convnet_ce": "raw_cropped_shallow_convnet_ce",
    "anchor_cropped_filterbank_shallow_ce": "raw_cropped_filterbank_shallow_ce",
}
PREVIOUS_NONCROPPED_MATCH = {
    "anchor_cropped_eegnet_ce": "anchor_raw_eegnet_ce",
    "anchor_cropped_filterbank_shallow_ce": "anchor_filterbank_no_graph",
}
MULTISCALE_PREVIOUS_CROPPED_MATCHES = {
    "anchor_multiscale_temporal_ce": (
        "anchor_cropped_shallow_convnet_ce",
        "anchor_cropped_eegnet_ce",
        "anchor_cropped_filterbank_shallow_ce",
    ),
    "raw_multiscale_temporal_ce": (
        "raw_cropped_shallow_convnet_ce",
        "raw_cropped_eegnet_ce",
        "raw_cropped_filterbank_shallow_ce",
    ),
}
DEFAULT_VARIANTS = (
    "anchor_multiscale_temporal_ce",
    "raw_multiscale_temporal_ce",
    "anchor_cropped_eegnet_ce",
    "raw_cropped_eegnet_ce",
    "anchor_cropped_shallow_convnet_ce",
    "raw_cropped_shallow_convnet_ce",
)
VARIANTS = tuple(CROPPED_TO_RAW)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=int, action="append", choices=range(1, N_SUBJECTS + 1))
    parser.add_argument("--subjects", type=int, nargs="+", choices=range(1, N_SUBJECTS + 1))
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS), choices=VARIANTS)
    parser.add_argument("--teacher", default="lda_svm_vote", choices=TEACHERS)
    parser.add_argument("--lambda-distill", type=float, default=0.5)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--crop-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--crop-strides", type=int, nargs="+", default=None)
    parser.add_argument("--crop-size", type=int, default=None, help="Deprecated singleton form; used only when --crop-sizes is absent.")
    parser.add_argument("--crop-stride", type=int, default=None, help="Deprecated singleton form; used only when --crop-strides is absent.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results") / "cropped_anchor")
    parser.add_argument("--baseline-dir", type=Path, default=Path("results") / "minimal_edl")
    parser.add_argument("--graph-rescue-dir", type=Path, default=Path("results") / "graph_rescue")
    parser.add_argument("--classical-anchor-dir", type=Path, default=Path("results") / "classical_anchor")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--balanced-sampler", dest="balanced_sampler", action="store_true", default=True)
    parser.add_argument("--no-balanced-sampler", dest="balanced_sampler", action="store_false")
    parser.add_argument("--cosine-lr", dest="cosine_lr", action="store_true", default=True)
    parser.add_argument("--no-cosine-lr", dest="cosine_lr", action="store_false")
    parser.add_argument("--augment", dest="augment", action="store_true", default=True)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--noise-std", type=float, default=0.02)
    parser.add_argument("--time-mask-frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default=None)
    parser.add_argument("--temperature-max", type=float, default=20.0)
    parser.add_argument("--calibration-anchor", dest="calibration_anchor", action="store_true", default=False)
    parser.add_argument("--no-calibration-anchor", dest="calibration_anchor", action="store_false")
    parser.add_argument("--alpha-grid-step", type=float, default=0.05)
    parser.add_argument("--gate-coverages", type=float, nargs="+", default=[0.60, 0.70, 0.80])
    parser.add_argument("--disagreement-score", choices=("dot", "js"), default="dot")
    args = parser.parse_args()
    if args.crop_sizes is None:
        args.crop_sizes = [int(args.crop_size)] if args.crop_size is not None else [384, 512, 640]
    if args.crop_strides is None:
        args.crop_strides = [int(args.crop_stride)] if args.crop_stride is not None else [128]
    if len(args.crop_strides) not in (1, len(args.crop_sizes)):
        parser.error("--crop-strides must have length 1 or match --crop-sizes")
    if any(size <= 0 for size in args.crop_sizes):
        parser.error("--crop-sizes must be positive")
    if any(stride <= 0 for stride in args.crop_strides):
        parser.error("--crop-strides must be positive")
    if not 0.0 < float(args.alpha_grid_step) <= 1.0:
        parser.error("--alpha-grid-step must be in (0, 1]")
    if any(not 0.0 < float(coverage) <= 1.0 for coverage in args.gate_coverages):
        parser.error("--gate-coverages must be in (0, 1]")
    return args


def crop_starts(n_times: int, crop_size: int, crop_stride: int) -> list[int]:
    if crop_size <= 0 or crop_size >= n_times:
        return [0]
    if crop_stride <= 0:
        raise ValueError("--crop-stride must be positive")
    last = n_times - crop_size
    starts = list(range(0, last + 1, crop_stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


def normalize_crop_strides(crop_sizes: list[int], crop_strides: list[int]) -> list[int]:
    if len(crop_strides) == 1:
        return [int(crop_strides[0]) for _ in crop_sizes]
    return [int(s) for s in crop_strides]


def pad_crop(crop: np.ndarray, target_size: int) -> np.ndarray:
    if crop.shape[1] == target_size:
        return crop
    out = np.zeros((crop.shape[0], target_size), dtype=np.float32)
    out[:, : crop.shape[1]] = crop
    return out


def make_multiscale_crops(
    X: np.ndarray,
    y: np.ndarray | None,
    crop_sizes: list[int],
    crop_strides: list[int],
    teacher_proba: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, dict[str, Any]]:
    X = np.asarray(X, dtype=np.float32)
    labels_source = np.zeros(X.shape[0], dtype=np.int64) if y is None else np.asarray(y, dtype=np.int64)
    strides = normalize_crop_strides(crop_sizes, crop_strides)
    scale_specs = []
    for crop_size, stride in zip(crop_sizes, strides):
        effective_size = X.shape[2] if crop_size >= X.shape[2] else crop_size
        starts = crop_starts(X.shape[2], effective_size, stride)
        scale_specs.append(
            {
                "requested_crop_size": int(crop_size),
                "effective_crop_size": int(effective_size),
                "crop_stride": int(stride),
                "starts": [int(s) for s in starts],
                "crops_per_trial": int(len(starts)),
            }
        )
    target_size = max(int(spec["effective_crop_size"]) for spec in scale_specs)
    crops = []
    labels = []
    trial_ids = []
    teacher_rows = [] if teacher_proba is not None else None
    for trial_id in range(X.shape[0]):
        for scale_id, spec in enumerate(scale_specs):
            effective_size = int(spec["effective_crop_size"])
            for start in spec["starts"]:
                crops.append(pad_crop(X[trial_id, :, start : start + effective_size], target_size))
                labels.append(labels_source[trial_id])
                trial_ids.append(trial_id)
                if teacher_rows is not None:
                    teacher_rows.append(teacher_proba[trial_id])
    X_crops = np.stack(crops, axis=0).astype(np.float32, copy=False)
    y_crops = np.asarray(labels, dtype=np.int64)
    trial_ids_arr = np.asarray(trial_ids, dtype=np.int64)
    teacher_crops = np.asarray(teacher_rows, dtype=np.float32) if teacher_rows is not None else None
    audit = {
        "crop_sizes": [int(s) for s in crop_sizes],
        "crop_strides": [int(s) for s in strides],
        "scales": scale_specs,
        "padded_to_n_times": int(target_size),
        "padding_value_after_standardization": 0.0,
        "crops_per_trial": int(sum(spec["crops_per_trial"] for spec in scale_specs)),
    }
    return X_crops, y_crops, trial_ids_arr, teacher_crops, audit


def preprocessing_view(data: dict[str, Any], args: argparse.Namespace, teacher: dict[str, Any], anchored: bool) -> dict[str, Any]:
    mean, std = standardize_fit(data["X_train_raw"])
    X_train = apply_standardize(data["X_train_raw"], mean, std)
    X_val = apply_standardize(data["X_val_raw"], mean, std)
    X_eval = apply_standardize(data["X_eval_raw"], mean, std)
    train_teacher = teacher["train_proba"] if anchored else None
    X_train_c, y_train_c, train_trial_ids, train_teacher_c, train_crop_audit = make_multiscale_crops(
        X_train,
        data["y_train"],
        args.crop_sizes,
        args.crop_strides,
        train_teacher,
    )
    X_val_c, _, val_trial_ids, _, val_crop_audit = make_multiscale_crops(X_val, None, args.crop_sizes, args.crop_strides, None)
    X_eval_c, _, eval_trial_ids, _, eval_crop_audit = make_multiscale_crops(X_eval, None, args.crop_sizes, args.crop_strides, None)
    return {
        "X_train_crops": X_train_c,
        "y_train_crops": y_train_c,
        "train_trial_ids": train_trial_ids,
        "train_teacher_crops": train_teacher_c,
        "X_val_crops": X_val_c,
        "val_trial_ids": val_trial_ids,
        "X_eval_crops": X_eval_c,
        "eval_trial_ids": eval_trial_ids,
        "crop_audit": {
            **train_crop_audit,
            "train_crops": int(X_train_c.shape[0]),
            "val_crops": int(X_val_c.shape[0]),
            "eval_crops": int(X_eval_c.shape[0]),
            "validation_crop_audit": val_crop_audit,
            "eval_crop_audit": eval_crop_audit,
            "train_source": "A0xT_train",
            "validation_aggregation": "mean_crop_logits_to_trial",
            "eval_aggregation": "mean_crop_logits_to_trial",
            "eval_labels_used_for_crop_selection": False,
        },
        "preprocessing": {
            "euclidean_alignment": {"used": False, "fit_split": None, "eval_used_for_fit": False},
            "standardization": {"fit_split": "A0xT_train", "per_channel": True, "eval_used_for_fit": False},
        },
    }


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    teacher_proba: np.ndarray | None,
    batch_size: int,
    shuffle: bool,
    balanced_sampler: bool = False,
    seed: int | None = None,
) -> DataLoader:
    tensors: list[torch.Tensor] = [torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long)]
    if teacher_proba is not None:
        tensors.append(torch.as_tensor(teacher_proba, dtype=torch.float32))
    sampler = None
    if balanced_sampler:
        classes, counts = np.unique(y, return_counts=True)
        class_weight = {int(cls): 1.0 / float(count) for cls, count in zip(classes, counts)}
        weights = torch.as_tensor([class_weight[int(label)] for label in y], dtype=torch.double)
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(int(seed))
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=generator)
        shuffle = False
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle, sampler=sampler)


def augment_crops(x: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    if not args.augment:
        return x
    out = x
    if args.noise_std > 0:
        out = out + torch.randn_like(out) * float(args.noise_std)
    mask_frac = float(args.time_mask_frac)
    if mask_frac > 0:
        n_times = int(out.shape[-1])
        mask_width = max(1, int(round(n_times * min(mask_frac, 0.95))))
        if mask_width < n_times:
            starts = torch.randint(0, n_times - mask_width + 1, (out.shape[0],), device=out.device)
            out = out.clone()
            for i, start in enumerate(starts.tolist()):
                out[i, :, start : start + mask_width] = 0.0
    return out


@torch.no_grad()
def collect_crop_logits(model: torch.nn.Module, X: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    model.eval()
    chunks = []
    for start in range(0, X.shape[0], batch_size):
        xb = torch.as_tensor(X[start : start + batch_size], dtype=torch.float32, device=device)
        chunks.append(model(xb).detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def aggregate_trial_logits(crop_logits: np.ndarray, trial_ids: np.ndarray, n_trials: int) -> np.ndarray:
    logits = np.zeros((n_trials, crop_logits.shape[1]), dtype=np.float64)
    counts = np.zeros((n_trials, 1), dtype=np.float64)
    np.add.at(logits, trial_ids, crop_logits)
    np.add.at(counts, trial_ids, 1.0)
    return (logits / np.maximum(counts, 1.0)).astype(np.float32)


def trial_metrics(y_true: np.ndarray, trial_logits: np.ndarray, temperature: float = 1.0) -> dict[str, float]:
    proba = proba_from_logits(trial_logits, temperature)
    abstention_scores = 1.0 - proba.max(axis=1)
    ece, _ = compute_ece(proba, y_true)
    sel_acc, coverage = compute_selective_acc(proba, y_true, abstention_scores, 0.6)
    return {
        "accuracy": float(np.mean(proba.argmax(axis=1) == y_true)),
        "ECE": float(ece),
        "Brier": compute_brier(proba, y_true),
        "sel_acc_60": float(sel_acc),
        "coverage_60": float(coverage),
    }


def coverage_key(coverage: float) -> str:
    return f"{float(coverage):.2f}"


def coverage_suffix(coverage: float) -> str:
    return str(int(round(float(coverage) * 100)))


def normalized_proba(proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba, dtype=np.float64)
    denom = proba.sum(axis=1, keepdims=True)
    denom = np.where(denom <= 0.0, 1.0, denom)
    return proba / denom


def alpha_grid(step: float) -> np.ndarray:
    grid = np.arange(0.0, 1.0 + float(step) * 0.5, float(step), dtype=np.float64)
    grid = np.unique(np.clip(np.round(grid, 10), 0.0, 1.0))
    if grid[0] != 0.0:
        grid = np.insert(grid, 0, 0.0)
    if grid[-1] != 1.0:
        grid = np.append(grid, 1.0)
    return grid


def fit_alpha_on_validation(
    student_val: np.ndarray,
    teacher_val: np.ndarray,
    y_val: np.ndarray,
    step: float,
) -> tuple[float, float, list[dict[str, float]]]:
    records = []
    for alpha in alpha_grid(step):
        mix = float(alpha) * student_val + (1.0 - float(alpha)) * teacher_val
        brier = compute_brier(mix, y_val)
        records.append({"alpha": float(alpha), "validation_Brier": float(brier)})
    best = min(records, key=lambda row: (row["validation_Brier"], abs(1.0 - row["alpha"])))
    return float(best["alpha"]), float(best["validation_Brier"]), records


def disagreement(student: np.ndarray, teacher: np.ndarray, score: str) -> np.ndarray:
    student = normalized_proba(student)
    teacher = normalized_proba(teacher)
    if score == "dot":
        return (1.0 - np.sum(student * teacher, axis=1)).astype(np.float64)
    if score == "js":
        eps = 1e-12
        p = np.clip(student, eps, 1.0)
        q = np.clip(teacher, eps, 1.0)
        m = np.clip(0.5 * (p + q), eps, 1.0)
        kl_pm = np.sum(p * (np.log(p) - np.log(m)), axis=1)
        kl_qm = np.sum(q * (np.log(q) - np.log(m)), axis=1)
        return (0.5 * (kl_pm + kl_qm)).astype(np.float64)
    raise ValueError(f"Unknown disagreement score: {score}")


def reliability_from_disagreement(disagreement_scores: np.ndarray, normalizer: float) -> np.ndarray:
    denom = max(float(normalizer), 1e-12)
    return (1.0 - np.clip(np.asarray(disagreement_scores, dtype=np.float64) / denom, 0.0, 1.0)).astype(np.float64)


def fit_coverage_thresholds(reliability: np.ndarray, target_coverages: list[float]) -> dict[str, float]:
    reliability = np.asarray(reliability, dtype=np.float64).reshape(-1)
    if reliability.size == 0:
        return {coverage_key(coverage): float("nan") for coverage in target_coverages}
    ordered = np.sort(reliability)[::-1]
    thresholds: dict[str, float] = {}
    for coverage in target_coverages:
        n_keep = int(np.ceil(float(coverage) * reliability.size))
        n_keep = min(max(n_keep, 1), reliability.size)
        thresholds[coverage_key(coverage)] = float(ordered[n_keep - 1])
    return thresholds


def subset_metrics(y_true: np.ndarray, proba: np.ndarray, covered: np.ndarray) -> dict[str, float]:
    covered = np.asarray(covered, dtype=bool)
    if not covered.any():
        return {"accuracy": float("nan"), "ECE": float("nan"), "Brier": float("nan")}
    p = proba[covered]
    y = y_true[covered]
    ece, _ = compute_ece(p, y)
    return {
        "accuracy": float(np.mean(p.argmax(axis=1) == y)),
        "ECE": float(ece),
        "Brier": compute_brier(p, y),
    }


def metrics_from_proba(
    y_true: np.ndarray,
    proba: np.ndarray,
    reliability: np.ndarray,
    thresholds: dict[str, float],
    target_coverages: list[float],
    validation_brier: float,
    alpha_selection_validation_brier: float,
) -> dict[str, Any]:
    proba = normalized_proba(proba)
    y_true = np.asarray(y_true, dtype=np.int64)
    ece, _ = compute_ece(proba, y_true)
    out: dict[str, Any] = {
        "accuracy": float(np.mean(proba.argmax(axis=1) == y_true)),
        "ECE": float(ece),
        "Brier": compute_brier(proba, y_true),
        "validation_Brier": float(validation_brier),
        "alpha_selection_validation_Brier": float(alpha_selection_validation_brier),
        "selective": {},
    }
    reliability = np.asarray(reliability, dtype=np.float64).reshape(-1)
    for coverage in target_coverages:
        key = coverage_key(coverage)
        suffix = coverage_suffix(coverage)
        threshold = float(thresholds[key])
        covered = reliability >= threshold
        covered_metrics = subset_metrics(y_true, proba, covered)
        actual_coverage = float(covered.mean()) if covered.size else float("nan")
        out[f"sel_acc_{suffix}"] = float(covered_metrics["accuracy"])
        out[f"coverage_{suffix}"] = actual_coverage
        out["selective"][key] = {
            "threshold": threshold,
            "coverage": actual_coverage,
            "accuracy": float(covered_metrics["accuracy"]),
            "ECE": float(covered_metrics["ECE"]),
            "Brier": float(covered_metrics["Brier"]),
        }
    return out


def alpha_orientation(alpha: float) -> str:
    if alpha >= 0.65:
        return "student-heavy"
    if alpha <= 0.35:
        return "teacher-heavy"
    return "balanced"


def calibration_anchor_useful(calibration: dict[str, Any]) -> tuple[bool, list[str]]:
    eval_metrics = calibration["eval"]
    student = eval_metrics["student_only"]
    mixture = eval_metrics["mixture"]
    gated = eval_metrics["disagreement_gated_mixture"]
    reasons = []
    accuracy_loss = float(student["accuracy"] - mixture["accuracy"])
    if accuracy_loss <= 0.02 and float(mixture["Brier"]) < float(student["Brier"]):
        reasons.append("mixture_improves_Brier_without_meaningful_accuracy_loss")
    if accuracy_loss <= 0.02 and float(mixture["ECE"]) < float(student["ECE"]):
        reasons.append("mixture_improves_ECE_without_meaningful_accuracy_loss")
    for coverage in calibration["target_coverages"]:
        suffix = coverage_suffix(float(coverage))
        gate_acc = float(gated.get(f"sel_acc_{suffix}", float("nan")))
        mix_acc = float(mixture.get(f"sel_acc_{suffix}", float("nan")))
        if np.isfinite(gate_acc) and np.isfinite(mix_acc) and gate_acc > mix_acc:
            reasons.append(f"disagreement_gate_improves_selective_accuracy_at_{suffix}")
    return bool(reasons), reasons


def compute_calibration_anchor(
    data: dict[str, Any],
    teacher: dict[str, Any],
    val_logits: np.ndarray,
    eval_logits: np.ndarray,
    temperature: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    target_coverages = [float(coverage) for coverage in args.gate_coverages]
    student_val = normalized_proba(proba_from_logits(val_logits, temperature))
    student_eval = normalized_proba(proba_from_logits(eval_logits, temperature))
    teacher_val = normalized_proba(teacher["val_proba"])
    teacher_eval = normalized_proba(teacher["eval_proba"])

    alpha, selected_val_brier, records = fit_alpha_on_validation(
        student_val,
        teacher_val,
        data["y_val"],
        args.alpha_grid_step,
    )
    mix_val = normalized_proba(alpha * student_val + (1.0 - alpha) * teacher_val)
    mix_eval = normalized_proba(alpha * student_eval + (1.0 - alpha) * teacher_eval)

    val_disagreement = disagreement(student_val, teacher_val, args.disagreement_score)
    eval_disagreement = disagreement(student_eval, teacher_eval, args.disagreement_score)
    disagreement_normalizer = float(np.max(val_disagreement)) if val_disagreement.size else 1.0
    val_disagreement_reliability = reliability_from_disagreement(val_disagreement, disagreement_normalizer)
    eval_disagreement_reliability = reliability_from_disagreement(eval_disagreement, disagreement_normalizer)

    confidence_thresholds = {
        "student_only": fit_coverage_thresholds(student_val.max(axis=1), target_coverages),
        "teacher_only": fit_coverage_thresholds(teacher_val.max(axis=1), target_coverages),
        "mixture": fit_coverage_thresholds(mix_val.max(axis=1), target_coverages),
    }
    disagreement_thresholds = fit_coverage_thresholds(val_disagreement_reliability, target_coverages)

    student_val_brier = compute_brier(student_val, data["y_val"])
    teacher_val_brier = compute_brier(teacher_val, data["y_val"])
    mix_val_brier = compute_brier(mix_val, data["y_val"])
    validation = {
        "student_only": metrics_from_proba(
            data["y_val"],
            student_val,
            student_val.max(axis=1),
            confidence_thresholds["student_only"],
            target_coverages,
            student_val_brier,
            selected_val_brier,
        ),
        "teacher_only": metrics_from_proba(
            data["y_val"],
            teacher_val,
            teacher_val.max(axis=1),
            confidence_thresholds["teacher_only"],
            target_coverages,
            teacher_val_brier,
            selected_val_brier,
        ),
        "mixture": metrics_from_proba(
            data["y_val"],
            mix_val,
            mix_val.max(axis=1),
            confidence_thresholds["mixture"],
            target_coverages,
            mix_val_brier,
            selected_val_brier,
        ),
        "disagreement_gated_mixture": metrics_from_proba(
            data["y_val"],
            mix_val,
            val_disagreement_reliability,
            disagreement_thresholds,
            target_coverages,
            mix_val_brier,
            selected_val_brier,
        ),
    }
    evaluation = {
        "student_only": metrics_from_proba(
            data["y_eval"],
            student_eval,
            student_eval.max(axis=1),
            confidence_thresholds["student_only"],
            target_coverages,
            student_val_brier,
            selected_val_brier,
        ),
        "teacher_only": metrics_from_proba(
            data["y_eval"],
            teacher_eval,
            teacher_eval.max(axis=1),
            confidence_thresholds["teacher_only"],
            target_coverages,
            teacher_val_brier,
            selected_val_brier,
        ),
        "mixture": metrics_from_proba(
            data["y_eval"],
            mix_eval,
            mix_eval.max(axis=1),
            confidence_thresholds["mixture"],
            target_coverages,
            mix_val_brier,
            selected_val_brier,
        ),
        "disagreement_gated_mixture": metrics_from_proba(
            data["y_eval"],
            mix_eval,
            eval_disagreement_reliability,
            disagreement_thresholds,
            target_coverages,
            mix_val_brier,
            selected_val_brier,
        ),
    }
    out: dict[str, Any] = {
        "enabled": True,
        "thesis": "Classical decoders as competence and calibration anchors for reliability-gated neural uncertainty in small-sample motor imagery decoding.",
        "alpha": float(alpha),
        "alpha_orientation": alpha_orientation(alpha),
        "alpha_grid_step": float(args.alpha_grid_step),
        "alpha_fit_split": "A0xT_validation",
        "alpha_selection": {
            "objective": "minimize_validation_Brier",
            "selected_validation_Brier": float(selected_val_brier),
            "grid": records,
        },
        "target_coverages": target_coverages,
        "confidence_thresholds": confidence_thresholds,
        "disagreement_gate": {
            "score": args.disagreement_score,
            "definition": "1 - dot(p_student, p_teacher)" if args.disagreement_score == "dot" else "Jensen-Shannon divergence",
            "normalizer_fit_split": "A0xT_validation",
            "normalizer": disagreement_normalizer,
            "threshold_fit_split": "A0xT_validation",
            "thresholds": disagreement_thresholds,
            "validation_disagreement_mean": float(np.mean(val_disagreement)),
            "eval_disagreement_mean": float(np.mean(eval_disagreement)),
        },
        "validation": validation,
        "eval": evaluation,
        "audit": {
            "alpha_fit_split": "A0xT_validation",
            "alpha_eval_labels_used": False,
            "gate_threshold_fit_split": "A0xT_validation",
            "gate_eval_labels_used": False,
            "teacher_fit_split": "A0xT_train",
            "eval_teacher_used_as_predictions_only": True,
            "student_probabilities": "trial_level_crop_averaged_logits_after_validation_temperature_scaling",
            "teacher_probabilities_validation": "classical_teacher_predictions_on_A0xT_validation",
            "teacher_probabilities_eval": "classical_teacher_predictions_on_A0xE_from_A0xT_train_fit",
            "eval_labels_used_for": "final_metrics_only",
        },
    }
    useful, reasons = calibration_anchor_useful(out)
    out["useful"] = useful
    out["usefulness_reasons"] = reasons
    return out


def validation_loss(trial_logits: np.ndarray, y: np.ndarray, teacher_proba: np.ndarray | None, args: argparse.Namespace, anchored: bool) -> float:
    logits_t = torch.as_tensor(trial_logits, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.long)
    loss = F.cross_entropy(logits_t, y_t)
    if anchored and teacher_proba is not None and args.lambda_distill > 0:
        teacher_t = torch.as_tensor(teacher_proba, dtype=torch.float32)
        loss = loss + float(args.lambda_distill) * distill_loss(logits_t, teacher_t, args.distill_temperature)
    return float(loss.item())


def build_model(X_crops: np.ndarray, args: argparse.Namespace, variant: str) -> torch.nn.Module:
    cfg = CompetenceConfig(
        n_channels=int(X_crops.shape[1]),
        n_times=int(X_crops.shape[2]),
        n_classes=N_CLASSES,
        variant=CROPPED_TO_RAW[variant],
        dropout=float(args.dropout),
    )
    return build_competence_model(cfg)


def train_model(data: dict[str, Any], args: argparse.Namespace, variant: str, teacher: dict[str, Any], device: str) -> dict[str, Any]:
    anchored = variant in ANCHOR_TO_RAW
    view = preprocessing_view(data, args, teacher, anchored)
    variant_offset = sum((i + 1) * ord(ch) for i, ch in enumerate(variant)) % 1000
    subject_seed = int(args.seed + int(data["subject"]) * 101 + variant_offset)
    torch.manual_seed(subject_seed)
    np.random.seed(subject_seed)

    model = build_model(view["X_train_crops"], args, variant).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs)) if args.cosine_lr else None
    loader = make_loader(
        view["X_train_crops"],
        view["y_train_crops"],
        view["train_teacher_crops"],
        args.batch_size,
        shuffle=True,
        balanced_sampler=bool(args.balanced_sampler),
        seed=subject_seed,
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")
    best_val_brier = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        train_ce_losses = []
        train_distill_losses = []
        for batch in loader:
            xb = augment_crops(batch[0].to(device), args)
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
            train_ce_losses.append(float(ce.detach().cpu().item()))
            train_distill_losses.append(float(kd.detach().cpu().item()))
        if scheduler is not None:
            scheduler.step()

        val_crop_logits = collect_crop_logits(model, view["X_val_crops"], device, args.batch_size)
        val_logits = aggregate_trial_logits(val_crop_logits, view["val_trial_ids"], data["y_val"].shape[0])
        val_loss = validation_loss(val_logits, data["y_val"], teacher["val_proba"] if anchored else None, args, anchored)
        val_metrics = trial_metrics(data["y_val"], val_logits)
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(np.mean(train_losses)),
                "train_ce_loss": float(np.mean(train_ce_losses)),
                "train_distill_loss": float(np.mean(train_distill_losses)),
                "val_loss": float(val_loss),
                "val_accuracy": float(val_metrics["accuracy"]),
                "val_brier": float(val_metrics["Brier"]),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_val_brier = val_metrics["Brier"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_logits = aggregate_trial_logits(
        collect_crop_logits(model, view["X_val_crops"], device, args.batch_size),
        view["val_trial_ids"],
        data["y_val"].shape[0],
    )
    eval_logits = aggregate_trial_logits(
        collect_crop_logits(model, view["X_eval_crops"], device, args.batch_size),
        view["eval_trial_ids"],
        data["y_eval"].shape[0],
    )
    temperature, temp_at_upper = fit_temperature_from_logits(val_logits, data["y_val"], args.temperature_max)
    val_metrics = trial_metrics(data["y_val"], val_logits, temperature)
    eval_metrics = trial_metrics(data["y_eval"], eval_logits, temperature)
    calibration_anchor = (
        compute_calibration_anchor(data, teacher, val_logits, eval_logits, temperature, args)
        if args.calibration_anchor
        else {
            "enabled": False,
            "thesis": "Classical decoders as competence and calibration anchors for reliability-gated neural uncertainty in small-sample motor imagery decoding.",
        }
    )
    return {
        "subject": int(data["subject"]),
        "variant": variant,
        "raw_architecture_variant": CROPPED_TO_RAW[variant],
        "anchored": bool(anchored),
        "teacher": teacher["name"] if anchored else None,
        "lambda_distill": float(args.lambda_distill) if anchored else 0.0,
        "distill_temperature": float(args.distill_temperature) if anchored else None,
        "metrics": eval_metrics,
        "validation_metrics": val_metrics,
        "calibration_anchor": calibration_anchor,
        "model_info": {
            **view["preprocessing"],
            "crop_training": view["crop_audit"],
            "augmentation": {
                "enabled": bool(args.augment),
                "fit_split": "A0xT_train_crops_only",
                "validation_enabled": False,
                "eval_enabled": False,
                "gaussian_noise_std": float(args.noise_std) if args.augment else 0.0,
                "time_mask_frac": float(args.time_mask_frac) if args.augment else 0.0,
            },
            "sampler": {
                "balanced_sampler": bool(args.balanced_sampler),
                "unit": "training_crops",
                "validation_sampler": False,
                "eval_sampler": False,
            },
            "scheduler": {
                "cosine_lr": bool(args.cosine_lr),
                "initial_lr": float(args.lr),
                "last_lr": float(optimizer.param_groups[0]["lr"]),
                "weight_decay": float(args.weight_decay),
            },
            "temperature_scaling": {
                "enabled": True,
                "fit_split": "A0xT_validation_trial_logits",
                "eval_used_for_fit": False,
                "temperature": float(temperature),
                "temperature_at_upper_clamp": temp_at_upper,
            },
            "distillation": {
                "enabled": bool(anchored),
                "teacher": teacher["name"] if anchored else None,
                "lambda_distill": float(args.lambda_distill) if anchored else 0.0,
                "distill_temperature": float(args.distill_temperature) if anchored else None,
                "teacher_fit_split": "A0xT_train" if anchored else None,
                "teacher_targets": "A0xT_train_crops_repeated_from_train_trials" if anchored else None,
                "validation_teacher_used_for_model_selection": bool(anchored),
                "eval_teacher_targets_used": False,
            },
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss),
            "best_val_brier": float(best_val_brier),
            "epochs_ran": len(history),
            "history": history,
            "chance_accuracy": CHANCE_ACC,
            "chance_brier": CHANCE_BRIER,
        },
    }


def get_current_control(row: dict[str, Any], by_subject_variant: dict[tuple[int, str], dict[str, Any]]) -> dict[str, Any] | None:
    raw_variant = ANCHOR_TO_RAW.get(row["variant"])
    if not raw_variant:
        return None
    control = by_subject_variant.get((int(row["subject"]), raw_variant))
    return control["metrics"] if control else None


def compute_sanity_flags(
    row: dict[str, Any],
    baseline_subjects: dict[int, dict[str, dict[str, float]]],
    by_subject_variant: dict[tuple[int, str], dict[str, Any]],
    previous_anchor_subjects: dict[int, dict[str, dict[str, float]]],
    previous_cropped_subjects: dict[int, dict[str, dict[str, float]]],
) -> list[str]:
    info = row["model_info"]
    history = info.get("history") or []
    flags = []
    if info.get("best_epoch") == 1:
        flags.append("best_epoch=1")
    if len(history) >= 2 and history[-1]["train_loss"] >= history[0]["train_loss"]:
        flags.append("train_loss_not_decreasing")
    if row["validation_metrics"]["accuracy"] <= CHANCE_ACC + 0.05:
        flags.append("validation_accuracy_at_chance")
    if row["validation_metrics"]["Brier"] >= CHANCE_BRIER:
        flags.append("validation_brier_not_below_chance")
    if info.get("temperature_scaling", {}).get("temperature_at_upper_clamp"):
        flags.append("temperature_upper_clamp")
    eegnet = baseline_subjects.get(int(row["subject"]), {}).get("eegnet_ts")
    if eegnet and row["metrics"]["accuracy"] < eegnet["accuracy"]:
        flags.append("eval_accuracy_below_eegnet_ts")
    control = get_current_control(row, by_subject_variant)
    if control and row["metrics"]["accuracy"] < control["accuracy"] and row["metrics"]["Brier"] > control["Brier"]:
        flags.append("anchor_worse_than_matched_raw_accuracy_and_brier")
    previous_variant = PREVIOUS_NONCROPPED_MATCH.get(row["variant"])
    previous = previous_anchor_subjects.get(int(row["subject"]), {}).get(previous_variant) if previous_variant else None
    if previous and row["metrics"]["accuracy"] <= previous["accuracy"] and row["metrics"]["Brier"] >= previous["Brier"]:
        flags.append("cropped_fails_previous_noncropped_anchor_accuracy_and_brier")
    if row["variant"] in MULTISCALE_PREVIOUS_CROPPED_MATCHES:
        controls = []
        for control_variant in MULTISCALE_PREVIOUS_CROPPED_MATCHES[row["variant"]]:
            control = by_subject_variant.get((int(row["subject"]), control_variant))
            if control:
                controls.append(control["metrics"])
            else:
                previous_control = previous_cropped_subjects.get(int(row["subject"]), {}).get(control_variant)
                if previous_control:
                    controls.append(previous_control)
        if controls:
            best_acc = max(control["accuracy"] for control in controls)
            best_brier = min(control["Brier"] for control in controls)
            if row["metrics"]["accuracy"] <= best_acc and row["metrics"]["Brier"] >= best_brier:
                flags.append("multiscale_worse_than_previous_cropped_control_accuracy_and_brier")
    return flags


def aggregate(subject_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for variant in sorted({row["variant"] for row in subject_rows}):
        rows = [row for row in subject_rows if row["variant"] == variant]
        out: dict[str, Any] = {
            "variant": variant,
            "anchored": bool(rows[0]["anchored"]),
            "teacher": rows[0]["teacher"],
            "n_subjects": len(rows),
        }
        for metric in ("accuracy", "ECE", "Brier", "sel_acc_60", "coverage_60"):
            values = np.array([float(row["metrics"][metric]) for row in rows], dtype=float)
            out[metric] = {"mean": float(values.mean()), "std": float(values.std(ddof=0))}
        for metric in ("val_accuracy", "val_Brier"):
            key = "accuracy" if metric == "val_accuracy" else "Brier"
            values = np.array([float(row["validation_metrics"][key]) for row in rows], dtype=float)
            out[metric] = {"mean": float(values.mean()), "std": float(values.std(ddof=0))}
        for metric in ("best_epoch", "best_val_loss", "best_val_brier", "temperature"):
            if metric == "temperature":
                values = np.array([float(row["model_info"]["temperature_scaling"]["temperature"]) for row in rows], dtype=float)
            else:
                values = np.array([float(row["model_info"][metric]) for row in rows], dtype=float)
            out[metric] = {"mean": float(values.mean()), "std": float(values.std(ddof=0))}
        out["temperature_upper_clamp_count"] = int(sum(row["model_info"]["temperature_scaling"]["temperature_at_upper_clamp"] for row in rows))
        out["sanity_flag_count"] = int(sum(len(row.get("sanity_flags", [])) for row in rows))
        summary.append(out)
    return summary


def finite_mean(values: list[float]) -> float | None:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else None


def calibration_anchor_summary(subject_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [row for row in subject_rows if row.get("calibration_anchor", {}).get("enabled")]
    summary = []
    for variant in sorted({row["variant"] for row in rows}):
        variant_rows = [row for row in rows if row["variant"] == variant]
        anchors = [row["calibration_anchor"] for row in variant_rows]
        target_coverages = anchors[0]["target_coverages"] if anchors else []
        alpha_values = [float(anchor["alpha"]) for anchor in anchors]
        orientation_counts = {
            orientation: int(sum(anchor["alpha_orientation"] == orientation for anchor in anchors))
            for orientation in ("student-heavy", "balanced", "teacher-heavy")
        }
        student = [anchor["eval"]["student_only"] for anchor in anchors]
        mixture = [anchor["eval"]["mixture"] for anchor in anchors]
        gated = [anchor["eval"]["disagreement_gated_mixture"] for anchor in anchors]
        out: dict[str, Any] = {
            "variant": variant,
            "n_subjects": len(anchors),
            "alpha_mean": float(np.mean(alpha_values)),
            "alpha_std": float(np.std(alpha_values, ddof=0)),
            "alpha_orientation": alpha_orientation(float(np.mean(alpha_values))),
            "alpha_orientation_counts": orientation_counts,
            "student_accuracy_mean": finite_mean([float(metric["accuracy"]) for metric in student]),
            "mixture_accuracy_mean": finite_mean([float(metric["accuracy"]) for metric in mixture]),
            "accuracy_delta_mean": finite_mean([float(mix["accuracy"]) - float(stu["accuracy"]) for stu, mix in zip(student, mixture)]),
            "student_Brier_mean": finite_mean([float(metric["Brier"]) for metric in student]),
            "mixture_Brier_mean": finite_mean([float(metric["Brier"]) for metric in mixture]),
            "Brier_delta_mean": finite_mean([float(mix["Brier"]) - float(stu["Brier"]) for stu, mix in zip(student, mixture)]),
            "student_ECE_mean": finite_mean([float(metric["ECE"]) for metric in student]),
            "mixture_ECE_mean": finite_mean([float(metric["ECE"]) for metric in mixture]),
            "ECE_delta_mean": finite_mean([float(mix["ECE"]) - float(stu["ECE"]) for stu, mix in zip(student, mixture)]),
            "validation_Brier_selected_mean": finite_mean([float(anchor["alpha_selection"]["selected_validation_Brier"]) for anchor in anchors]),
            "subjects_useful": int(sum(bool(anchor["useful"]) for anchor in anchors)),
        }
        for coverage in target_coverages:
            suffix = coverage_suffix(float(coverage))
            out[f"mixture_sel_acc_{suffix}_mean"] = finite_mean([float(metric.get(f"sel_acc_{suffix}", float("nan"))) for metric in mixture])
            out[f"gated_sel_acc_{suffix}_mean"] = finite_mean([float(metric.get(f"sel_acc_{suffix}", float("nan"))) for metric in gated])
            out[f"gated_coverage_{suffix}_mean"] = finite_mean([float(metric.get(f"coverage_{suffix}", float("nan"))) for metric in gated])
            out[f"gate_sel_acc_delta_{suffix}_mean"] = finite_mean(
                [
                    float(gate.get(f"sel_acc_{suffix}", float("nan"))) - float(mix.get(f"sel_acc_{suffix}", float("nan")))
                    for mix, gate in zip(mixture, gated)
                ]
            )
        out["useful"] = bool(
            out["subjects_useful"] > 0
            or ((out["Brier_delta_mean"] is not None and out["Brier_delta_mean"] < 0.0) or (out["ECE_delta_mean"] is not None and out["ECE_delta_mean"] < 0.0))
            and (out["accuracy_delta_mean"] is None or out["accuracy_delta_mean"] >= -0.02)
        )
        summary.append(out)
    return summary


def paired_against_variant(subject_rows: list[dict[str, Any]], controls: dict[int, dict[str, dict[str, float]]], control_variant: str) -> dict[str, Any]:
    out = {}
    for variant in sorted({row["variant"] for row in subject_rows}):
        rows = [row for row in subject_rows if row["variant"] == variant and control_variant in controls.get(row["subject"], {})]
        if not rows:
            continue
        out[variant] = {}
        for metric in ("accuracy", "ECE", "Brier", "sel_acc_60"):
            values = np.array([row["metrics"][metric] - controls[row["subject"]][control_variant][metric] for row in rows], dtype=float)
            out[variant][metric] = {"mean": float(values.mean()), "std": float(values.std(ddof=0)), "n": int(values.size)}
    return out


def anchor_usefulness(subject_rows: list[dict[str, Any]], by_subject_variant: dict[tuple[int, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for anchor, raw in ANCHOR_TO_RAW.items():
        anchors = [row for row in subject_rows if row["variant"] == anchor and (row["subject"], raw) in by_subject_variant]
        if not anchors:
            continue
        acc = np.array([row["metrics"]["accuracy"] - by_subject_variant[(row["subject"], raw)]["metrics"]["accuracy"] for row in anchors], dtype=float)
        brier = np.array([row["metrics"]["Brier"] - by_subject_variant[(row["subject"], raw)]["metrics"]["Brier"] for row in anchors], dtype=float)
        ece = np.array([row["metrics"]["ECE"] - by_subject_variant[(row["subject"], raw)]["metrics"]["ECE"] for row in anchors], dtype=float)
        rows.append(
            {
                "anchor_variant": anchor,
                "matched_raw": raw,
                "n_subjects": len(anchors),
                "accuracy_delta_mean": float(acc.mean()),
                "brier_delta_mean": float(brier.mean()),
                "ece_delta_mean": float(ece.mean()),
                "subjects_accuracy_improved": int(np.sum(acc > 0)),
                "subjects_brier_improved": int(np.sum(brier < 0)),
                "anchor_useful": bool((acc.mean() > 0 or brier.mean() < 0 or np.sum(acc > 0) >= 5) and brier.mean() <= 0.02),
            }
        )
    return rows


def cropped_usefulness(subject_rows: list[dict[str, Any]], previous_anchor_subjects: dict[int, dict[str, dict[str, float]]]) -> list[dict[str, Any]]:
    rows = []
    for cropped, previous in PREVIOUS_NONCROPPED_MATCH.items():
        current = [row for row in subject_rows if row["variant"] == cropped and previous in previous_anchor_subjects.get(row["subject"], {})]
        if not current:
            continue
        acc = np.array([row["metrics"]["accuracy"] - previous_anchor_subjects[row["subject"]][previous]["accuracy"] for row in current], dtype=float)
        brier = np.array([row["metrics"]["Brier"] - previous_anchor_subjects[row["subject"]][previous]["Brier"] for row in current], dtype=float)
        ece = np.array([row["metrics"]["ECE"] - previous_anchor_subjects[row["subject"]][previous]["ECE"] for row in current], dtype=float)
        rows.append(
            {
                "cropped_variant": cropped,
                "previous_noncropped": previous,
                "n_subjects": len(current),
                "accuracy_delta_mean": float(acc.mean()),
                "brier_delta_mean": float(brier.mean()),
                "ece_delta_mean": float(ece.mean()),
                "subjects_accuracy_improved": int(np.sum(acc > 0)),
                "subjects_brier_improved": int(np.sum(brier < 0)),
                "cropping_useful": bool((acc.mean() > 0 or brier.mean() < 0 or np.sum(acc > 0) >= 5) and brier.mean() <= 0.02),
            }
        )
    return rows


def get_cropped_control_metrics(
    subject: int,
    variant: str,
    by_subject_variant: dict[tuple[int, str], dict[str, Any]],
    previous_cropped_subjects: dict[int, dict[str, dict[str, float]]],
) -> dict[str, float] | None:
    row = by_subject_variant.get((subject, variant))
    if row:
        return row["metrics"]
    return previous_cropped_subjects.get(subject, {}).get(variant)


def multiscale_usefulness(
    subject_rows: list[dict[str, Any]],
    by_subject_variant: dict[tuple[int, str], dict[str, Any]],
    previous_cropped_subjects: dict[int, dict[str, dict[str, float]]],
) -> list[dict[str, Any]]:
    rows = []
    for multiscale, controls in MULTISCALE_PREVIOUS_CROPPED_MATCHES.items():
        current = [row for row in subject_rows if row["variant"] == multiscale]
        for control_variant in controls:
            acc_delta = []
            brier_delta = []
            ece_delta = []
            for row in current:
                control = get_cropped_control_metrics(int(row["subject"]), control_variant, by_subject_variant, previous_cropped_subjects)
                if not control:
                    continue
                acc_delta.append(row["metrics"]["accuracy"] - control["accuracy"])
                brier_delta.append(row["metrics"]["Brier"] - control["Brier"])
                ece_delta.append(row["metrics"]["ECE"] - control["ECE"])
            if not acc_delta:
                continue
            acc = np.asarray(acc_delta, dtype=float)
            brier = np.asarray(brier_delta, dtype=float)
            ece = np.asarray(ece_delta, dtype=float)
            rows.append(
                {
                    "multiscale_variant": multiscale,
                    "previous_cropped_control": control_variant,
                    "n_subjects": int(acc.size),
                    "accuracy_delta_mean": float(acc.mean()),
                    "accuracy_delta_median": float(np.median(acc)),
                    "brier_delta_mean": float(brier.mean()),
                    "brier_delta_median": float(np.median(brier)),
                    "ece_delta_mean": float(ece.mean()),
                    "subjects_accuracy_improved": int(np.sum(acc > 0)),
                    "subjects_brier_improved": int(np.sum(brier < 0)),
                    "multiscale_useful": bool((acc.mean() > 0 or brier.mean() < 0 or np.sum(acc > 0) >= 5) and brier.mean() <= 0.02),
                }
            )
    return rows


def sign_test_pvalue(values: np.ndarray, beneficial_positive: bool) -> float | None:
    nonzero = values[values != 0]
    n = int(nonzero.size)
    if n == 0:
        return None
    successes = int(np.sum(nonzero > 0)) if beneficial_positive else int(np.sum(nonzero < 0))
    tail = sum(math.comb(n, k) for k in range(0, min(successes, n - successes) + 1)) / (2**n)
    return float(min(1.0, 2.0 * tail))


def paired_test(values: np.ndarray, beneficial_positive: bool, n_sufficient: bool) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    out: dict[str, Any] = {
        "n": int(values.size),
        "mean_delta": float(values.mean()) if values.size else None,
        "median_delta": float(np.median(values)) if values.size else None,
        "subjects_improved": int(np.sum(values > 0)) if beneficial_positive else int(np.sum(values < 0)),
        "pilot_only": bool(values.size < N_SUBJECTS),
        "test": None,
        "p_value": None,
    }
    if not n_sufficient or values.size < N_SUBJECTS:
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


def add_evidence_entry(
    rows: list[dict[str, Any]],
    comparison: str,
    variant: str,
    metric: str,
    values: list[float],
    beneficial_positive: bool,
    n_sufficient: bool,
) -> None:
    if not values:
        return
    test = paired_test(np.asarray(values, dtype=float), beneficial_positive, n_sufficient)
    rows.append(
        {
            "comparison": comparison,
            "variant": variant,
            "metric": metric,
            "beneficial_direction": "positive" if beneficial_positive else "negative",
            **test,
        }
    )


def significance_evidence(
    subject_rows: list[dict[str, Any]],
    baseline_subjects: dict[int, dict[str, dict[str, float]]],
    graph_rescue_subjects: dict[int, dict[str, dict[str, float]]],
    by_subject_variant: dict[tuple[int, str], dict[str, Any]],
    previous_cropped_subjects: dict[int, dict[str, dict[str, float]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n_sufficient = len({int(row["subject"]) for row in subject_rows}) >= N_SUBJECTS
    controls = (
        ("vs_eegnet_ts", baseline_subjects, "eegnet_ts"),
        ("vs_lda_svm_vote", baseline_subjects, "lda_svm_vote"),
        ("vs_graph_ce_raw", graph_rescue_subjects, "graph_ce_raw"),
        ("vs_graph_ce_no_graph_ablation", graph_rescue_subjects, "graph_ce_no_graph_ablation"),
    )
    for variant in sorted({row["variant"] for row in subject_rows}):
        variant_rows = [row for row in subject_rows if row["variant"] == variant]
        for comparison, control_subjects, control_variant in controls:
            for metric, beneficial_positive in (("accuracy", True), ("Brier", False)):
                values = [
                    row["metrics"][metric] - control_subjects[int(row["subject"])][control_variant][metric]
                    for row in variant_rows
                    if control_variant in control_subjects.get(int(row["subject"]), {})
                ]
                add_evidence_entry(rows, comparison, variant, metric, values, beneficial_positive, n_sufficient)
    for anchor, raw in ANCHOR_TO_RAW.items():
        anchors = [row for row in subject_rows if row["variant"] == anchor and (int(row["subject"]), raw) in by_subject_variant]
        for metric, beneficial_positive in (("accuracy", True), ("Brier", False)):
            values = [row["metrics"][metric] - by_subject_variant[(int(row["subject"]), raw)]["metrics"][metric] for row in anchors]
            add_evidence_entry(rows, "anchored_vs_matched_raw", anchor, metric, values, beneficial_positive, n_sufficient)
    for multiscale, controls in MULTISCALE_PREVIOUS_CROPPED_MATCHES.items():
        current = [row for row in subject_rows if row["variant"] == multiscale]
        for control_variant in controls:
            for metric, beneficial_positive in (("accuracy", True), ("Brier", False)):
                values = []
                for row in current:
                    control = get_cropped_control_metrics(int(row["subject"]), control_variant, by_subject_variant, previous_cropped_subjects)
                    if control:
                        values.append(row["metrics"][metric] - control[metric])
                add_evidence_entry(rows, f"multiscale_vs_{control_variant}", multiscale, metric, values, beneficial_positive, n_sufficient)
    return rows


def beats_graph_rescue(row: dict[str, Any], graph_rescue_summary: dict[str, dict[str, float]]) -> bool:
    rescue = graph_rescue_summary.get("graph_ce_no_graph_ablation")
    if not rescue:
        return False
    return bool(row["accuracy"]["mean"] > rescue["accuracy"] or row["Brier"]["mean"] < rescue["Brier"])


def decide_verdict(summary: list[dict[str, Any]], baseline_summary: dict[str, dict[str, float]], graph_rescue_summary: dict[str, dict[str, float]]) -> tuple[str, str]:
    eegnet = baseline_summary.get("eegnet_ts")
    vote = baseline_summary.get("lda_svm_vote")
    anchored = [row for row in summary if row["anchored"]]
    if not eegnet or not anchored:
        return "failed", "not_yet"
    best = max(anchored, key=lambda r: (r["accuracy"]["mean"], -r["Brier"]["mean"]))
    vote_gap_narrowed = True
    if vote:
        vote_gap_narrowed = (
            abs(vote["accuracy"] - best["accuracy"]["mean"]) < abs(vote["accuracy"] - eegnet["accuracy"])
            or abs(best["Brier"]["mean"] - vote["Brier"]) < abs(eegnet["Brier"] - vote["Brier"])
        )
    competent = bool(
        best["accuracy"]["mean"] > eegnet["accuracy"]
        and best["Brier"]["mean"] < eegnet["Brier"]
        and beats_graph_rescue(best, graph_rescue_summary)
        and vote_gap_narrowed
        and best["sanity_flag_count"] < best["n_subjects"]
    )
    if competent and vote and best["accuracy"]["mean"] < vote["accuracy"] and best["Brier"]["mean"] > vote["Brier"]:
        return "classical_still_best", "not_yet"
    if competent:
        edl_verdict = "justified_later" if best["n_subjects"] == N_SUBJECTS else "not_yet"
        if best["variant"] == "anchor_multiscale_temporal_ce":
            return "anchored_multiscale_candidate", edl_verdict
        if "cropped" in best["variant"]:
            return "anchored_cropped_candidate", edl_verdict
        return "anchored_deep_no_graph", edl_verdict
    if vote and vote["accuracy"] >= best["accuracy"]["mean"]:
        return "classical_still_best", "not_yet"
    return "failed", "not_yet"


def write_csvs(results_dir: Path, subject_rows: list[dict[str, Any]], summary: list[dict[str, Any]]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    subject_fields = [
        "subject",
        "variant",
        "anchored",
        "teacher",
        "lambda_distill",
        "distill_temperature",
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
    with (results_dir / "cropped_anchor_subject_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=subject_fields)
        writer.writeheader()
        for row in sorted(subject_rows, key=lambda r: (r["variant"], r["subject"])):
            info = row["model_info"]
            writer.writerow(
                {
                    "subject": row["subject"],
                    "variant": row["variant"],
                    "anchored": row["anchored"],
                    "teacher": row["teacher"] or "",
                    "lambda_distill": row["lambda_distill"],
                    "distill_temperature": row["distill_temperature"] or "",
                    **row["metrics"],
                    "val_accuracy": row["validation_metrics"]["accuracy"],
                    "val_Brier": row["validation_metrics"]["Brier"],
                    "best_epoch": info["best_epoch"],
                    "best_val_loss": info["best_val_loss"],
                    "best_val_brier": info["best_val_brier"],
                    "temperature": info["temperature_scaling"]["temperature"],
                    "temperature_at_upper_clamp": info["temperature_scaling"]["temperature_at_upper_clamp"],
                    "sanity_flags": ";".join(row.get("sanity_flags", [])),
                }
            )

    fields = ["variant", "anchored", "teacher", "n_subjects"]
    for metric in ("accuracy", "ECE", "Brier", "sel_acc_60", "coverage_60", "val_accuracy", "val_Brier", "best_epoch", "best_val_loss", "best_val_brier", "temperature"):
        fields.extend([f"{metric}_mean", f"{metric}_std"])
    fields.extend(["temperature_upper_clamp_count", "sanity_flag_count"])
    with (results_dir / "cropped_anchor_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sorted(summary, key=lambda r: r["variant"]):
            flat = {"variant": row["variant"], "anchored": row["anchored"], "teacher": row["teacher"] or "", "n_subjects": row["n_subjects"]}
            for metric in ("accuracy", "ECE", "Brier", "sel_acc_60", "coverage_60", "val_accuracy", "val_Brier", "best_epoch", "best_val_loss", "best_val_brier", "temperature"):
                flat[f"{metric}_mean"] = row[metric]["mean"]
                flat[f"{metric}_std"] = row[metric]["std"]
            flat["temperature_upper_clamp_count"] = row["temperature_upper_clamp_count"]
            flat["sanity_flag_count"] = row["sanity_flag_count"]
            writer.writerow(flat)


def fmt_mean_std(row: dict[str, Any], metric: str) -> str:
    return f"{row[metric]['mean']:.4f} ({row[metric]['std']:.4f})"


def fmt_optional(value: Any, signed: bool = False) -> str:
    if value is None:
        return ""
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(value_f):
        return ""
    return f"{value_f:+.4f}" if signed else f"{value_f:.4f}"


def add_delta_table(lines: list[str], title: str, deltas: dict[str, Any]) -> None:
    if not deltas:
        return
    lines.extend(["", f"## {title}", "", "| variant | accuracy | ECE | Brier | sel_acc_60 | n |", "|---|---:|---:|---:|---:|---:|"])
    for variant, vals in sorted(deltas.items()):
        lines.append(
            f"| {variant} | {vals['accuracy']['mean']:+.4f} | {vals['ECE']['mean']:+.4f} | "
            f"{vals['Brier']['mean']:+.4f} | {vals['sel_acc_60']['mean']:+.4f} | {vals['accuracy']['n']} |"
        )


def add_evidence_table(lines: list[str], evidence_rows: list[dict[str, Any]]) -> None:
    lines.extend(
        [
            "",
            "## Significance / Evidence",
            "",
            "Runs with n<9 are pilot evidence only; p-values are reported only for full multi-subject paired evidence.",
            "",
            "| comparison | variant | metric | mean delta | median delta | improved | n | test | p | note |",
            "|---|---|---|---:|---:|---:|---:|---|---:|---|",
        ]
    )
    if not evidence_rows:
        lines.append("| none | none | none |  |  |  |  |  |  | no paired controls available |")
        return
    for row in evidence_rows:
        p_value = row["p_value"]
        note = "pilot evidence only" if row["pilot_only"] else "full paired evidence"
        p_text = "" if p_value is None else f"{p_value:.4f}"
        lines.append(
            f"| {row['comparison']} | {row['variant']} | {row['metric']} | "
            f"{row['mean_delta']:+.4f} | {row['median_delta']:+.4f} | "
            f"{row['subjects_improved']} | {row['n']} | {row['test'] or ''} | {p_text} | {note} |"
        )


def add_calibration_anchor_tables(lines: list[str], payload: dict[str, Any]) -> None:
    calibration_rows = payload.get("calibration_anchor_summary", [])
    lines.extend(
        [
            "",
            "## Calibration Anchor Mixture",
            "",
            "| variant | alpha | orientation | acc delta | Brier delta | ECE delta | selected val Brier | useful |",
            "|---|---:|---|---:|---:|---:|---:|---|",
        ]
    )
    if not calibration_rows:
        lines.append("| disabled |  |  |  |  |  |  | run with `--calibration-anchor` |")
    for row in calibration_rows:
        lines.append(
            f"| {row['variant']} | {row['alpha_mean']:.3f} ({row['alpha_std']:.3f}) | {row['alpha_orientation']} | "
            f"{fmt_optional(row['accuracy_delta_mean'], signed=True)} | {fmt_optional(row['Brier_delta_mean'], signed=True)} | "
            f"{fmt_optional(row['ECE_delta_mean'], signed=True)} | {fmt_optional(row['validation_Brier_selected_mean'])} | {row['useful']} |"
        )

    target_coverages = payload.get("gate_coverages", [0.60, 0.70, 0.80])
    header = "| variant | " + " | ".join(
        f"gate sel acc {coverage_suffix(float(coverage))} | mix sel acc {coverage_suffix(float(coverage))} | actual cov {coverage_suffix(float(coverage))}"
        for coverage in target_coverages
    ) + " |"
    sep = "|---" + "".join("|---:|---:|---:" for _ in target_coverages) + "|"
    lines.extend(["", "## Teacher-Student Disagreement Gate", "", header, sep])
    if not calibration_rows:
        lines.append("| disabled | " + " | ".join(" |  | " for _ in target_coverages) + "|")
    for row in calibration_rows:
        cells = [row["variant"]]
        for coverage in target_coverages:
            suffix = coverage_suffix(float(coverage))
            cells.extend(
                [
                    fmt_optional(row.get(f"gated_sel_acc_{suffix}_mean")),
                    fmt_optional(row.get(f"mixture_sel_acc_{suffix}_mean")),
                    fmt_optional(row.get(f"gated_coverage_{suffix}_mean")),
                ]
            )
        lines.append("| " + " | ".join(cells) + " |")

    useful = [row for row in calibration_rows if row.get("useful")]
    lines.extend(
        [
            "",
            "## Verdict on whether calibration anchoring is useful",
            "",
            (
                "Calibration anchoring is useful in this run: at least one variant improves Brier/ECE without a meaningful accuracy loss or improves disagreement-gated selective accuracy."
                if useful
                else "Calibration anchoring is not yet useful in this run under the predefined Brier/ECE and disagreement-gate criteria."
            ),
            "",
            "This is calibration novelty, not architecture novelty: no graph layers are added and EDL is not trained by this path.",
        ]
    )


def write_readout(results_dir: Path, payload: dict[str, Any]) -> str:
    verdict = payload["verdict"]
    if verdict == "anchored_multiscale_candidate":
        framing = "Classical decoders provide leakage-safe calibration anchors for a no-graph multiscale temporal CE student; this supports competence-first reliability gating while graph and EDL claims remain downstream."
    elif verdict == "anchored_cropped_candidate":
        framing = "Classical decoders provide leakage-safe calibration anchors that improve neural closed-set competence under crop-aggregated MI training, enabling reliability-gated uncertainty claims without eval leakage."
    elif verdict == "anchored_deep_no_graph":
        framing = "The protocol deliberately separates competence engineering from novelty claims: crop-aggregated neural students are promoted only as anchored CE decoders, while graph and EDL claims remain gated."
    elif verdict == "classical_still_best":
        framing = "Classical decoders remain the competence floor in small-sample MI; the contribution becomes a diagnostic framework and teacher-anchored route toward neural uncertainty."
    else:
        framing = "The cropped no-graph CE run did not clear the competence gates; classical anchors remain the reference floor and EDL remains deferred."
    lines = [
        "# Cropped Anchor Competence Readout",
        "",
        f"Verdict: **{verdict}**",
        f"EDL verdict: **{payload['edl_verdict']}**",
        "",
        "## Aggregate Metrics",
        "",
        "| variant | accuracy | ECE | Brier | sel_acc_60 | val acc | val Brier | flags |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(payload["metrics"], key=lambda r: r["variant"]):
        lines.append(
            f"| {row['variant']} | {fmt_mean_std(row, 'accuracy')} | {fmt_mean_std(row, 'ECE')} | "
            f"{fmt_mean_std(row, 'Brier')} | {fmt_mean_std(row, 'sel_acc_60')} | "
            f"{fmt_mean_std(row, 'val_accuracy')} | {fmt_mean_std(row, 'val_Brier')} | {row['sanity_flag_count']} |"
        )
    add_delta_table(lines, "Paired Deltas vs EEGNet+TS", payload["paired_deltas_vs_eegnet_ts"])
    add_delta_table(lines, "Paired Deltas vs LDA+SVM Vote", payload["paired_deltas_vs_lda_svm_vote"])
    add_delta_table(lines, "Paired Deltas vs Graph Rescue graph_ce_raw", payload["paired_deltas_vs_graph_ce_raw"])
    add_delta_table(lines, "Paired Deltas vs Graph Rescue graph_ce_no_graph_ablation", payload["paired_deltas_vs_graph_ce_no_graph_ablation"])
    lines.extend(["", "## Multiscale Usefulness", "", "| multiscale | previous cropped control | acc delta | Brier delta | ECE delta | acc improved | Brier improved | useful |", "|---|---|---:|---:|---:|---:|---:|---|"])
    for row in payload["multiscale_usefulness"]:
        lines.append(
            f"| {row['multiscale_variant']} | {row['previous_cropped_control']} | {row['accuracy_delta_mean']:+.4f} | "
            f"{row['brier_delta_mean']:+.4f} | {row['ece_delta_mean']:+.4f} | {row['subjects_accuracy_improved']}/{row['n_subjects']} | "
            f"{row['subjects_brier_improved']}/{row['n_subjects']} | {row['multiscale_useful']} |"
        )
    lines.extend(["", "## Cropped Usefulness", "", "| cropped | previous non-cropped | acc delta | Brier delta | ECE delta | acc improved | Brier improved | useful |", "|---|---|---:|---:|---:|---:|---:|---|"])
    for row in payload["cropped_usefulness"]:
        lines.append(
            f"| {row['cropped_variant']} | {row['previous_noncropped']} | {row['accuracy_delta_mean']:+.4f} | "
            f"{row['brier_delta_mean']:+.4f} | {row['ece_delta_mean']:+.4f} | {row['subjects_accuracy_improved']}/{row['n_subjects']} | "
            f"{row['subjects_brier_improved']}/{row['n_subjects']} | {row['cropping_useful']} |"
        )
    lines.extend(["", "## Anchor Usefulness", "", "| anchor | matched raw | acc delta | Brier delta | ECE delta | acc improved | Brier improved | useful |", "|---|---|---:|---:|---:|---:|---:|---|"])
    for row in payload["anchor_usefulness"]:
        lines.append(
            f"| {row['anchor_variant']} | {row['matched_raw']} | {row['accuracy_delta_mean']:+.4f} | "
            f"{row['brier_delta_mean']:+.4f} | {row['ece_delta_mean']:+.4f} | {row['subjects_accuracy_improved']}/{row['n_subjects']} | "
            f"{row['subjects_brier_improved']}/{row['n_subjects']} | {row['anchor_useful']} |"
        )
    add_calibration_anchor_tables(lines, payload)
    add_evidence_table(lines, payload["significance_evidence"])
    lines.extend(["", "## Per-Subject Sanity Flags", ""])
    for key, flags in sorted(payload["sanity_flags_by_subject_variant"].items()):
        lines.append(f"- {key}: {', '.join(flags) if flags else 'none'}")
    lines.extend(
        [
            "",
            "## Explicit Verdicts",
            "",
            f"- project_verdict: {verdict}",
            f"- edl_verdict: {payload['edl_verdict']}",
            "",
            "## Publication Framing",
            "",
            framing,
            "",
            "## Leakage Audit",
            "",
            "- A0xE labels are used only for final trial-level metrics.",
            "- Crop windows are fixed from A0xT dimensions and never selected with A0xE labels.",
            "- EA is disabled.",
            "- Neural standardization is fitted on A0xT train only.",
            "- Teacher decoders are fitted on A0xT train only.",
            "- Teacher probabilities are repeated over A0xT training crops only.",
            "- Student temperature scaling is fitted on A0xT validation trial-level logits only.",
            "- Calibration-anchor alpha is fitted on A0xT validation Brier only.",
            "- Disagreement-gate thresholds are fitted on A0xT validation reliability only.",
            "- A0xE teacher probabilities are predictions only from teachers fitted on A0xT train.",
            "- No graph or EDL claim is promoted by this cropped no-graph run.",
        ]
    )
    text = "\n".join(lines)
    (results_dir / "cropped_anchor_readout.md").write_text(text, encoding="utf-8")
    return text


def print_table(
    summary: list[dict[str, Any]],
    cropped_rows: list[dict[str, Any]],
    anchor_rows: list[dict[str, Any]],
    multiscale_rows: list[dict[str, Any]],
    evidence_rows: list[dict[str, Any]],
    verdict: str,
    edl_verdict: str,
) -> None:
    print("\nAggregate cropped/multiscale-anchor table")
    print("variant                                      accuracy        ECE             Brier           sel_acc_60")
    for row in sorted(summary, key=lambda r: r["variant"]):
        print(f"{row['variant']:<44} {fmt_mean_std(row, 'accuracy'):<15} {fmt_mean_std(row, 'ECE'):<15} {fmt_mean_std(row, 'Brier'):<15} {fmt_mean_std(row, 'sel_acc_60'):<15}")
    print("\nCropped-usefulness paired deltas")
    if cropped_rows:
        for row in cropped_rows:
            print(f"{row['cropped_variant']} vs {row['previous_noncropped']}: acc {row['accuracy_delta_mean']:+.4f}, Brier {row['brier_delta_mean']:+.4f}, improved {row['subjects_accuracy_improved']}/{row['n_subjects']}, useful={row['cropping_useful']}")
    else:
        print("No previous matched non-cropped anchor rows available.")
    print("\nMultiscale-usefulness paired deltas")
    if multiscale_rows:
        for row in multiscale_rows:
            print(f"{row['multiscale_variant']} vs {row['previous_cropped_control']}: acc {row['accuracy_delta_mean']:+.4f}, Brier {row['brier_delta_mean']:+.4f}, improved {row['subjects_accuracy_improved']}/{row['n_subjects']}, useful={row['multiscale_useful']}")
    else:
        print("No previous matched cropped controls available.")
    print("\nAnchor-usefulness paired deltas")
    if anchor_rows:
        for row in anchor_rows:
            print(f"{row['anchor_variant']} vs {row['matched_raw']}: acc {row['accuracy_delta_mean']:+.4f}, Brier {row['brier_delta_mean']:+.4f}, improved {row['subjects_accuracy_improved']}/{row['n_subjects']}, useful={row['anchor_useful']}")
    else:
        print("No matched raw cropped controls available.")
    print("\nSignificance/evidence table")
    if evidence_rows:
        for row in evidence_rows[:20]:
            p_text = "pilot only" if row["pilot_only"] else f"p={row['p_value']:.4f}" if row["p_value"] is not None else "no p-value"
            print(f"{row['comparison']} {row['variant']} {row['metric']}: mean {row['mean_delta']:+.4f}, median {row['median_delta']:+.4f}, improved {row['subjects_improved']}/{row['n']} ({p_text})")
        if len(evidence_rows) > 20:
            print(f"... {len(evidence_rows) - 20} more rows in readout/summary JSON")
    else:
        print("No paired controls available.")
    print(f"\nVerdict: {verdict}")
    print(f"EDL justified yet: {edl_verdict}")


def main() -> int:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    subject_list = args.subjects or args.subject or list(range(1, N_SUBJECTS + 1))
    subjects = sorted(dict.fromkeys(subject_list))
    print(
        f"Running cropped anchor competence on device={device}; subjects={subjects}; variants={args.variants}; "
        f"teacher={args.teacher}; lambda={args.lambda_distill}; T={args.distill_temperature}; "
        f"crop_sizes={args.crop_sizes}; crop_strides={args.crop_strides}; balanced_sampler={args.balanced_sampler}; "
        f"cosine_lr={args.cosine_lr}; augment={args.augment}; calibration_anchor={args.calibration_anchor}; "
        f"alpha_grid_step={args.alpha_grid_step}; gate_coverages={args.gate_coverages}; disagreement={args.disagreement_score}",
        flush=True,
    )

    baseline_summary, baseline_subjects = load_minimal_baselines(args.baseline_dir)
    graph_rescue_summary, graph_rescue_subjects = load_variant_metrics(args.graph_rescue_dir, "graph_rescue")
    previous_anchor_summary, previous_anchor_subjects = load_variant_metrics(args.classical_anchor_dir, "classical_anchor")
    previous_cropped_summary, previous_cropped_subjects = load_variant_metrics(args.results_dir, "cropped_anchor")

    subject_rows: list[dict[str, Any]] = []
    by_subject_variant: dict[tuple[int, str], dict[str, Any]] = {}
    subject_payloads: dict[int, dict[str, Any]] = {}
    for subject in subjects:
        data = prepare_subject(subject, args.data_dir, args.seed)
        teacher = fit_teacher(data, args.teacher)
        models = {}
        print(f"S{subject:02d} teacher={args.teacher} train={data['X_train_raw'].shape[0]} val={data['X_val_raw'].shape[0]}", flush=True)
        for variant in args.variants:
            print(f"S{subject:02d} {variant}", flush=True)
            row = train_model(data, args, variant, teacher, device)
            by_subject_variant[(subject, variant)] = row
            models[variant] = row
            subject_rows.append(row)
        subject_payloads[subject] = {
            "protocol": "BCI_IV_2a_A0xT_to_A0xE_cropped_anchor_competence",
            "subject": subject,
            "classes": data["classes"],
            "paths": data["paths"],
            "split": data["split"],
            "teacher_audit": teacher["audit"],
            "models": models,
        }

    for row in subject_rows:
        row["sanity_flags"] = compute_sanity_flags(row, baseline_subjects, by_subject_variant, previous_anchor_subjects, previous_cropped_subjects)

    for subject, payload in subject_payloads.items():
        payload["sanity_flags_by_variant"] = {variant: model.get("sanity_flags", []) for variant, model in payload["models"].items()}
        write_json(args.results_dir / f"cropped_anchor_subject{subject}.json", payload)

    summary = aggregate(subject_rows)
    calibration_rows = calibration_anchor_summary(subject_rows)
    anchor_rows = anchor_usefulness(subject_rows, by_subject_variant)
    cropped_rows = cropped_usefulness(subject_rows, previous_anchor_subjects)
    multiscale_rows = multiscale_usefulness(subject_rows, by_subject_variant, previous_cropped_subjects)
    evidence_rows = significance_evidence(subject_rows, baseline_subjects, graph_rescue_subjects, by_subject_variant, previous_cropped_subjects)
    verdict, edl_verdict = decide_verdict(summary, baseline_summary, graph_rescue_summary)
    payload = {
        "protocol": "BCI_IV_2a_A0xT_to_A0xE_cropped_anchor_competence",
        "n_subjects": len(subjects),
        "variants": args.variants,
        "teacher": args.teacher,
        "lambda_distill": float(args.lambda_distill),
        "distill_temperature": float(args.distill_temperature),
        "crop_sizes": [int(s) for s in args.crop_sizes],
        "crop_strides": [int(s) for s in args.crop_strides],
        "balanced_sampler": bool(args.balanced_sampler),
        "cosine_lr": bool(args.cosine_lr),
        "augment": bool(args.augment),
        "noise_std": float(args.noise_std),
        "time_mask_frac": float(args.time_mask_frac),
        "calibration_anchor_enabled": bool(args.calibration_anchor),
        "alpha_grid_step": float(args.alpha_grid_step),
        "gate_coverages": [float(coverage) for coverage in args.gate_coverages],
        "disagreement_score": args.disagreement_score,
        "metrics": summary,
        "calibration_anchor_summary": calibration_rows,
        "paired_deltas_vs_eegnet_ts": paired_deltas(subject_rows, baseline_subjects, "eegnet_ts"),
        "paired_deltas_vs_lda_svm_vote": paired_deltas(subject_rows, baseline_subjects, "lda_svm_vote"),
        "paired_deltas_vs_graph_ce_raw": paired_against_variant(subject_rows, graph_rescue_subjects, "graph_ce_raw"),
        "paired_deltas_vs_graph_ce_no_graph_ablation": paired_against_variant(subject_rows, graph_rescue_subjects, "graph_ce_no_graph_ablation"),
        "cropped_usefulness": cropped_rows,
        "multiscale_usefulness": multiscale_rows,
        "anchor_usefulness": anchor_rows,
        "significance_evidence": evidence_rows,
        "sanity_flags_by_subject_variant": {f"S{row['subject']:02d}:{row['variant']}": row.get("sanity_flags", []) for row in subject_rows},
        "baseline_summary_available": bool(baseline_summary),
        "baseline_subject_metrics_available": bool(baseline_subjects),
        "graph_rescue_summary_available": bool(graph_rescue_summary),
        "graph_rescue_subject_metrics_available": bool(graph_rescue_subjects),
        "previous_anchor_summary_available": bool(previous_anchor_summary),
        "previous_anchor_subject_metrics_available": bool(previous_anchor_subjects),
        "previous_cropped_summary_available": bool(previous_cropped_summary),
        "previous_cropped_subject_metrics_available": bool(previous_cropped_subjects),
        "verdict": verdict,
        "edl_verdict": edl_verdict,
        "success_criteria": {
            "anchored_competent": "beats EEGNet+TS on mean accuracy and Brier, beats graph_rescue no-graph on accuracy or Brier, narrows LDA+SVM vote gap, and avoids systemic sanity failures",
            "cropping_useful": "cropped model beats matched non-cropped anchor on mean accuracy or Brier, or improves enough subjects, without meaningful calibration loss",
            "multiscale_useful": "multiscale temporal model beats matched previous cropped control on mean accuracy or Brier, or improves enough subjects, without meaningful calibration loss",
            "anchor_useful": "anchored cropped model beats matched raw cropped control on mean accuracy or Brier, or improves enough subjects, without meaningful calibration loss",
            "edl": "justified_later only if anchored multiscale/cropped CE is competent and stable on the full multi-subject run",
            "calibration_anchor": "useful if mixture improves mean Brier or ECE over student-only without >0.02 accuracy loss, or disagreement gating improves selective accuracy at matched target coverage",
        },
        "calibration_anchor_leakage_audit": {
            "alpha_fit_split": "A0xT_validation",
            "alpha_eval_labels_used": False,
            "gate_threshold_fit_split": "A0xT_validation",
            "gate_eval_labels_used": False,
            "teacher_fit_split": "A0xT_train",
            "eval_teacher_used_as_predictions_only": True,
        },
        "sanity_rules": {
            "best_epoch": "flag when == 1",
            "train_loss": "flag when final train loss is not below first train loss",
            "validation_accuracy": f"flag when <= chance + 0.05 ({CHANCE_ACC + 0.05:.2f})",
            "validation_brier": f"flag when >= chance Brier ({CHANCE_BRIER:.2f})",
            "temperature": "flag when validation temperature hits upper clamp",
            "eval_accuracy": "flag when below per-subject EEGNet+TS from minimal_edl_subject_metrics.csv",
            "anchor": "flag anchored variants that underperform matched raw cropped control on both accuracy and Brier",
            "cropping": "flag cropped anchors that fail matched previous non-cropped anchors on both accuracy and Brier, when available",
            "multiscale": "flag multiscale variants that underperform available previous cropped controls on both accuracy and Brier",
        },
    }
    write_csvs(args.results_dir, subject_rows, summary)
    write_json(args.results_dir / "cropped_anchor_summary.json", payload)
    write_readout(args.results_dir, payload)
    print_table(summary, cropped_rows, anchor_rows, multiscale_rows, evidence_rows, verdict, edl_verdict)
    if calibration_rows:
        print("\nCalibration Anchor Mixture")
        for row in calibration_rows:
            print(
                f"{row['variant']}: alpha={row['alpha_mean']:.3f} ({row['alpha_orientation']}), "
                f"Brier delta={fmt_optional(row['Brier_delta_mean'], signed=True)}, "
                f"ECE delta={fmt_optional(row['ECE_delta_mean'], signed=True)}, useful={row['useful']}"
            )
        print("\nTeacher-Student Disagreement Gate")
        for row in calibration_rows:
            parts = []
            for coverage in args.gate_coverages:
                suffix = coverage_suffix(float(coverage))
                parts.append(
                    f"{suffix}: gate sel={fmt_optional(row.get(f'gated_sel_acc_{suffix}_mean'))}, "
                    f"mix sel={fmt_optional(row.get(f'mixture_sel_acc_{suffix}_mean'))}, "
                    f"cov={fmt_optional(row.get(f'gated_coverage_{suffix}_mean'))}"
                )
            print(f"{row['variant']}: " + "; ".join(parts))
        print("\nCalibration anchoring is calibration novelty, not architecture novelty.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
