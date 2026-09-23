#!/usr/bin/env python3
"""Classical-anchored neural competence for BCI IV-2a A0xT -> A0xE.

This runner keeps the paper claim deliberately staged: classical decoders are
train-split-only competence anchors for closed-set CE neural models.  EDL and
graph claims remain downstream sanity-gated decisions.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from decode import build_decoder
from evaluate import compute_brier, compute_ece, compute_selective_acc
from experiments.run_primary import DEFAULT_BATCH_SIZE, DEFAULT_SEED, encode_labels, load_npz_session
from models.competence_graph_eegnet import CompetenceConfig, build_competence_model


N_SUBJECTS = 9
N_CLASSES = 4
CHANCE_ACC = 1.0 / N_CLASSES
CHANCE_BRIER = (N_CLASSES - 1.0) / N_CLASSES

ANCHOR_VARIANT_MAP = {
    "anchor_raw_eegnet_ce": "raw_eegnet_ce",
    "anchor_filterbank_no_graph": "raw_filterbank_no_graph",
    "anchor_filterbank_graph_anatomical": "raw_filterbank_graph_anatomical",
    "anchor_no_graph_compact": "raw_no_graph_compact",
    "anchor_graph_anatomical_learned_residual": "raw_graph_anatomical_learned_residual",
}
RAW_VARIANTS = (
    "raw_eegnet_ce",
    "raw_filterbank_no_graph",
    "raw_filterbank_graph_anatomical",
    "raw_no_graph_compact",
    "raw_graph_anatomical_learned_residual",
)
VARIANTS = tuple(ANCHOR_VARIANT_MAP) + RAW_VARIANTS
ANCHOR_TO_RAW = {anchor: raw for anchor, raw in ANCHOR_VARIANT_MAP.items()}
ANCHOR_GRAPH_MATCHES = {
    "anchor_filterbank_graph_anatomical": "anchor_filterbank_no_graph",
    "anchor_graph_anatomical_learned_residual": "anchor_no_graph_compact",
}
TEACHERS = ("lda", "svm", "lda_svm_vote")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=int, action="append", choices=range(1, N_SUBJECTS + 1))
    parser.add_argument("--subjects", type=int, nargs="+", choices=range(1, N_SUBJECTS + 1), help="Alternative multi-subject form.")
    parser.add_argument("--variants", nargs="+", default=list(ANCHOR_VARIANT_MAP), choices=VARIANTS)
    parser.add_argument("--teacher", default="lda_svm_vote", choices=TEACHERS)
    parser.add_argument("--lambda-distill", type=float, default=0.5)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results") / "classical_anchor")
    parser.add_argument("--baseline-dir", type=Path, default=Path("results") / "minimal_edl")
    parser.add_argument("--graph-rescue-dir", type=Path, default=Path("results") / "graph_rescue")
    parser.add_argument("--graph-competence-dir", type=Path, default=Path("results") / "graph_competence")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--graph-reg", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default=None)
    parser.add_argument("--temperature-max", type=float, default=20.0)
    return parser.parse_args()


def make_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): make_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(make_jsonable(payload), f, indent=2, sort_keys=True)
        f.write("\n")


def prepare_subject(subject: int, data_dir: Path, seed: int) -> dict[str, Any]:
    train_path = data_dir / f"bci4_2a_subject{subject}_train.npz"
    eval_path = data_dir / f"bci4_2a_subject{subject}_eval.npz"
    train = load_npz_session(train_path)
    eval_ = load_npz_session(eval_path)
    y_train_all, y_eval, classes = encode_labels(train["y"], eval_["y"])
    indices = np.arange(y_train_all.size)
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=seed + subject,
        stratify=y_train_all,
    )
    return {
        "subject": subject,
        "classes": classes,
        "X_train_raw": train["X"][train_idx],
        "y_train": y_train_all[train_idx],
        "X_val_raw": train["X"][val_idx],
        "y_val": y_train_all[val_idx],
        "X_eval_raw": eval_["X"],
        "y_eval": y_eval,
        "paths": {"train": str(train_path), "eval": str(eval_path)},
        "split": {
            "fit_split": "A0xT_train",
            "validation_split": "A0xT_validation",
            "eval_labels_used_for": "final_metrics_only",
            "stratified": True,
        },
    }


def standardize_fit(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((np.asarray(X, dtype=np.float32) - mean) / std).astype(np.float32, copy=False)


def preprocessing_view(data: dict[str, Any]) -> dict[str, Any]:
    mean, std = standardize_fit(data["X_train_raw"])
    return {
        "X_train": apply_standardize(data["X_train_raw"], mean, std),
        "X_val": apply_standardize(data["X_val_raw"], mean, std),
        "X_eval": apply_standardize(data["X_eval_raw"], mean, std),
        "preprocessing": {
            "euclidean_alignment": {"used": False, "fit_split": None, "eval_used_for_fit": False},
            "standardization": {"fit_split": "A0xT_train", "per_channel": True, "eval_used_for_fit": False},
        },
    }


def align_proba(p: np.ndarray, classes: np.ndarray, n_rows: int) -> np.ndarray:
    aligned = np.zeros((n_rows, N_CLASSES), dtype=np.float64)
    for j, cls in enumerate(classes):
        aligned[:, int(cls)] = p[:, j]
    denom = aligned.sum(axis=1, keepdims=True)
    denom = np.where(denom <= 0.0, 1.0, denom)
    return aligned / denom


def fit_teacher(data: dict[str, Any], teacher_name: str) -> dict[str, Any]:
    """Fit LDA/SVM only on A0xT train and return train/validation/eval predictions."""

    needed = ("lda", "svm") if teacher_name == "lda_svm_vote" else (teacher_name,)
    train_probas: dict[str, np.ndarray] = {}
    val_probas: dict[str, np.ndarray] = {}
    eval_probas: dict[str, np.ndarray] = {}
    audits: dict[str, Any] = {}
    for name in needed:
        decoder = build_decoder(name.upper())
        decoder.fit(data["X_train_raw"], data["y_train"])
        fitted_classes = getattr(decoder, "classes_", np.arange(N_CLASSES))
        train_probas[name] = align_proba(decoder.predict_proba(data["X_train_raw"]), fitted_classes, data["X_train_raw"].shape[0])
        val_probas[name] = align_proba(decoder.predict_proba(data["X_val_raw"]), fitted_classes, data["X_val_raw"].shape[0])
        eval_probas[name] = align_proba(decoder.predict_proba(data["X_eval_raw"]), fitted_classes, data["X_eval_raw"].shape[0])
        audits[name] = {
            "fit_split": "A0xT_train",
            "eval_used_for_fit": False,
            "eval_predictions_used_as_targets": False,
            "n_train": int(data["X_train_raw"].shape[0]),
            "n_val_predicted": int(data["X_val_raw"].shape[0]),
            "n_eval_predicted": int(data["X_eval_raw"].shape[0]),
            "classes": [int(c) for c in fitted_classes],
            "decoder": name,
            "calibration": "CalibratedClassifierCV inside train split",
        }
    if teacher_name == "lda_svm_vote":
        p_train = 0.5 * (train_probas["lda"] + train_probas["svm"])
        p_val = 0.5 * (val_probas["lda"] + val_probas["svm"])
        p_eval = 0.5 * (eval_probas["lda"] + eval_probas["svm"])
    else:
        p_train = train_probas[teacher_name]
        p_val = val_probas[teacher_name]
        p_eval = eval_probas[teacher_name]
    return {
        "name": teacher_name,
        "train_proba": p_train.astype(np.float32),
        "val_proba": p_val.astype(np.float32),
        "eval_proba": p_eval.astype(np.float32),
        "audit": {
            "teacher": teacher_name,
            "fit_split": "A0xT_train",
            "validation_predictions_for_model_selection": True,
            "eval_predictions_used_as_predictions_only": True,
            "eval_predictions_used_as_targets": False,
            "eval_labels_used": False,
            "members": audits,
        },
    }


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    teacher_proba: np.ndarray | None,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    tensors: list[torch.Tensor] = [torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long)]
    if teacher_proba is not None:
        tensors.append(torch.as_tensor(teacher_proba, dtype=torch.float32))
    ds = TensorDataset(*tensors)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def collect_logits(model: torch.nn.Module, X: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    model.eval()
    chunks = []
    for start in range(0, X.shape[0], batch_size):
        xb = torch.as_tensor(X[start : start + batch_size], dtype=torch.float32, device=device)
        chunks.append(model(xb).detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def proba_from_logits(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    z -= z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def metrics_for(y_true: np.ndarray, logits: np.ndarray, temperature: float = 1.0) -> dict[str, float]:
    proba = proba_from_logits(logits, temperature)
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


def fit_temperature_from_logits(logits: np.ndarray, y: np.ndarray, upper: float) -> tuple[float, bool]:
    logits_t = torch.as_tensor(logits, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.long)
    log_t = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=300, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = torch.exp(log_t).clamp(0.05, float(upper))
        loss = F.cross_entropy(logits_t / temperature, y_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(torch.exp(log_t).detach().clamp(0.05, float(upper)).item())
    return temperature, bool(temperature >= float(upper) - 1e-3)


def distill_target(proba: torch.Tensor, temperature: float) -> torch.Tensor:
    logp = torch.log(proba.clamp_min(1e-8))
    return torch.softmax(logp / float(temperature), dim=1)


def distill_loss(logits: torch.Tensor, teacher_proba: torch.Tensor, temperature: float) -> torch.Tensor:
    t = float(temperature)
    teacher_t = distill_target(teacher_proba, t)
    student_logp = F.log_softmax(logits / t, dim=1)
    return F.kl_div(student_logp, teacher_t, reduction="batchmean") * (t * t)


def build_model(X: np.ndarray, args: argparse.Namespace, variant: str) -> torch.nn.Module:
    raw_variant = ANCHOR_VARIANT_MAP.get(variant, variant)
    cfg = CompetenceConfig(
        n_channels=int(X.shape[1]),
        n_times=int(X.shape[2]),
        n_classes=N_CLASSES,
        variant=raw_variant,
        dropout=float(args.dropout),
    )
    return build_competence_model(cfg)


def validation_loss(
    logits: np.ndarray,
    y: np.ndarray,
    teacher_proba: np.ndarray | None,
    args: argparse.Namespace,
    anchored: bool,
) -> float:
    logits_t = torch.as_tensor(logits, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.long)
    loss = F.cross_entropy(logits_t, y_t)
    if anchored and teacher_proba is not None and args.lambda_distill > 0:
        teacher_t = torch.as_tensor(teacher_proba, dtype=torch.float32)
        loss = loss + float(args.lambda_distill) * distill_loss(logits_t, teacher_t, args.distill_temperature)
    return float(loss.item())


def train_model(
    data: dict[str, Any],
    args: argparse.Namespace,
    variant: str,
    teacher: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    view = preprocessing_view(data)
    variant_offset = sum((i + 1) * ord(ch) for i, ch in enumerate(variant)) % 1000
    subject_seed = int(args.seed + int(data["subject"]) * 101 + variant_offset)
    torch.manual_seed(subject_seed)
    np.random.seed(subject_seed)

    anchored = variant in ANCHOR_VARIANT_MAP
    train_teacher = teacher["train_proba"] if anchored else None
    val_teacher = teacher["val_proba"] if anchored else None
    model = build_model(view["X_train"], args, variant).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = make_loader(view["X_train"], data["y_train"], train_teacher, args.batch_size, shuffle=True)
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
            xb = batch[0].to(device)
            yb = batch[1].to(device)
            tb = batch[2].to(device) if anchored else None
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            ce = loss_fn(logits, yb)
            kd = distill_loss(logits, tb, args.distill_temperature) if tb is not None and args.lambda_distill > 0 else torch.zeros((), device=device)
            reg = model.regularization_loss() if hasattr(model, "regularization_loss") else torch.zeros((), device=device)
            loss = ce + float(args.lambda_distill) * kd + float(args.graph_reg) * reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))
            train_ce_losses.append(float(ce.detach().cpu().item()))
            train_distill_losses.append(float(kd.detach().cpu().item()))

        val_logits = collect_logits(model, view["X_val"], device, args.batch_size)
        val_loss = validation_loss(val_logits, data["y_val"], val_teacher, args, anchored)
        val_metrics = metrics_for(data["y_val"], val_logits)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "train_ce_loss": float(np.mean(train_ce_losses)),
            "train_distill_loss": float(np.mean(train_distill_losses)),
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_brier": val_metrics["Brier"],
        }
        history.append(row)
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

    val_logits = collect_logits(model, view["X_val"], device, args.batch_size)
    eval_logits = collect_logits(model, view["X_eval"], device, args.batch_size)
    temperature, temp_at_upper = fit_temperature_from_logits(val_logits, data["y_val"], args.temperature_max)
    val_metrics = metrics_for(data["y_val"], val_logits, temperature)
    eval_metrics = metrics_for(data["y_eval"], eval_logits, temperature)
    return {
        "subject": int(data["subject"]),
        "variant": variant,
        "raw_architecture_variant": ANCHOR_VARIANT_MAP.get(variant, variant),
        "anchored": bool(anchored),
        "teacher": teacher["name"] if anchored else None,
        "lambda_distill": float(args.lambda_distill) if anchored else 0.0,
        "distill_temperature": float(args.distill_temperature) if anchored else None,
        "metrics": eval_metrics,
        "validation_metrics": val_metrics,
        "model_info": {
            **view["preprocessing"],
            "temperature_scaling": {
                "enabled": True,
                "fit_split": "A0xT_validation",
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
                "teacher_targets": "A0xT_train_and_A0xT_validation_only" if anchored else None,
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


def load_minimal_baselines(path: Path) -> tuple[dict[str, dict[str, float]], dict[int, dict[str, dict[str, float]]]]:
    summary: dict[str, dict[str, float]] = {}
    subjects: dict[int, dict[str, dict[str, float]]] = {}
    summary_path = path / "minimal_edl_summary.json"
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        for row in payload.get("metrics", []):
            summary[row["model"]] = {
                "accuracy": row.get("accuracy", {}).get("mean"),
                "Brier": row.get("Brier", {}).get("mean"),
                "ECE": row.get("ECE", {}).get("mean"),
                "sel_acc_60": row.get("sel_acc_60", {}).get("mean"),
            }
    subject_path = path / "minimal_edl_subject_metrics.csv"
    if subject_path.exists():
        with subject_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                subject = int(row["subject"])
                model = row["model"]
                subjects.setdefault(subject, {})[model] = {
                    key: float(row[key]) for key in ("accuracy", "ECE", "Brier", "sel_acc_60", "coverage_60") if row.get(key) not in ("", None)
                }
    return summary, subjects


def load_variant_metrics(path: Path, stem: str) -> tuple[dict[str, dict[str, float]], dict[int, dict[str, dict[str, float]]]]:
    summary: dict[str, dict[str, float]] = {}
    subjects: dict[int, dict[str, dict[str, float]]] = {}
    subject_path = path / f"{stem}_subject_metrics.csv"
    if subject_path.exists():
        with subject_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                subject = int(row["subject"])
                variant = row["variant"]
                subjects.setdefault(subject, {})[variant] = {
                    key: float(row[key]) for key in ("accuracy", "ECE", "Brier", "sel_acc_60", "coverage_60") if row.get(key) not in ("", None)
                }
    summary_path = path / f"{stem}_summary.csv"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                variant = row["variant"]
                summary[variant] = {
                    "accuracy": float(row["accuracy_mean"]),
                    "Brier": float(row["Brier_mean"]),
                    "ECE": float(row["ECE_mean"]),
                    "sel_acc_60": float(row["sel_acc_60_mean"]),
                }
    return summary, subjects


def get_external_control(
    subject: int,
    variant: str,
    by_subject_variant: dict[tuple[int, str], dict[str, Any]],
    graph_competence_subjects: dict[int, dict[str, dict[str, float]]],
) -> dict[str, Any] | None:
    row = by_subject_variant.get((subject, variant))
    if row:
        return row["metrics"]
    return graph_competence_subjects.get(subject, {}).get(variant)


def compute_sanity_flags(
    row: dict[str, Any],
    baseline_subjects: dict[int, dict[str, dict[str, float]]],
    by_subject_variant: dict[tuple[int, str], dict[str, Any]],
    graph_competence_subjects: dict[int, dict[str, dict[str, float]]],
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
    if row["variant"] in ANCHOR_TO_RAW:
        raw = get_external_control(int(row["subject"]), ANCHOR_TO_RAW[row["variant"]], by_subject_variant, graph_competence_subjects)
        if raw and row["metrics"]["accuracy"] < raw["accuracy"] and row["metrics"]["Brier"] > raw["Brier"]:
            flags.append("anchor_worse_than_matched_raw_accuracy_and_brier")
    if row["variant"] in ANCHOR_GRAPH_MATCHES:
        baseline = by_subject_variant.get((int(row["subject"]), ANCHOR_GRAPH_MATCHES[row["variant"]]))
        if baseline and row["metrics"]["accuracy"] < baseline["metrics"]["accuracy"] and row["metrics"]["Brier"] > baseline["metrics"]["Brier"]:
            flags.append("anchored_graph_underperforms_anchored_no_graph_accuracy_and_brier")
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


def paired_deltas(subject_rows: list[dict[str, Any]], baseline_subjects: dict[int, dict[str, dict[str, float]]], baseline: str) -> dict[str, Any]:
    out = {}
    for variant in sorted({row["variant"] for row in subject_rows}):
        rows = [row for row in subject_rows if row["variant"] == variant and baseline in baseline_subjects.get(row["subject"], {})]
        if not rows:
            continue
        out[variant] = {}
        for metric in ("accuracy", "ECE", "Brier", "sel_acc_60"):
            values = np.array([row["metrics"][metric] - baseline_subjects[row["subject"]][baseline][metric] for row in rows], dtype=float)
            out[variant][metric] = {"mean": float(values.mean()), "std": float(values.std(ddof=0)), "n": int(values.size)}
    return out


def anchor_usefulness(
    subject_rows: list[dict[str, Any]],
    by_subject_variant: dict[tuple[int, str], dict[str, Any]],
    graph_competence_subjects: dict[int, dict[str, dict[str, float]]],
) -> list[dict[str, Any]]:
    rows = []
    for anchor, raw in ANCHOR_TO_RAW.items():
        anchors = [row for row in subject_rows if row["variant"] == anchor]
        subjects = []
        acc_delta = []
        brier_delta = []
        ece_delta = []
        for row in anchors:
            control = get_external_control(int(row["subject"]), raw, by_subject_variant, graph_competence_subjects)
            if not control:
                continue
            subjects.append(int(row["subject"]))
            acc_delta.append(row["metrics"]["accuracy"] - control["accuracy"])
            brier_delta.append(row["metrics"]["Brier"] - control["Brier"])
            ece_delta.append(row["metrics"]["ECE"] - control["ECE"])
        if not subjects:
            continue
        acc = np.array(acc_delta, dtype=float)
        brier = np.array(brier_delta, dtype=float)
        ece = np.array(ece_delta, dtype=float)
        rows.append(
            {
                "anchor_variant": anchor,
                "matched_raw": raw,
                "n_subjects": len(subjects),
                "accuracy_delta_mean": float(acc.mean()),
                "brier_delta_mean": float(brier.mean()),
                "ece_delta_mean": float(ece.mean()),
                "subjects_accuracy_improved": int(np.sum(acc > 0)),
                "subjects_brier_improved": int(np.sum(brier < 0)),
                "anchor_useful": bool((acc.mean() > 0 or brier.mean() < 0 or np.sum(acc > 0) >= 5) and brier.mean() <= 0.02),
                "control_source": "current_run_or_graph_competence",
            }
        )
    return rows


def graph_usefulness(subject_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    by_key = {(row["subject"], row["variant"]): row for row in subject_rows}
    for graph_variant, baseline in ANCHOR_GRAPH_MATCHES.items():
        subjects = sorted({s for s, v in by_key if v == graph_variant} & {s for s, v in by_key if v == baseline})
        if not subjects:
            continue
        acc_delta = np.array([by_key[(s, graph_variant)]["metrics"]["accuracy"] - by_key[(s, baseline)]["metrics"]["accuracy"] for s in subjects])
        brier_delta = np.array([by_key[(s, graph_variant)]["metrics"]["Brier"] - by_key[(s, baseline)]["metrics"]["Brier"] for s in subjects])
        rows.append(
            {
                "graph_variant": graph_variant,
                "matched_anchor_no_graph": baseline,
                "n_subjects": len(subjects),
                "accuracy_delta_mean": float(acc_delta.mean()),
                "brier_delta_mean": float(brier_delta.mean()),
                "subjects_accuracy_improved": int(np.sum(acc_delta > 0)),
                "subjects_brier_improved": int(np.sum(brier_delta < 0)),
                "graph_useful": bool(acc_delta.mean() > 0 and brier_delta.mean() <= 0.02 and (np.sum(acc_delta > 0) >= 5 or acc_delta.mean() > 0)),
            }
        )
    return rows


def beats_graph_rescue(row: dict[str, Any], graph_rescue_summary: dict[str, dict[str, float]]) -> bool:
    rescue = graph_rescue_summary.get("graph_ce_no_graph_ablation")
    if not rescue:
        return False
    return bool(row["accuracy"]["mean"] > rescue["accuracy"] or row["Brier"]["mean"] < rescue["Brier"])


def decide_verdict(
    summary: list[dict[str, Any]],
    graph_rows: list[dict[str, Any]],
    baseline_summary: dict[str, dict[str, float]],
    graph_rescue_summary: dict[str, dict[str, float]],
) -> tuple[str, str]:
    eegnet = baseline_summary.get("eegnet_ts")
    vote = baseline_summary.get("lda_svm_vote")
    if not eegnet:
        return "failed", "not_yet"
    anchored = [row for row in summary if row["anchored"]]
    if not anchored:
        return "failed", "not_yet"
    best = max(anchored, key=lambda r: (r["accuracy"]["mean"], -r["Brier"]["mean"]))
    vote_gap_narrowed = True
    if vote:
        vote_gap_narrowed = (
            abs(vote["accuracy"] - best["accuracy"]["mean"]) < abs(vote["accuracy"] - eegnet["accuracy"])
            or abs(best["Brier"]["mean"] - vote["Brier"]) < abs(eegnet["Brier"] - vote["Brier"])
        )
    anchored_competent = bool(
        best["accuracy"]["mean"] > eegnet["accuracy"]
        and best["Brier"]["mean"] < eegnet["Brier"]
        and beats_graph_rescue(best, graph_rescue_summary)
        and vote_gap_narrowed
        and best["sanity_flag_count"] < best["n_subjects"]
    )
    useful_graph_variants = {row["graph_variant"] for row in graph_rows if row["graph_useful"]}
    competent_graph = any(row["variant"] in useful_graph_variants for row in anchored if row["sanity_flag_count"] < row["n_subjects"])
    if anchored_competent and competent_graph:
        return "anchored_graph_candidate", "justified_later"
    if anchored_competent:
        if vote and best["accuracy"]["mean"] < vote["accuracy"] and best["Brier"]["mean"] > vote["Brier"]:
            return "classical_still_best", "not_yet"
        return "anchored_deep_no_graph", "not_yet"
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
    with (results_dir / "classical_anchor_subject_metrics.csv").open("w", encoding="utf-8", newline="") as f:
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
    for metric in (
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
    ):
        fields.extend([f"{metric}_mean", f"{metric}_std"])
    fields.extend(["temperature_upper_clamp_count", "sanity_flag_count"])
    with (results_dir / "classical_anchor_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sorted(summary, key=lambda r: r["variant"]):
            flat = {"variant": row["variant"], "anchored": row["anchored"], "teacher": row["teacher"] or "", "n_subjects": row["n_subjects"]}
            for metric in (
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
            ):
                flat[f"{metric}_mean"] = row[metric]["mean"]
                flat[f"{metric}_std"] = row[metric]["std"]
            flat["temperature_upper_clamp_count"] = row["temperature_upper_clamp_count"]
            flat["sanity_flag_count"] = row["sanity_flag_count"]
            writer.writerow(flat)


def fmt_mean_std(row: dict[str, Any], metric: str) -> str:
    return f"{row[metric]['mean']:.4f} ({row[metric]['std']:.4f})"


def add_delta_table(lines: list[str], title: str, deltas: dict[str, Any]) -> None:
    if not deltas:
        return
    lines.extend(["", f"## {title}", "", "| variant | accuracy | ECE | Brier | sel_acc_60 | n |", "|---|---:|---:|---:|---:|---:|"])
    for variant, vals in sorted(deltas.items()):
        lines.append(
            f"| {variant} | {vals['accuracy']['mean']:+.4f} | {vals['ECE']['mean']:+.4f} | "
            f"{vals['Brier']['mean']:+.4f} | {vals['sel_acc_60']['mean']:+.4f} | {vals['accuracy']['n']} |"
        )


def write_readout(results_dir: Path, payload: dict[str, Any]) -> str:
    verdict = payload["verdict"]
    if verdict == "anchored_graph_candidate":
        framing = (
            "Classical decoders provide leakage-safe calibration anchors that improve neural closed-set competence, "
            "enabling reliability-gated uncertainty claims without using eval information.\n\n"
            "Anatomically constrained graph mixing adds value only after classical anchoring establishes neural competence."
        )
    elif verdict == "anchored_deep_no_graph":
        framing = (
            "The protocol prevents premature graph claims; classical anchoring helps neural calibration/competence "
            "while graph novelty remains unproven."
        )
    elif verdict == "classical_still_best":
        framing = (
            "Classical decoders remain the competence floor in small-sample MI; the contribution becomes a diagnostic "
            "framework and teacher-anchored path toward neural uncertainty."
        )
    else:
        framing = (
            "Classical decoders remain the necessary competence anchor; this run does not yet support neural or graph "
            "claims beyond the diagnostic framework."
        )

    lines = [
        "# Classical Anchor Competence Readout",
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
    lines.extend(
        [
            "",
            "## Anchor Usefulness",
            "",
            "| anchor | matched raw | acc delta | Brier delta | ECE delta | acc improved | Brier improved | useful |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in payload["anchor_usefulness"]:
        lines.append(
            f"| {row['anchor_variant']} | {row['matched_raw']} | {row['accuracy_delta_mean']:+.4f} | "
            f"{row['brier_delta_mean']:+.4f} | {row['ece_delta_mean']:+.4f} | "
            f"{row['subjects_accuracy_improved']}/{row['n_subjects']} | "
            f"{row['subjects_brier_improved']}/{row['n_subjects']} | {row['anchor_useful']} |"
        )
    lines.extend(
        [
            "",
            "## Graph Usefulness",
            "",
            "| graph anchor | matched anchor no-graph | acc delta | Brier delta | acc improved | Brier improved | useful |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in payload["graph_usefulness"]:
        lines.append(
            f"| {row['graph_variant']} | {row['matched_anchor_no_graph']} | {row['accuracy_delta_mean']:+.4f} | "
            f"{row['brier_delta_mean']:+.4f} | {row['subjects_accuracy_improved']}/{row['n_subjects']} | "
            f"{row['subjects_brier_improved']}/{row['n_subjects']} | {row['graph_useful']} |"
        )
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
            "- A0xE labels are used only for final metrics.",
            "- EA is disabled.",
            "- Neural standardization is fitted on A0xT train only.",
            "- Teacher decoders are fitted on A0xT train only.",
            "- Teacher predictions on A0xE are never used as training targets.",
            "- Student temperature scaling is fitted on A0xT validation logits only.",
        ]
    )
    text = "\n".join(lines)
    (results_dir / "classical_anchor_readout.md").write_text(text, encoding="utf-8")
    return text


def print_table(summary: list[dict[str, Any]], anchor_rows: list[dict[str, Any]], graph_rows: list[dict[str, Any]], verdict: str, edl_verdict: str) -> None:
    print("\nAggregate classical-anchor table")
    print("variant                                      accuracy        ECE             Brier           sel_acc_60")
    for row in sorted(summary, key=lambda r: r["variant"]):
        print(
            f"{row['variant']:<44} {fmt_mean_std(row, 'accuracy'):<15} "
            f"{fmt_mean_std(row, 'ECE'):<15} {fmt_mean_std(row, 'Brier'):<15} {fmt_mean_std(row, 'sel_acc_60'):<15}"
        )
    if anchor_rows:
        print("\nAnchor-usefulness paired deltas")
        for row in anchor_rows:
            print(
                f"{row['anchor_variant']} vs {row['matched_raw']}: "
                f"acc {row['accuracy_delta_mean']:+.4f}, Brier {row['brier_delta_mean']:+.4f}, "
                f"improved {row['subjects_accuracy_improved']}/{row['n_subjects']}, useful={row['anchor_useful']}"
            )
    if graph_rows:
        print("\nGraph-usefulness paired deltas")
        for row in graph_rows:
            print(
                f"{row['graph_variant']} vs {row['matched_anchor_no_graph']}: "
                f"acc {row['accuracy_delta_mean']:+.4f}, Brier {row['brier_delta_mean']:+.4f}, "
                f"improved {row['subjects_accuracy_improved']}/{row['n_subjects']}, useful={row['graph_useful']}"
            )
    print(f"\nVerdict: {verdict}")
    print(f"EDL justified yet: {edl_verdict}")


def main() -> int:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    subject_list = args.subjects or args.subject or list(range(1, N_SUBJECTS + 1))
    subjects = sorted(dict.fromkeys(subject_list))
    print(
        f"Running classical anchor competence on device={device}; subjects={subjects}; "
        f"variants={args.variants}; teacher={args.teacher}; lambda={args.lambda_distill}; T={args.distill_temperature}",
        flush=True,
    )

    baseline_summary, baseline_subjects = load_minimal_baselines(args.baseline_dir)
    graph_rescue_summary, graph_rescue_subjects = load_variant_metrics(args.graph_rescue_dir, "graph_rescue")
    graph_competence_summary, graph_competence_subjects = load_variant_metrics(args.graph_competence_dir, "graph_competence")

    subject_rows: list[dict[str, Any]] = []
    by_subject_variant: dict[tuple[int, str], dict[str, Any]] = {}
    subject_payloads: dict[int, dict[str, Any]] = {}
    for subject in subjects:
        data = prepare_subject(subject, args.data_dir, args.seed)
        teacher = fit_teacher(data, args.teacher)
        models = {}
        print(
            f"S{subject:02d} teacher={args.teacher} train={data['X_train_raw'].shape[0]} val={data['X_val_raw'].shape[0]}",
            flush=True,
        )
        for variant in args.variants:
            print(f"S{subject:02d} {variant}", flush=True)
            row = train_model(data, args, variant, teacher, device)
            by_subject_variant[(subject, variant)] = row
            models[variant] = row
            subject_rows.append(row)
        subject_payloads[subject] = {
            "protocol": "BCI_IV_2a_A0xT_to_A0xE_classical_anchor_competence",
            "subject": subject,
            "classes": data["classes"],
            "paths": data["paths"],
            "split": data["split"],
            "teacher_audit": teacher["audit"],
            "models": models,
        }

    for row in subject_rows:
        row["sanity_flags"] = compute_sanity_flags(row, baseline_subjects, by_subject_variant, graph_competence_subjects)

    for subject, payload in subject_payloads.items():
        payload["sanity_flags_by_variant"] = {
            variant: model.get("sanity_flags", []) for variant, model in payload["models"].items()
        }
        write_json(args.results_dir / f"classical_anchor_subject{subject}.json", payload)

    summary = aggregate(subject_rows)
    anchor_rows = anchor_usefulness(subject_rows, by_subject_variant, graph_competence_subjects)
    graph_rows = graph_usefulness(subject_rows)
    verdict, edl_verdict = decide_verdict(summary, graph_rows, baseline_summary, graph_rescue_summary)
    payload = {
        "protocol": "BCI_IV_2a_A0xT_to_A0xE_classical_anchor_competence",
        "n_subjects": len(subjects),
        "variants": args.variants,
        "teacher": args.teacher,
        "lambda_distill": float(args.lambda_distill),
        "distill_temperature": float(args.distill_temperature),
        "metrics": summary,
        "paired_deltas_vs_eegnet_ts": paired_deltas(subject_rows, baseline_subjects, "eegnet_ts"),
        "paired_deltas_vs_lda_svm_vote": paired_deltas(subject_rows, baseline_subjects, "lda_svm_vote"),
        "paired_deltas_vs_graph_ce_raw": paired_deltas(subject_rows, graph_rescue_subjects, "graph_ce_raw"),
        "paired_deltas_vs_graph_ce_no_graph_ablation": paired_deltas(subject_rows, graph_rescue_subjects, "graph_ce_no_graph_ablation"),
        "anchor_usefulness": anchor_rows,
        "graph_usefulness": graph_rows,
        "sanity_flags_by_subject_variant": {f"S{row['subject']:02d}:{row['variant']}": row.get("sanity_flags", []) for row in subject_rows},
        "baseline_summary_available": bool(baseline_summary),
        "baseline_subject_metrics_available": bool(baseline_subjects),
        "graph_rescue_summary_available": bool(graph_rescue_summary),
        "graph_rescue_subject_metrics_available": bool(graph_rescue_subjects),
        "graph_competence_summary_available": bool(graph_competence_summary),
        "graph_competence_subject_metrics_available": bool(graph_competence_subjects),
        "verdict": verdict,
        "edl_verdict": edl_verdict,
        "success_criteria": {
            "anchored_competent": "beats EEGNet+TS on mean accuracy and Brier, beats graph_rescue no-graph on accuracy or Brier, narrows LDA+SVM vote gap, and avoids systemic sanity failures",
            "anchor_useful": "anchored version beats matched raw student on mean accuracy or Brier, or improves enough subjects, without meaningful calibration loss",
            "graph_useful": "anchored graph beats matched anchored no-graph on mean accuracy without meaningful Brier loss",
        },
        "sanity_rules": {
            "best_epoch": "flag when == 1",
            "train_loss": "flag when final train loss is not below first train loss",
            "validation_accuracy": f"flag when <= chance + 0.05 ({CHANCE_ACC + 0.05:.2f})",
            "validation_brier": f"flag when >= chance Brier ({CHANCE_BRIER:.2f})",
            "temperature": "flag when validation temperature hits upper clamp",
            "eval_accuracy": "flag when below per-subject EEGNet+TS from minimal_edl_subject_metrics.csv",
            "anchor": "flag anchored variants that underperform matched raw control on both accuracy and Brier",
            "graph": "flag anchored graph variants that underperform matched anchored no-graph on both accuracy and Brier",
        },
    }
    write_csvs(args.results_dir, subject_rows, summary)
    write_json(args.results_dir / "classical_anchor_summary.json", payload)
    write_readout(args.results_dir, payload)
    print_table(summary, anchor_rows, graph_rows, verdict, edl_verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
