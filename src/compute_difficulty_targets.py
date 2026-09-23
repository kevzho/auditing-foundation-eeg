#!/usr/bin/env python3
"""Compute fixed auxiliary difficulty targets from EEGNet anchor checkpoints."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

N_SUBJECTS = 9
N_BINS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-trial difficulty targets from EEGNet anchor confidence bins."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing target pickle files.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default=None, help="Torch device. Defaults to cuda when available, else cpu.")
    return parser.parse_args()


def require_torch():
    try:
        import torch
        from torch import nn
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("This script requires PyTorch to run EEGNet inference.") from exc
    return torch, nn


def import_eegnet_builder():
    script_dir = Path(__file__).resolve().parent
    parent_dir = script_dir.parent
    for path in (script_dir, parent_dir):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    from eegnet import _build_eegnet

    def build_eegnet(torch, nn, n_channels: int, n_times: int, n_classes: int):
        return _build_eegnet(torch, nn, n_channels, n_times, n_classes, dropout=0.5)

    return build_eegnet


def unwrap_state_dict(checkpoint: Any) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("state_dict", "model_state_dict", "model", "net"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    if checkpoint and all(hasattr(v, "shape") for v in checkpoint.values()):
        return checkpoint
    raise ValueError("Could not find a model state_dict in checkpoint.")


def strip_prefix(state: dict[str, Any], prefix: str) -> dict[str, Any]:
    if not all(key.startswith(prefix) for key in state):
        return state
    return {key[len(prefix) :]: value for key, value in state.items()}


def load_state_dict_flexibly(model, state: dict[str, Any]) -> None:
    candidates = [
        state,
        strip_prefix(state, "module."),
        strip_prefix(state, "model."),
        strip_prefix(state, "backbone."),
    ]
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            missing, unexpected = model.load_state_dict(candidate, strict=False)
        except RuntimeError as exc:
            last_error = exc
            continue
        loaded = len(candidate) - len(unexpected)
        if loaded > 0:
            if missing or unexpected:
                print(
                    f"  checkpoint load used strict=False "
                    f"({len(missing)} missing, {len(unexpected)} unexpected keys)"
                )
            return
    if last_error is not None:
        raise last_error
    raise RuntimeError("No checkpoint tensors matched the EEGNet model.")


def checkpoint_temperature(checkpoint: Any) -> float:
    if not isinstance(checkpoint, dict):
        return 1.0
    for key in ("temperature", "temp", "calibration_temperature"):
        if key in checkpoint:
            value = checkpoint[key]
            if hasattr(value, "detach"):
                value = value.detach().cpu().item()
            elif isinstance(value, np.ndarray):
                value = value.item()
            return float(value)
    scaler = checkpoint.get("scaler") or checkpoint.get("temperature_scaler")
    if scaler is not None and hasattr(scaler, "temperature"):
        return float(scaler.temperature)
    return 1.0


def checkpoint_classes(checkpoint: Any) -> np.ndarray | None:
    if not isinstance(checkpoint, dict):
        return None
    for key in ("classes", "classes_", "label_classes", "label_encoder_classes"):
        if key in checkpoint:
            return np.asarray(checkpoint[key])
    return None


def load_checkpoint(path: Path):
    torch, _ = require_torch()
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def full_model_from_checkpoint(checkpoint: Any):
    if hasattr(checkpoint, "forward") and hasattr(checkpoint, "eval"):
        return checkpoint
    if isinstance(checkpoint, dict):
        for key in ("model", "net"):
            value = checkpoint.get(key)
            if hasattr(value, "forward") and hasattr(value, "eval"):
                return value
    return None


def infer_n_classes(state: dict[str, Any], y: np.ndarray) -> int:
    for key, value in state.items():
        if key.endswith("classifier.weight") and hasattr(value, "shape") and len(value.shape) == 2:
            return int(value.shape[0])
    return int(np.unique(y).size)


def encode_labels(y: np.ndarray, classes: np.ndarray | None, n_classes: int) -> np.ndarray:
    if classes is None:
        classes = np.asarray(sorted(np.unique(y).tolist()))
    mapping = {label.item() if hasattr(label, "item") else label: idx for idx, label in enumerate(classes)}
    try:
        encoded = np.asarray([mapping[label.item() if hasattr(label, "item") else label] for label in y], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"Validation label {exc.args[0]!r} is missing from checkpoint classes.") from exc
    if encoded.min(initial=0) < 0 or encoded.max(initial=0) >= n_classes:
        raise ValueError("Encoded validation labels are outside the model class range.")
    return encoded


def predict_probabilities(model, X: np.ndarray, *, temperature: float, batch_size: int, device: str):
    torch, _ = require_torch()
    model.eval()
    probas = []
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            batch = torch.tensor(X[start : start + batch_size, None, :, :], dtype=torch.float32, device=device)
            logits = model(batch) / temperature
            probas.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probas, axis=0)


def equal_frequency_bins(confidence: np.ndarray, n_bins: int) -> list[np.ndarray]:
    order = np.argsort(confidence, kind="mergesort")
    return [idx for idx in np.array_split(order, min(n_bins, confidence.size)) if idx.size > 0]


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def compute_subject(subject: int, args: argparse.Namespace) -> None:
    torch, nn = require_torch()
    build_eegnet = import_eegnet_builder()

    ckpt_path = args.checkpoint_dir / f"eegnet_ts_subject{subject}.pt"
    val_path = args.data_dir / f"bci4_2a_subject{subject}_val.npz"
    out_path = args.data_dir / f"difficulty_targets_subject{subject}.pkl"

    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"{out_path} already exists. Pass --overwrite to replace it.")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Missing validation split: {val_path}")

    data = np.load(val_path, allow_pickle=False)
    required = {"X", "y", "trial_id"}
    missing = required.difference(data.files)
    if missing:
        raise KeyError(f"{val_path} is missing keys: {sorted(missing)}")
    X = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"])
    trial_id = np.asarray(data["trial_id"])
    if X.ndim != 3:
        raise ValueError(f"Expected X with shape [N, C, T], got {X.shape}.")
    if not (X.shape[0] == y.shape[0] == trial_id.shape[0]):
        raise ValueError("X, y, and trial_id must have matching first dimensions.")

    checkpoint = load_checkpoint(ckpt_path)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    full_model = full_model_from_checkpoint(checkpoint)
    if full_model is not None:
        model = full_model.to(device)
        classes = checkpoint_classes(checkpoint)
        n_classes = len(classes) if classes is not None else int(np.unique(y).size)
    else:
        state = unwrap_state_dict(checkpoint)
        n_classes = infer_n_classes(state, y)
        model = build_eegnet(torch, nn, int(X.shape[1]), int(X.shape[2]), n_classes).to(device)
        load_state_dict_flexibly(model, state)

    temperature = checkpoint_temperature(checkpoint)
    if temperature <= 0.0:
        raise ValueError(f"Checkpoint temperature must be positive, got {temperature}.")
    y_encoded = encode_labels(y, checkpoint_classes(checkpoint), n_classes)
    proba = predict_probabilities(
        model,
        X,
        temperature=temperature,
        batch_size=args.batch_size,
        device=device,
    )

    confidence = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    errors = (pred != y_encoded).astype(np.float64)
    targets = np.empty(X.shape[0], dtype=np.float64)

    bins = equal_frequency_bins(confidence, N_BINS)
    for idx in bins:
        err_count = float(errors[idx].sum())
        total = int(idx.size)
        targets[idx] = (err_count + 1.0) / (total + 2.0)

    target_by_trial = {
        trial.item() if hasattr(trial, "item") else trial: float(target)
        for trial, target in zip(trial_id, targets, strict=True)
    }
    with out_path.open("wb") as f:
        pickle.dump(target_by_trial, f, protocol=pickle.HIGHEST_PROTOCOL)

    corr = pearson_corr(confidence, targets)
    corr_text = "nan" if np.isnan(corr) else f"{corr:.4f}"
    print(
        f"Subject {subject}: bins={len(bins)}/{N_BINS}, "
        f"tilde_s_mean={targets.mean():.4f}, tilde_s_std={targets.std(ddof=0):.4f}, "
        f"corr(c, tilde_s)={corr_text}, saved={out_path}"
    )


def main() -> int:
    args = parse_args()
    if not args.overwrite:
        existing = [
            args.data_dir / f"difficulty_targets_subject{subject}.pkl"
            for subject in range(1, N_SUBJECTS + 1)
            if (args.data_dir / f"difficulty_targets_subject{subject}.pkl").exists()
        ]
        if existing:
            paths = "\n  ".join(str(path) for path in existing)
            raise FileExistsError(
                f"Refusing to overwrite existing target files:\n  {paths}\nPass --overwrite to replace them."
            )
    for subject in range(1, N_SUBJECTS + 1):
        compute_subject(subject, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
