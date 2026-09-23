#!/usr/bin/env python3
"""Run the primary CalibMI EDL-SGraph experiment on BCI IV 2a A0xT -> A0xE."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adaptation import apply_channel_matrix, euclidean_alignment_matrix
from evaluate import compute_brier, edl_abstention_unknown, edl_abstention_vacuity, evaluate_model
from models.edl_sgraph import EDLSGraph


N_SUBJECTS = 9
N_CLASSES = 4
SWEEP_SUBJECTS = (1, 5)
MAX_EPOCHS = 200
PATIENCE = 20
DEFAULT_BATCH_SIZE = 32
DEFAULT_SEED = 2026
HP_GRID = {
    "lambda_B": [0.1, 0.5, 1.0],
    "lambda_KL": [0.01, 0.1, 0.5],
    "lambda_diff": [0.05, 0.1, 0.3],
    "lambda_sel": [0.0, 0.05, 0.1],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=int, choices=range(1, N_SUBJECTS + 1), help="Run one subject only.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned files/sweep and exit.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default=None, help="Torch device. Defaults to cuda when available, else cpu.")
    parser.add_argument("--no-temperature", action="store_true", help="Disable validation temperature scaling.")
    return parser.parse_args()


def hyperparameter_grid() -> list[dict[str, float]]:
    keys = list(HP_GRID)
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*(HP_GRID[k] for k in keys))]


def load_npz_session(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing session file: {path}")
    data = np.load(path, allow_pickle=False)
    missing = {"X", "y"}.difference(data.files)
    if missing:
        raise KeyError(f"{path} is missing keys: {sorted(missing)}. Found: {list(data.files)}")
    X = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"])
    if X.ndim != 3:
        raise ValueError(f"{path}: expected X with shape [N, C, T], got {X.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"{path}: X and y have different first dimensions: {X.shape[0]} vs {y.shape[0]}")
    out = {"X": X, "y": y}
    if "trial_id" in data.files:
        out["trial_id"] = np.asarray(data["trial_id"])
    return out


def encode_labels(y_train: np.ndarray, y_eval: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None, list[Any]]:
    classes = sorted(np.unique(y_train).tolist())
    if len(classes) != N_CLASSES:
        raise ValueError(f"Expected {N_CLASSES} training classes, found {classes}")
    mapping = {label: idx for idx, label in enumerate(classes)}

    def encode(y: np.ndarray) -> np.ndarray:
        encoded = []
        for value in y.tolist():
            if value not in mapping:
                raise ValueError(f"Label {value!r} is not present in training classes {classes}")
            encoded.append(mapping[value])
        return np.asarray(encoded, dtype=np.int64)

    return encode(y_train), (encode(y_eval) if y_eval is not None else None), classes


def load_difficulty_targets(path: Path, n_trials: int, trial_ids: np.ndarray | None = None) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing difficulty target file: {path}")
    with path.open("rb") as f:
        raw = pickle.load(f)

    if isinstance(raw, dict):
        for key in ("s_target", "difficulty", "targets", "difficulty_targets"):
            if key in raw and not isinstance(raw[key], dict):
                values = np.asarray(raw[key], dtype=np.float32).reshape(-1)
                if values.size == n_trials:
                    return values
        if trial_ids is None:
            keys = list(raw.keys())
            if len(keys) == n_trials and all(isinstance(k, (int, np.integer)) for k in keys):
                return np.asarray([raw[k] for k in sorted(keys)], dtype=np.float32)
            raise ValueError(f"{path}: dict targets require trial_id in the training npz.")
        values = []
        for trial in trial_ids:
            key = trial.item() if hasattr(trial, "item") else trial
            if key not in raw:
                raise KeyError(f"{path}: missing difficulty target for trial_id={key!r}")
            values.append(raw[key])
        return np.asarray(values, dtype=np.float32)

    values = np.asarray(raw, dtype=np.float32).reshape(-1)
    if values.size != n_trials:
        raise ValueError(f"{path}: expected {n_trials} difficulty targets, got {values.size}")
    return values


def train_val_split(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if n < 5:
        raise ValueError(f"Need at least 5 training trials for an 80/20 split, got {n}")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(round(0.2 * n)))
    return idx[n_val:], idx[:n_val]


def as_loader(X: np.ndarray, y: np.ndarray, s: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.as_tensor(X, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.long),
        torch.as_tensor(s, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def trim_known_probs(p_hat: np.ndarray) -> np.ndarray:
    known = p_hat[:, :N_CLASSES]
    denom = np.maximum(known.sum(axis=1, keepdims=True), 1e-12)
    return known / denom


@torch.no_grad()
def collect_outputs(model: EDLSGraph, X: np.ndarray, device: str, batch_size: int) -> dict[str, np.ndarray]:
    model.eval()
    outputs: dict[str, list[np.ndarray]] = {"p_hat": [], "vacuity": [], "p_unknown": [], "alpha": []}
    for start in range(0, X.shape[0], batch_size):
        xb = torch.as_tensor(X[start : start + batch_size], dtype=torch.float32, device=device)
        out = model.inference(xb)
        for key in outputs:
            value = out[key] if key in out else None
            if value is not None:
                outputs[key].append(value.detach().cpu().numpy())
    return {key: np.concatenate(value, axis=0) for key, value in outputs.items() if value}


def fit_temperature(probs: np.ndarray, y: np.ndarray, max_iter: int = 300) -> float:
    logits = np.log(np.clip(probs, 1e-8, 1.0))
    logits_t = torch.as_tensor(logits, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.long)
    log_t = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = torch.exp(log_t).clamp(0.05, 20.0)
        loss = F.cross_entropy(logits_t / temperature, y_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_t).detach().clamp(0.05, 20.0).item())


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probs, 1e-8, 1.0)) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def model_outputs_for_eval(outputs: dict[str, np.ndarray], temperature: float | None) -> dict[str, np.ndarray]:
    known_probs = trim_known_probs(outputs["p_hat"])
    if temperature is not None:
        known_probs = apply_temperature(known_probs, temperature)
    return {
        "p_hat": known_probs,
        "vacuity": outputs["vacuity"],
        "p_unknown": outputs["p_unknown"],
    }


def val_brier(model: EDLSGraph, X: np.ndarray, y: np.ndarray, device: str, batch_size: int) -> float:
    outputs = collect_outputs(model, X, device, batch_size)
    return compute_brier(trim_known_probs(outputs["p_hat"]), y)


def build_model(X: np.ndarray, device: str) -> EDLSGraph:
    return EDLSGraph(n_channels=int(X.shape[1]), T=int(X.shape[2]), n_classes=N_CLASSES).to(device)


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    s_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    lambdas: dict[str, float],
    args: argparse.Namespace,
    device: str,
) -> tuple[EDLSGraph, dict[str, Any]]:
    torch.manual_seed(args.seed)
    model = build_model(X_train, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = as_loader(X_train, y_train, s_train, args.batch_size, shuffle=True)

    best_state: dict[str, torch.Tensor] | None = None
    best_brier = float("inf")
    best_epoch = 0
    stale_epochs = 0
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
            loss = loss_dict["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        current_brier = val_brier(model, X_val, y_val, device, args.batch_size)
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_brier": current_brier})
        if current_brier < best_brier - 1e-7:
            best_brier = current_brier
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_brier": best_brier, "best_epoch": best_epoch, "epochs_ran": len(history), "history": history}


def prepare_subject(subject: int, args: argparse.Namespace) -> dict[str, Any]:
    train_path = args.data_dir / f"bci4_2a_subject{subject}_train.npz"
    eval_path = args.data_dir / f"bci4_2a_subject{subject}_eval.npz"
    target_path = args.data_dir / f"difficulty_targets_subject{subject}.pkl"
    train = load_npz_session(train_path)
    eval_ = load_npz_session(eval_path)
    y_train, y_eval, classes = encode_labels(train["y"], eval_["y"])
    s_target = load_difficulty_targets(target_path, train["X"].shape[0], train.get("trial_id"))

    aligner = euclidean_alignment_matrix(train["X"])
    X_train_aligned = apply_channel_matrix(train["X"], aligner)
    X_eval_aligned = apply_channel_matrix(eval_["X"], aligner)
    train_idx, val_idx = train_val_split(X_train_aligned.shape[0], args.seed + subject)

    return {
        "subject": subject,
        "classes": classes,
        "X_train": X_train_aligned[train_idx],
        "y_train": y_train[train_idx],
        "s_train": s_target[train_idx],
        "X_val": X_train_aligned[val_idx],
        "y_val": y_train[val_idx],
        "s_val": s_target[val_idx],
        "X_eval": X_eval_aligned,
        "y_eval": y_eval,
        "paths": {"train": str(train_path), "eval": str(eval_path), "difficulty_targets": str(target_path)},
    }


def sweep_best_config(args: argparse.Namespace, device: str) -> tuple[dict[str, float], list[dict[str, Any]]]:
    grid = hyperparameter_grid()
    records = []
    best: tuple[float, dict[str, float]] | None = None
    for config_id, lambdas in enumerate(grid, start=1):
        scores = []
        for subject in SWEEP_SUBJECTS:
            prepared = prepare_subject(subject, args)
            _, info = train_model(
                prepared["X_train"],
                prepared["y_train"],
                prepared["s_train"],
                prepared["X_val"],
                prepared["y_val"],
                lambdas,
                args,
                device,
            )
            scores.append(info["best_val_brier"])
            records.append({"config_id": config_id, "subject": subject, "lambdas": lambdas, **info})
        mean_brier = float(np.mean(scores))
        if best is None or mean_brier < best[0]:
            best = (mean_brier, dict(lambdas))
        print(f"sweep {config_id:03d}/{len(grid)} lambdas={lambdas} mean_val_brier={mean_brier:.6f}", flush=True)
    assert best is not None
    return best[1], records


def evaluate_subject(subject: int, lambdas: dict[str, float], args: argparse.Namespace, device: str) -> dict[str, Any]:
    prepared = prepare_subject(subject, args)
    model, train_info = train_model(
        prepared["X_train"],
        prepared["y_train"],
        prepared["s_train"],
        prepared["X_val"],
        prepared["y_val"],
        lambdas,
        args,
        device,
    )

    temperature = None
    val_outputs = collect_outputs(model, prepared["X_val"], device, args.batch_size)
    val_probs = trim_known_probs(val_outputs["p_hat"])
    if not args.no_temperature:
        temperature = fit_temperature(val_probs, prepared["y_val"])

    def model_fn(X: np.ndarray) -> dict[str, np.ndarray]:
        raw = collect_outputs(model, np.asarray(X, dtype=np.float32), device, args.batch_size)
        return model_outputs_for_eval(raw, temperature)

    vacuity_metrics = evaluate_model(model_fn, prepared["X_eval"], prepared["y_eval"], edl_abstention_vacuity)
    unknown_metrics = evaluate_model(model_fn, prepared["X_eval"], prepared["y_eval"], edl_abstention_unknown)

    return {
        "subject": subject,
        "protocol": "BCI_IV_2a A0xT -> A0xE",
        "classes": prepared["classes"],
        "paths": prepared["paths"],
        "lambdas": lambdas,
        "train_info": {k: v for k, v in train_info.items() if k != "history"},
        "temperature": temperature,
        "metrics": {
            "vacuity": make_jsonable(vacuity_metrics),
            "unknown": make_jsonable(unknown_metrics),
        },
    }


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


def metric_scalars(subject_result: dict[str, Any]) -> dict[str, float]:
    flat = {}
    for abstention_name, metrics in subject_result["metrics"].items():
        for key in ("accuracy", "ece", "brier"):
            flat[f"{abstention_name}_{key}"] = float(metrics[key])
        for coverage, values in metrics["selective_acc"].items():
            flat[f"{abstention_name}_selective_acc_{coverage}"] = float(values["accuracy"])
            flat[f"{abstention_name}_coverage_{coverage}"] = float(values["coverage"])
    return flat


def aggregate_results(subject_results: list[dict[str, Any]], lambdas: dict[str, float], sweep_records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [metric_scalars(result) for result in subject_results]
    keys = sorted({key for row in rows for key in row})
    summary = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows if np.isfinite(row.get(key, np.nan))], dtype=float)
        summary[key] = {
            "mean": float(values.mean()) if values.size else None,
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0 if values.size == 1 else None,
        }
    return {
        "protocol": "BCI_IV_2a A0xT -> A0xE",
        "subjects": [result["subject"] for result in subject_results],
        "selected_lambdas": lambdas,
        "sweep_subjects": list(SWEEP_SUBJECTS),
        "sweep_records": make_jsonable(sweep_records),
        "metrics": summary,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(make_jsonable(payload), f, indent=2, sort_keys=True)
        f.write("\n")


def dry_run(args: argparse.Namespace) -> int:
    subjects = [args.subject] if args.subject is not None else list(range(1, N_SUBJECTS + 1))
    print("Primary CalibMI dry run")
    print(f"Subjects: {subjects}")
    print(f"Hyperparameter configs: {len(hyperparameter_grid())}; sweep subjects: {list(SWEEP_SUBJECTS)}")
    print(f"Temperature scaling: {'off' if args.no_temperature else 'on'}")
    for subject in subjects:
        print(
            f"S{subject:02d}: "
            f"{args.data_dir / f'bci4_2a_subject{subject}_train.npz'} | "
            f"{args.data_dir / f'bci4_2a_subject{subject}_eval.npz'} | "
            f"{args.data_dir / f'difficulty_targets_subject{subject}.pkl'} -> "
            f"{args.results_dir / f'edl_sgraph_subject{subject}.json'}"
        )
    print(f"Aggregate -> {args.results_dir / 'edl_sgraph_aggregate.json'}")
    return 0


def main() -> int:
    args = parse_args()
    if args.dry_run:
        return dry_run(args)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    subjects = [args.subject] if args.subject is not None else list(range(1, N_SUBJECTS + 1))

    print(f"Running primary experiment on device={device}", flush=True)
    selected_lambdas, sweep_records = sweep_best_config(args, device)
    print(f"Selected lambdas from S01/S05 sweep: {selected_lambdas}", flush=True)

    subject_results = []
    for subject in subjects:
        print(f"Training/evaluating subject S{subject:02d}", flush=True)
        result = evaluate_subject(subject, selected_lambdas, args, device)
        write_json(args.results_dir / f"edl_sgraph_subject{subject}.json", result)
        subject_results.append(result)

    aggregate = aggregate_results(subject_results, selected_lambdas, sweep_records)
    write_json(args.results_dir / "edl_sgraph_aggregate.json", aggregate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
