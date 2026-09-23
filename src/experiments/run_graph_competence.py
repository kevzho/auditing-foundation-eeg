#!/usr/bin/env python3
"""Graph competence expansion under leakage-safe BCI IV-2a A0xT -> A0xE."""

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

from evaluate import compute_brier, compute_ece, compute_selective_acc
from experiments.run_primary import DEFAULT_BATCH_SIZE, DEFAULT_SEED, encode_labels, load_npz_session
from models.competence_graph_eegnet import CompetenceConfig, build_competence_model


N_SUBJECTS = 9
N_CLASSES = 4
CHANCE_ACC = 1.0 / N_CLASSES
CHANCE_BRIER = (N_CLASSES - 1.0) / N_CLASSES
VARIANTS = (
    "raw_eegnet_ce",
    "raw_no_graph_compact",
    "raw_graph_residual_identity",
    "raw_graph_anatomical_fixed",
    "raw_graph_anatomical_learned_residual",
    "raw_filterbank_no_graph",
    "raw_filterbank_graph_anatomical",
)
GRAPH_MATCHES = {
    "raw_graph_residual_identity": "raw_no_graph_compact",
    "raw_graph_anatomical_fixed": "raw_no_graph_compact",
    "raw_graph_anatomical_learned_residual": "raw_no_graph_compact",
    "raw_filterbank_graph_anatomical": "raw_filterbank_no_graph",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=int, action="append", choices=range(1, N_SUBJECTS + 1))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results") / "graph_competence")
    parser.add_argument("--baseline-dir", type=Path, default=Path("results") / "minimal_edl")
    parser.add_argument("--graph-rescue-dir", type=Path, default=Path("results") / "graph_rescue")
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


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long))
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


def build_model(X: np.ndarray, args: argparse.Namespace, variant: str) -> torch.nn.Module:
    cfg = CompetenceConfig(
        n_channels=int(X.shape[1]),
        n_times=int(X.shape[2]),
        n_classes=N_CLASSES,
        variant=variant,
        dropout=float(args.dropout),
    )
    return build_competence_model(cfg)


def train_model(data: dict[str, Any], args: argparse.Namespace, variant: str, device: str) -> dict[str, Any]:
    view = preprocessing_view(data)
    torch.manual_seed(args.seed + int(data["subject"]))
    np.random.seed(args.seed + int(data["subject"]))
    model = build_model(view["X_train"], args, variant).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = make_loader(view["X_train"], data["y_train"], args.batch_size, shuffle=True)
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
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            reg = model.regularization_loss() if hasattr(model, "regularization_loss") else torch.zeros((), device=device)
            loss = loss_fn(logits, yb) + float(args.graph_reg) * reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        val_logits = collect_logits(model, view["X_val"], device, args.batch_size)
        val_loss = float(F.cross_entropy(torch.as_tensor(val_logits, dtype=torch.float32), torch.as_tensor(data["y_val"], dtype=torch.long)).item())
        val_metrics = metrics_for(data["y_val"], val_logits)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
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
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss),
            "best_val_brier": float(best_val_brier),
            "epochs_ran": len(history),
            "history": history,
            "chance_accuracy": CHANCE_ACC,
            "chance_brier": CHANCE_BRIER,
        },
    }


def load_baselines(path: Path) -> tuple[dict[str, dict[str, float]], dict[int, dict[str, dict[str, float]]]]:
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


def load_graph_rescue(path: Path) -> tuple[dict[str, dict[str, float]], dict[int, dict[str, dict[str, float]]]]:
    summary: dict[str, dict[str, float]] = {}
    subjects: dict[int, dict[str, dict[str, float]]] = {}
    subject_path = path / "graph_rescue_subject_metrics.csv"
    if subject_path.exists():
        with subject_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                subject = int(row["subject"])
                variant = row["variant"]
                subjects.setdefault(subject, {})[variant] = {
                    key: float(row[key]) for key in ("accuracy", "ECE", "Brier", "sel_acc_60", "coverage_60") if row.get(key) not in ("", None)
                }
    summary_path = path / "graph_rescue_summary.csv"
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


def sanity_flags(
    row: dict[str, Any],
    baseline_subjects: dict[int, dict[str, dict[str, float]]],
    by_subject_variant: dict[tuple[int, str], dict[str, Any]],
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
    baseline_name = GRAPH_MATCHES.get(row["variant"])
    if baseline_name:
        baseline = by_subject_variant.get((int(row["subject"]), baseline_name))
        if baseline and row["metrics"]["accuracy"] < baseline["metrics"]["accuracy"] and row["metrics"]["Brier"] > baseline["metrics"]["Brier"]:
            flags.append("graph_underperforms_matched_no_graph_accuracy_and_brier")
    return flags


def aggregate(subject_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = ("accuracy", "ECE", "Brier", "sel_acc_60", "coverage_60")
    summary = []
    for variant in sorted({row["variant"] for row in subject_rows}):
        rows = [row for row in subject_rows if row["variant"] == variant]
        out: dict[str, Any] = {"variant": variant, "n_subjects": len(rows)}
        for metric in metrics:
            values = np.array([float(row["metrics"][metric]) for row in rows], dtype=float)
            out[metric] = {"mean": float(values.mean()), "std": float(values.std(ddof=0))}
        for metric in ("best_epoch", "best_val_loss", "best_val_brier", "temperature"):
            if metric == "temperature":
                values = np.array([float(row["model_info"]["temperature_scaling"]["temperature"]) for row in rows], dtype=float)
            else:
                values = np.array([float(row["model_info"][metric]) for row in rows], dtype=float)
            out[metric] = {"mean": float(values.mean()), "std": float(values.std(ddof=0))}
        out["temperature_upper_clamp_count"] = int(sum(row["model_info"]["temperature_scaling"]["temperature_at_upper_clamp"] for row in rows))
        out["sanity_flag_count"] = int(sum(len(row["sanity_flags"]) for row in rows))
        summary.append(out)
    return summary


def paired_deltas(subject_rows: list[dict[str, Any]], baseline_subjects: dict[int, dict[str, dict[str, float]]], baseline: str) -> dict[str, Any]:
    out = {}
    for variant in sorted({row["variant"] for row in subject_rows}):
        rows = [row for row in subject_rows if baseline in baseline_subjects.get(row["subject"], {})]
        if not rows:
            continue
        rows = [row for row in rows if row["variant"] == variant]
        if not rows:
            continue
        out[variant] = {}
        for metric in ("accuracy", "ECE", "Brier", "sel_acc_60"):
            values = np.array([row["metrics"][metric] - baseline_subjects[row["subject"]][baseline][metric] for row in rows], dtype=float)
            out[variant][metric] = {"mean": float(values.mean()), "std": float(values.std(ddof=0)), "n": int(values.size)}
    return out


def graph_usefulness(subject_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    by_key = {(row["subject"], row["variant"]): row for row in subject_rows}
    for graph_variant, baseline in GRAPH_MATCHES.items():
        subjects = sorted({s for s, v in by_key if v == graph_variant} & {s for s, v in by_key if v == baseline})
        if not subjects:
            continue
        acc_delta = np.array([by_key[(s, graph_variant)]["metrics"]["accuracy"] - by_key[(s, baseline)]["metrics"]["accuracy"] for s in subjects])
        brier_delta = np.array([by_key[(s, graph_variant)]["metrics"]["Brier"] - by_key[(s, baseline)]["metrics"]["Brier"] for s in subjects])
        rows.append(
            {
                "graph_variant": graph_variant,
                "matched_baseline": baseline,
                "n_subjects": len(subjects),
                "accuracy_delta_mean": float(acc_delta.mean()),
                "brier_delta_mean": float(brier_delta.mean()),
                "subjects_accuracy_improved": int(np.sum(acc_delta > 0)),
                "graph_useful": bool(acc_delta.mean() > 0 and brier_delta.mean() <= 0.02 and (np.sum(acc_delta > 0) >= 5 or acc_delta.mean() > 0)),
            }
        )
    return rows


def decide_verdict(summary: list[dict[str, Any]], graph_rows: list[dict[str, Any]], baseline_summary: dict[str, dict[str, float]]) -> tuple[str, bool]:
    eegnet = baseline_summary.get("eegnet_ts")
    vote = baseline_summary.get("lda_svm_vote")
    if not eegnet:
        return "failed", False
    best_deep = max(summary, key=lambda r: r["accuracy"]["mean"])
    competent_deep = best_deep["accuracy"]["mean"] > eegnet["accuracy"] and best_deep["Brier"]["mean"] < eegnet["Brier"] and best_deep["sanity_flag_count"] < best_deep["n_subjects"]
    graph_ok_variants = {row["graph_variant"] for row in graph_rows if row["graph_useful"]}
    graph_summary = [row for row in summary if row["variant"] in graph_ok_variants]
    competent_graph = any(
        row["accuracy"]["mean"] > eegnet["accuracy"] and row["Brier"]["mean"] < eegnet["Brier"] and row["sanity_flag_count"] < row["n_subjects"]
        for row in graph_summary
    )
    if competent_graph:
        return "competent_graph", True
    if competent_deep:
        if vote and best_deep["accuracy"]["mean"] < vote["accuracy"]:
            return "classical_still_best", False
        return "competent_deep_no_graph", False
    return "failed", False


def write_csvs(results_dir: Path, subject_rows: list[dict[str, Any]], summary: list[dict[str, Any]]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    subject_fields = [
        "subject",
        "variant",
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
    with (results_dir / "graph_competence_subject_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=subject_fields)
        writer.writeheader()
        for row in sorted(subject_rows, key=lambda r: (r["variant"], r["subject"])):
            info = row["model_info"]
            writer.writerow(
                {
                    "subject": row["subject"],
                    "variant": row["variant"],
                    **row["metrics"],
                    "val_accuracy": row["validation_metrics"]["accuracy"],
                    "val_Brier": row["validation_metrics"]["Brier"],
                    "best_epoch": info["best_epoch"],
                    "best_val_loss": info["best_val_loss"],
                    "best_val_brier": info["best_val_brier"],
                    "temperature": info["temperature_scaling"]["temperature"],
                    "temperature_at_upper_clamp": info["temperature_scaling"]["temperature_at_upper_clamp"],
                    "sanity_flags": ";".join(row["sanity_flags"]),
                }
            )

    fields = ["variant", "n_subjects"]
    for metric in ("accuracy", "ECE", "Brier", "sel_acc_60", "coverage_60", "best_epoch", "best_val_loss", "best_val_brier", "temperature"):
        fields.extend([f"{metric}_mean", f"{metric}_std"])
    fields.extend(["temperature_upper_clamp_count", "sanity_flag_count"])
    with (results_dir / "graph_competence_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sorted(summary, key=lambda r: r["variant"]):
            flat = {"variant": row["variant"], "n_subjects": row["n_subjects"]}
            for metric in ("accuracy", "ECE", "Brier", "sel_acc_60", "coverage_60", "best_epoch", "best_val_loss", "best_val_brier", "temperature"):
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
    lines.extend(["", f"## {title}", "", "| variant | accuracy | ECE | Brier | sel_acc_60 |", "|---|---:|---:|---:|---:|"])
    for variant, vals in sorted(deltas.items()):
        lines.append(
            f"| {variant} | {vals['accuracy']['mean']:+.4f} | {vals['ECE']['mean']:+.4f} | "
            f"{vals['Brier']['mean']:+.4f} | {vals['sel_acc_60']['mean']:+.4f} |"
        )


def write_readout(results_dir: Path, payload: dict[str, Any]) -> str:
    verdict = payload["verdict"]
    if verdict == "competent_graph":
        framing = "Anatomically constrained residual graph mixing improves closed-set MI competence under leakage-safe validation, justifying a later evidential head."
    elif verdict == "competent_deep_no_graph":
        framing = "The staged protocol prevents premature graph/EDL claims; stronger temporal neural baselines learn signal, while graph novelty requires further constraint."
    else:
        framing = "Classical decoders remain strong in small-sample MI; the contribution shifts toward calibration-gated neural uncertainty and classical-to-neural distillation."

    lines = [
        "# Graph Competence Expansion Readout",
        "",
        f"Verdict: **{verdict}**",
        "",
        "## Aggregate Metrics",
        "",
        "| variant | accuracy | ECE | Brier | sel_acc_60 | flags |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(payload["metrics"], key=lambda r: r["variant"]):
        lines.append(
            f"| {row['variant']} | {fmt_mean_std(row, 'accuracy')} | {fmt_mean_std(row, 'ECE')} | "
            f"{fmt_mean_std(row, 'Brier')} | {fmt_mean_std(row, 'sel_acc_60')} | {row['sanity_flag_count']} |"
        )
    add_delta_table(lines, "Paired Deltas vs EEGNet+TS", payload["paired_deltas_vs_eegnet_ts"])
    add_delta_table(lines, "Paired Deltas vs LDA+SVM Vote", payload["paired_deltas_vs_lda_svm_vote"])
    add_delta_table(lines, "Paired Deltas vs Graph Rescue graph_ce_raw", payload["paired_deltas_vs_graph_ce_raw"])
    add_delta_table(lines, "Paired Deltas vs Graph Rescue graph_ce_no_graph_ablation", payload["paired_deltas_vs_graph_ce_no_graph_ablation"])
    lines.extend(
        [
            "",
            "## Graph Usefulness",
            "",
            "| graph variant | matched no-graph | acc delta | Brier delta | improved subjects | useful |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in payload["graph_usefulness"]:
        lines.append(
            f"| {row['graph_variant']} | {row['matched_baseline']} | {row['accuracy_delta_mean']:+.4f} | "
            f"{row['brier_delta_mean']:+.4f} | {row['subjects_accuracy_improved']}/{row['n_subjects']} | {row['graph_useful']} |"
        )
    lines.extend(["", "## Sanity Flags", ""])
    for key, flags in sorted(payload["sanity_flags_by_subject_variant"].items()):
        lines.append(f"- {key}: {', '.join(flags) if flags else 'none'}")
    lines.extend(["", "## Paper Framing", "", framing, "", "## EDL Head Justification", ""])
    lines.append("Yes, as a later stage only." if payload["edl_head_justified"] else "No. Keep EDL off until a CE model is both competent and graph-useful.")
    text = "\n".join(lines)
    (results_dir / "graph_competence_readout.md").write_text(text, encoding="utf-8")
    return text


def print_table(summary: list[dict[str, Any]], graph_rows: list[dict[str, Any]], verdict: str, edl_ok: bool) -> None:
    print("\nAggregate graph competence table")
    print("variant                              accuracy        ECE             Brier           sel_acc_60")
    for row in sorted(summary, key=lambda r: r["variant"]):
        print(
            f"{row['variant']:<36} {fmt_mean_std(row, 'accuracy'):<15} "
            f"{fmt_mean_std(row, 'ECE'):<15} {fmt_mean_std(row, 'Brier'):<15} {fmt_mean_std(row, 'sel_acc_60'):<15}"
        )
    if graph_rows:
        print("\nGraph-usefulness paired deltas")
        for row in graph_rows:
            print(
                f"{row['graph_variant']} vs {row['matched_baseline']}: "
                f"acc {row['accuracy_delta_mean']:+.4f}, Brier {row['brier_delta_mean']:+.4f}, "
                f"improved {row['subjects_accuracy_improved']}/{row['n_subjects']}, useful={row['graph_useful']}"
            )
    print(f"\nVerdict: {verdict}")
    print(f"EDL justified yet: {edl_ok}")


def main() -> int:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    subjects = args.subject or list(range(1, N_SUBJECTS + 1))
    print(f"Running graph competence on device={device}; subjects={subjects}; variants={args.variants}", flush=True)
    baseline_summary, baseline_subjects = load_baselines(args.baseline_dir)
    _, graph_rescue_subjects = load_graph_rescue(args.graph_rescue_dir)

    subject_rows: list[dict[str, Any]] = []
    by_subject_variant: dict[tuple[int, str], dict[str, Any]] = {}
    for subject in subjects:
        data = prepare_subject(subject, args.data_dir, args.seed)
        models = {}
        for variant in args.variants:
            print(f"S{subject:02d} {variant}", flush=True)
            row = train_model(data, args, variant, device)
            by_subject_variant[(subject, variant)] = row
            row["sanity_flags"] = sanity_flags(row, baseline_subjects, by_subject_variant)
            models[variant] = row
            subject_rows.append(row)
        payload = {
            "protocol": "BCI_IV_2a_A0xT_to_A0xE_graph_competence_expansion",
            "subject": subject,
            "classes": data["classes"],
            "paths": data["paths"],
            "split": data["split"],
            "models": models,
        }
        write_json(args.results_dir / f"graph_competence_subject{subject}.json", payload)

    summary = aggregate(subject_rows)
    graph_rows = graph_usefulness(subject_rows)
    verdict, edl_ok = decide_verdict(summary, graph_rows, baseline_summary)
    payload = {
        "protocol": "BCI_IV_2a_A0xT_to_A0xE_graph_competence_expansion",
        "n_subjects": len(subjects),
        "variants": args.variants,
        "metrics": summary,
        "paired_deltas_vs_eegnet_ts": paired_deltas(subject_rows, baseline_subjects, "eegnet_ts"),
        "paired_deltas_vs_lda_svm_vote": paired_deltas(subject_rows, baseline_subjects, "lda_svm_vote"),
        "paired_deltas_vs_graph_ce_raw": paired_deltas(subject_rows, graph_rescue_subjects, "graph_ce_raw"),
        "paired_deltas_vs_graph_ce_no_graph_ablation": paired_deltas(subject_rows, graph_rescue_subjects, "graph_ce_no_graph_ablation"),
        "graph_usefulness": graph_rows,
        "sanity_flags_by_subject_variant": {f"S{row['subject']:02d}:{row['variant']}": row["sanity_flags"] for row in subject_rows},
        "baseline_summary_available": bool(baseline_summary),
        "baseline_subject_metrics_available": bool(baseline_subjects),
        "graph_rescue_subject_metrics_available": bool(graph_rescue_subjects),
        "verdict": verdict,
        "edl_head_justified": edl_ok,
        "sanity_rules": {
            "best_epoch": "flag when == 1",
            "train_loss": "flag when final train loss is not below first train loss",
            "validation_accuracy": f"flag when <= chance + 0.05 ({CHANCE_ACC + 0.05:.2f})",
            "validation_brier": f"flag when >= chance Brier ({CHANCE_BRIER:.2f})",
            "temperature": "flag when validation temperature hits upper clamp",
            "eval_accuracy": "flag when below per-subject EEGNet+TS from minimal_edl_subject_metrics.csv",
            "graph": "flag graph variants that underperform matched no-graph on both accuracy and Brier",
        },
    }
    write_csvs(args.results_dir, subject_rows, summary)
    write_json(args.results_dir / "graph_competence_summary.json", payload)
    write_readout(args.results_dir, payload)
    print_table(summary, graph_rows, verdict, edl_ok)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
