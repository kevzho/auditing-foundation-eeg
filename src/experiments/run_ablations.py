#!/usr/bin/env python3
"""Run CalibMI EDL-SGraph ablations on BCI IV 2a A0xT -> A0xE."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate import compute_brier, evaluate_model
from experiments.run_primary import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_SEED,
    MAX_EPOCHS,
    N_CLASSES,
    N_SUBJECTS,
    PATIENCE,
    as_loader,
    collect_outputs,
    fit_temperature,
    make_jsonable,
    model_outputs_for_eval,
    prepare_subject,
    trim_known_probs,
    write_json,
)
from models.dual_graph_encoder import DualGraphEncoder  # noqa: E402
from models.edl_sgraph import EDLSGraph  # noqa: E402
from models.spectral_graph import SpectralCovGraph  # noqa: E402


STANDARD_LAMBDAS = {
    "lambda_B": 0.5,
    "lambda_KL": 0.1,
    "lambda_diff": 0.1,
    "lambda_sel": 0.05,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=int, choices=range(1, N_SUBJECTS + 1), help="Run one subject only.")
    parser.add_argument(
        "--ablation",
        choices=tuple(ABLATION_NAMES),
        action="append",
        help="Run one ablation. May be supplied multiple times. Defaults to all.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned ablations/files and exit.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--primary-aggregate",
        type=Path,
        default=Path("results/edl_sgraph_aggregate.json"),
        help="Optional primary aggregate JSON containing selected_lambdas.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default=None, help="Torch device. Defaults to cuda when available, else cpu.")
    parser.add_argument("--no-temperature", action="store_true", help="Disable validation temperature scaling.")
    return parser.parse_args()


def entropy_abstention(outputs: dict[str, Any]) -> np.ndarray:
    probs = np.asarray(outputs["p_hat"], dtype=float)
    return -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)


def maxprob_abstention(outputs: dict[str, Any]) -> np.ndarray:
    probs = np.asarray(outputs["p_hat"], dtype=float)
    return 1.0 - probs.max(axis=1)


def vacuity_abstention(outputs: dict[str, Any]) -> np.ndarray:
    return np.asarray(outputs["vacuity"], dtype=float).reshape(-1)


class ZeroDifficultyHead(nn.Module):
    """Parameter-free replacement for the auxiliary difficulty head."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.new_zeros(x.shape[0])


class AnatomicalOnlyEncoder(DualGraphEncoder):
    """DualGraph-compatible encoder that keeps only the anatomical branch."""

    def forward(self, X: torch.Tensor, reliability_stats: torch.Tensor | None = None) -> torch.Tensor:
        if X.dim() != 3:
            raise ValueError(f"Expected X with shape [batch, C, T], got {tuple(X.shape)}")
        if X.size(1) != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {X.size(1)}")

        self._ensure_gats(X.size(-1), X.device)
        reliability = self._effective_reliability(reliability_stats).to(device=X.device, dtype=X.dtype)
        A_anat = self.A_anat.to(device=X.device, dtype=X.dtype)
        return self.gat_anat(X, A_anat, reliability)


class NoReliabilityDualGraphEncoder(DualGraphEncoder):
    """DualGraph encoder with the r-channel prior removed from GAT attention."""

    def forward(self, X: torch.Tensor, reliability_stats: torch.Tensor | None = None) -> torch.Tensor:
        if X.dim() != 3:
            raise ValueError(f"Expected X with shape [batch, C, T], got {tuple(X.shape)}")
        if X.size(1) != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {X.size(1)}")

        self._ensure_gats(X.size(-1), X.device)
        reliability = torch.ones(self.n_channels, device=X.device, dtype=X.dtype)
        A_anat = self.A_anat.to(device=X.device, dtype=X.dtype)
        A_spec = self.spectral_graph.compute_adjacency(X)
        A_spec = self._masked_row_softmax(self._topk_adjacency(A_spec, self.k_spec))
        return self.gat_anat(X, A_anat, reliability) + self.gat_spec(X, A_spec, reliability)


def build_model(X: np.ndarray, device: str, ablation_name: str) -> EDLSGraph:
    model = EDLSGraph(n_channels=int(X.shape[1]), T=int(X.shape[2]), n_classes=N_CLASSES)
    if ablation_name == "no_dual_graph":
        model.graph_encoder = SpectralCovGraph(
            n_channels=int(X.shape[1]),
            d_s=model.d_s,
            sfreq=model.sfreq,
        )
    elif ablation_name == "anat_only":
        model.graph_encoder = AnatomicalOnlyEncoder(
            n_channels=int(X.shape[1]),
            d_s=model.d_s,
            sfreq=model.sfreq,
        )
    elif ablation_name == "no_reliability_priors":
        model.graph_encoder = NoReliabilityDualGraphEncoder(
            n_channels=int(X.shape[1]),
            d_s=model.d_s,
            sfreq=model.sfreq,
        )
    elif ablation_name == "no_difficulty_head":
        model.s_head = ZeroDifficultyHead()
    return model.to(device)


@dataclass(frozen=True)
class AblationSpec:
    name: str
    lambdas: dict[str, float]
    abstention_fn: Callable[[dict[str, Any]], np.ndarray] = vacuity_abstention
    use_ea: bool = True


ABLATION_NAMES = (
    "loss_ce_only",
    "loss_ce_brier",
    "full_loss",
    "no_dual_graph",
    "anat_only",
    "no_reliability_priors",
    "no_ea",
    "no_difficulty_head",
    "no_selective_loss",
    "vacuity_vs_entropy",
    "vacuity_vs_maxprob",
)


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


def ablation_specs(standard_lambdas: dict[str, float]) -> dict[str, AblationSpec]:
    ce_only = {**standard_lambdas, "lambda_B": 0.0, "lambda_KL": 0.0, "lambda_diff": 0.0, "lambda_sel": 0.0}
    ce_brier = {**standard_lambdas, "lambda_KL": 0.0, "lambda_diff": 0.0, "lambda_sel": 0.0}
    no_diff = {**standard_lambdas, "lambda_diff": 0.0}
    no_sel = {**standard_lambdas, "lambda_sel": 0.0}
    return {
        "loss_ce_only": AblationSpec("loss_ce_only", ce_only),
        "loss_ce_brier": AblationSpec("loss_ce_brier", ce_brier),
        "full_loss": AblationSpec("full_loss", dict(standard_lambdas)),
        "no_dual_graph": AblationSpec("no_dual_graph", dict(standard_lambdas)),
        "anat_only": AblationSpec("anat_only", dict(standard_lambdas)),
        "no_reliability_priors": AblationSpec("no_reliability_priors", dict(standard_lambdas)),
        "no_ea": AblationSpec("no_ea", dict(standard_lambdas), use_ea=False),
        "no_difficulty_head": AblationSpec("no_difficulty_head", no_diff),
        "no_selective_loss": AblationSpec("no_selective_loss", no_sel),
        "vacuity_vs_entropy": AblationSpec("vacuity_vs_entropy", dict(standard_lambdas), entropy_abstention),
        "vacuity_vs_maxprob": AblationSpec("vacuity_vs_maxprob", dict(standard_lambdas), maxprob_abstention),
    }


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    s_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    spec: AblationSpec,
    args: argparse.Namespace,
    device: str,
) -> tuple[EDLSGraph, dict[str, Any]]:
    torch.manual_seed(args.seed)
    model = build_model(X_train, device, spec.name)
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
            _, _, loss_dict = model(xb, s_target=sb, lambdas={**spec.lambdas, "labels": yb})
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


def val_brier(model: EDLSGraph, X: np.ndarray, y: np.ndarray, device: str, batch_size: int) -> float:
    outputs = collect_outputs(model, X, device, batch_size)
    return compute_brier(trim_known_probs(outputs["p_hat"]), y)


def prepare_subject_without_ea(subject: int, args: argparse.Namespace) -> dict[str, Any]:
    train_path = args.data_dir / f"bci4_2a_subject{subject}_train.npz"
    eval_path = args.data_dir / f"bci4_2a_subject{subject}_eval.npz"
    target_path = args.data_dir / f"difficulty_targets_subject{subject}.pkl"

    from experiments.run_primary import encode_labels, load_difficulty_targets, load_npz_session, train_val_split

    train = load_npz_session(train_path)
    eval_ = load_npz_session(eval_path)
    y_train, y_eval, classes = encode_labels(train["y"], eval_["y"])
    s_target = load_difficulty_targets(target_path, train["X"].shape[0], train.get("trial_id"))
    train_idx, val_idx = train_val_split(train["X"].shape[0], args.seed + subject)
    return {
        "subject": subject,
        "classes": classes,
        "X_train": train["X"][train_idx],
        "y_train": y_train[train_idx],
        "s_train": s_target[train_idx],
        "X_val": train["X"][val_idx],
        "y_val": y_train[val_idx],
        "s_val": s_target[val_idx],
        "X_eval": eval_["X"],
        "y_eval": y_eval,
        "paths": {"train": str(train_path), "eval": str(eval_path), "difficulty_targets": str(target_path)},
    }


def prepare_ablation_subject(subject: int, spec: AblationSpec, args: argparse.Namespace) -> dict[str, Any]:
    if spec.use_ea:
        return prepare_subject(subject, args)
    return prepare_subject_without_ea(subject, args)


def evaluate_subject(subject: int, spec: AblationSpec, args: argparse.Namespace, device: str) -> dict[str, Any]:
    prepared = prepare_ablation_subject(subject, spec, args)
    model, train_info = train_model(
        prepared["X_train"],
        prepared["y_train"],
        prepared["s_train"],
        prepared["X_val"],
        prepared["y_val"],
        spec,
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

    metrics = evaluate_model(model_fn, prepared["X_eval"], prepared["y_eval"], spec.abstention_fn)
    return {
        "ablation": spec.name,
        "subject": subject,
        "protocol": "BCI_IV_2a A0xT -> A0xE",
        "classes": prepared["classes"],
        "paths": prepared["paths"],
        "use_ea": spec.use_ea,
        "lambdas": spec.lambdas,
        "abstention_score": abstention_name(spec),
        "train_info": {k: v for k, v in train_info.items() if k != "history"},
        "temperature": temperature,
        "metrics": make_jsonable(metrics),
    }


def abstention_name(spec: AblationSpec) -> str:
    if spec.abstention_fn is entropy_abstention:
        return "entropy"
    if spec.abstention_fn is maxprob_abstention:
        return "1-max_prob"
    return "vacuity"


def subject_scalars(result: dict[str, Any]) -> dict[str, float]:
    metrics = result["metrics"]
    return {
        "accuracy": float(metrics["accuracy"]),
        "ECE": float(metrics["ece"]),
        "Brier": float(metrics["brier"]),
        "sel_acc_60": float(metrics["selective_acc"]["0.6"]["accuracy"]),
    }


def aggregate_results(subject_results: list[dict[str, Any]], spec: AblationSpec) -> dict[str, Any]:
    rows = [subject_scalars(result) for result in subject_results]
    summary = {}
    for key in ("accuracy", "ECE", "Brier", "sel_acc_60"):
        values = np.asarray([row[key] for row in rows if np.isfinite(row.get(key, np.nan))], dtype=float)
        summary[key] = {
            "mean": float(values.mean()) if values.size else None,
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0 if values.size == 1 else None,
        }
    return {
        "ablation": spec.name,
        "protocol": "BCI_IV_2a A0xT -> A0xE",
        "subjects": [result["subject"] for result in subject_results],
        "use_ea": spec.use_ea,
        "lambdas": spec.lambdas,
        "abstention_score": abstention_name(spec),
        "metrics": summary,
    }


def write_summary_csv(path: Path, aggregates: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ablation", "accuracy", "ECE", "Brier", "sel_acc_60"])
        writer.writeheader()
        for aggregate in aggregates:
            row = {"ablation": aggregate["ablation"]}
            for key in ("accuracy", "ECE", "Brier", "sel_acc_60"):
                row[key] = aggregate["metrics"][key]["mean"]
            writer.writerow(row)


def dry_run(args: argparse.Namespace, specs: dict[str, AblationSpec]) -> int:
    subjects = [args.subject] if args.subject is not None else list(range(1, N_SUBJECTS + 1))
    selected = args.ablation or list(ABLATION_NAMES)
    print("CalibMI ablation dry run")
    print(f"Subjects: {subjects}")
    print(f"Ablations: {selected}")
    print(f"Temperature scaling: {'off' if args.no_temperature else 'on'}")
    for name in selected:
        spec = specs[name]
        print(f"{name}: use_ea={spec.use_ea} abstention={abstention_name(spec)} lambdas={spec.lambdas}")
        for subject in subjects:
            print(f"  S{subject:02d} -> {args.results_dir / f'ablation_{name}_subject{subject}.json'}")
        print(f"  aggregate -> {args.results_dir / f'ablation_{name}_aggregate.json'}")
    print(f"Summary CSV -> {args.results_dir / 'ablation_summary.csv'}")
    return 0


def main() -> int:
    args = parse_args()
    standard_lambdas = load_standard_lambdas(args.primary_aggregate)
    specs = ablation_specs(standard_lambdas)
    selected = args.ablation or list(ABLATION_NAMES)

    if args.dry_run:
        return dry_run(args, specs)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    subjects = [args.subject] if args.subject is not None else list(range(1, N_SUBJECTS + 1))
    print(f"Running ablations on device={device}; standard_lambdas={standard_lambdas}", flush=True)

    aggregates = []
    for name in selected:
        spec = specs[name]
        print(f"Running ablation={name} use_ea={spec.use_ea} abstention={abstention_name(spec)}", flush=True)
        subject_results = []
        for subject in subjects:
            print(f"Training/evaluating ablation={name} subject=S{subject:02d}", flush=True)
            result = evaluate_subject(subject, spec, args, device)
            write_json(args.results_dir / f"ablation_{name}_subject{subject}.json", result)
            subject_results.append(result)

        aggregate = aggregate_results(subject_results, spec)
        write_json(args.results_dir / f"ablation_{name}_aggregate.json", aggregate)
        aggregates.append(aggregate)

    write_summary_csv(args.results_dir / "ablation_summary.csv", aggregates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
