#!/usr/bin/env python3
"""Run minimal calibration-first EEGNet+EDL sanity experiments."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adaptation import apply_channel_matrix, euclidean_alignment_matrix
from baselines.mdrm_t import MDRMT, audit_leakage
from decode import fit_predict_subject
from eegnet import (
    EEGNetConfig,
    EEGNetEDLConfig,
    fit_predict_eegnet,
    fit_predict_eegnet_edl,
    fit_predict_eegnet_edl_calibrated,
)
from evaluate import compute_brier, compute_ece, compute_selective_acc
from experiments.run_primary import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_SEED,
    apply_temperature,
    encode_labels as encode_bci_labels,
    fit_temperature,
    load_npz_session,
    train_val_split,
)


LOSS_GRID = {
    "ce": {"lambda_brier": 0.0, "lambda_kl": 0.0},
    "ce_brier": {"lambda_brier": 0.5, "lambda_kl": 0.0},
    "ce_brier_kl": {"lambda_brier": 0.5, "lambda_kl": 0.01},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=int, action="append", default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results") / "minimal_edl")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--skip-mdrm", action="store_true")
    parser.add_argument("--ea", action="store_true", help="Apply train-split Euclidean Alignment to EEGNet/EDL only.")
    parser.add_argument(
        "--edl-temperature",
        action="store_true",
        help="Fit validation-only temperature scaling for EDL p_hat metrics.",
    )
    parser.add_argument(
        "--include-variants",
        action="store_true",
        help="Run raw minimal models plus the EA/EDL-temperature variants in one combined artifact set.",
    )
    return parser.parse_args()


def normalize_proba(proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba, dtype=float)
    return proba / np.maximum(proba.sum(axis=1, keepdims=True), 1e-12)


def metrics_for(y_true: np.ndarray, proba: np.ndarray, abstention_scores: np.ndarray | None = None) -> dict[str, float]:
    proba = normalize_proba(proba)
    if abstention_scores is None:
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


def prepare_bci_subject(subject: int, args: argparse.Namespace) -> dict[str, Any]:
    train_path = args.data_dir / f"bci4_2a_subject{subject}_train.npz"
    eval_path = args.data_dir / f"bci4_2a_subject{subject}_eval.npz"
    train = load_npz_session(train_path)
    eval_ = load_npz_session(eval_path)
    y_train, y_eval, classes = encode_bci_labels(train["y"], eval_["y"])
    train_idx, val_idx = train_val_split(train["X"].shape[0], args.seed + subject)
    return {
        "subject": subject,
        "classes": classes,
        "X_train_raw": train["X"][train_idx],
        "y_train": y_train[train_idx],
        "X_val_raw": train["X"][val_idx],
        "y_val": y_train[val_idx],
        "X_dev_raw": train["X"],
        "y_dev": y_train,
        "X_eval_raw": eval_["X"],
        "y_eval": y_eval,
        "paths": {"train": str(train_path), "eval": str(eval_path)},
    }


def ea_audit(enabled: bool) -> dict[str, Any]:
    return {
        "used": bool(enabled),
        "fit_split": "train" if enabled else None,
        "eval_used_for_fit": False,
    }


def eegnet_data_view(data: dict[str, Any], use_ea: bool) -> dict[str, Any]:
    """Return EEGNet arrays, optionally aligned with train-only EA."""

    if not use_ea:
        return {
            "X_train": data["X_train_raw"],
            "X_val": data["X_val_raw"],
            "X_dev": data["X_dev_raw"],
            "X_eval": data["X_eval_raw"],
            "ea": ea_audit(False),
        }

    aligner = euclidean_alignment_matrix(data["X_train_raw"])
    X_train = apply_channel_matrix(data["X_train_raw"], aligner)
    X_val = apply_channel_matrix(data["X_val_raw"], aligner)
    X_eval = apply_channel_matrix(data["X_eval_raw"], aligner)
    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_dev": np.concatenate([X_train, X_val], axis=0),
        "X_eval": X_eval,
        "ea": {
            **ea_audit(True),
            "matrix_shape": list(aligner.shape),
            "eps": 1e-6,
        },
    }


def fit_classical_decoders(data: dict[str, Any]) -> dict[str, np.ndarray]:
    pred = fit_predict_subject(
        subject=int(data["subject"]),
        X_train=data["X_dev_raw"],
        y_train=data["y_dev"],
        X_test=data["X_eval_raw"],
        y_test=data["y_eval"],
        model_names=("LDA", "SVM"),
    )
    if not np.array_equal(pred.y_true, data["y_eval"]):
        raise RuntimeError("Classical decoder label encoding does not match shared encoding.")
    return {
        "lda": pred.proba["LDA"],
        "svm": pred.proba["SVM"],
        "lda_svm_vote": 0.5 * (pred.proba["LDA"] + pred.proba["SVM"]),
    }


def fit_mdrm(data: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    model = MDRMT(coverage_targets=(0.4, 0.6, 0.8))
    model.fit(data["X_train_raw"], data["y_train"], data["X_val_raw"], data["y_val"])
    result = model.evaluate(data["X_eval_raw"], data["y_eval"])
    audit_leakage(model)
    return model.predict_proba(data["X_eval_raw"]), {"metrics": result, "audit_trail": model.audit_trail_}


def fit_eegnet_ts(data: dict[str, Any], args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    view = eegnet_data_view(data, args.ea)
    cfg = EEGNetConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        random_state=args.seed,
    )
    y_val, p_val = fit_predict_eegnet(
        X_train=view["X_train"],
        y_train=data["y_train"],
        X_test=view["X_val"],
        y_test=data["y_val"],
        config=cfg,
    )
    if not np.array_equal(y_val, data["y_val"]):
        raise RuntimeError("EEGNet validation label encoding does not match shared encoding.")
    temperature = fit_temperature(p_val, data["y_val"])
    y_eval, p_eval = fit_predict_eegnet(
        X_train=view["X_dev"],
        y_train=np.concatenate([data["y_train"], data["y_val"]], axis=0),
        X_test=view["X_eval"],
        y_test=data["y_eval"],
        config=cfg,
    )
    if not np.array_equal(y_eval, data["y_eval"]):
        raise RuntimeError("EEGNet eval label encoding does not match shared encoding.")
    return apply_temperature(p_eval, temperature), {
        "euclidean_alignment": view["ea"],
        "temperature": float(temperature),
        "temperature_at_upper_clamp": bool(temperature >= 19.999),
    }


def fit_edl_grid(data: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    view = eegnet_data_view(data, args.ea)
    outputs = {}
    for name, loss_cfg in LOSS_GRID.items():
        cfg = EEGNetEDLConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            random_state=args.seed,
            lambda_brier=loss_cfg["lambda_brier"],
            lambda_kl=loss_cfg["lambda_kl"],
        )
        temperature = None
        temperature_at_upper_clamp = False
        if args.edl_temperature:
            y_val, val_outputs, y_eval, edl_outputs, info = fit_predict_eegnet_edl_calibrated(
                X_train=view["X_train"],
                y_train=data["y_train"],
                X_calib=view["X_val"],
                y_calib=data["y_val"],
                X_eval=view["X_eval"],
                y_eval=data["y_eval"],
                config=cfg,
            )
            if not np.array_equal(y_val, data["y_val"]):
                raise RuntimeError("EEGNet-EDL validation label encoding does not match shared encoding.")
            if not np.array_equal(y_eval, data["y_eval"]):
                raise RuntimeError("EEGNet-EDL label encoding does not match shared encoding.")
            temperature = fit_temperature(val_outputs["p_hat"], data["y_val"])
            temperature_at_upper_clamp = bool(temperature >= 19.999)
            val_info = {
                "calibration_split": "validation",
                "calibration_trials": int(data["y_val"].shape[0]),
            }
        else:
            val_info = None
            y_eval, edl_outputs, info = fit_predict_eegnet_edl(
                X_train=view["X_dev"],
                y_train=np.concatenate([data["y_train"], data["y_val"]], axis=0),
                X_test=view["X_eval"],
                y_test=data["y_eval"],
                config=cfg,
            )
            if not np.array_equal(y_eval, data["y_eval"]):
                raise RuntimeError("EEGNet-EDL label encoding does not match shared encoding.")
        p_hat_for_metrics = edl_outputs["p_hat"]
        if temperature is not None:
            p_hat_for_metrics = apply_temperature(p_hat_for_metrics, temperature)
        info = {
            **info,
            "euclidean_alignment": view["ea"],
            "temperature_scaling": {
                "enabled": bool(args.edl_temperature),
                "fit_split": "validation" if args.edl_temperature else None,
                "eval_used_for_fit": False,
                "temperature": float(temperature) if temperature is not None else None,
                "temperature_at_upper_clamp": temperature_at_upper_clamp,
                "vacuity_temperature_scaled": False,
            },
            "validation_temperature_fit_info": val_info,
            "chance_brier": float((len(data["classes"]) - 1) / len(data["classes"])),
        }
        info["temperature"] = info["temperature_scaling"]["temperature"]
        info["temperature_at_upper_clamp"] = temperature_at_upper_clamp
        outputs[f"eegnet_edl_{name}"] = {
            "metrics": metrics_for(data["y_eval"], p_hat_for_metrics, edl_outputs["vacuity"]),
            "model_info": info,
        }
    return outputs


def sanity_flags(result: dict[str, Any]) -> list[str]:
    flags = []
    for model_name, model in result["models"].items():
        info = model.get("model_info", {})
        history = info.get("history") or []
        if model_name.startswith("eegnet_edl"):
            if info.get("best_epoch") == 1:
                flags.append(f"{model_name}: best_epoch=1")
            if history and history[-1]["train_loss"] >= history[0]["train_loss"]:
                flags.append(f"{model_name}: train_loss did not decrease")
            if info.get("best_val_brier", 0.0) >= info.get("chance_brier", float("inf")):
                flags.append(f"{model_name}: validation Brier did not improve below chance")
            if info.get("vacuity_wrong_minus_correct", 0.0) <= 0:
                flags.append(f"{model_name}: vacuity not higher on wrong predictions")
        if info.get("temperature_at_upper_clamp"):
            flags.append(f"{model_name}: temperature hit upper clamp")
    return flags


def run_subject(subject: int, args: argparse.Namespace) -> dict[str, Any]:
    data = prepare_bci_subject(subject, args)
    models: dict[str, Any] = {}

    for model_name, proba in fit_classical_decoders(data).items():
        models[model_name] = {"metrics": metrics_for(data["y_eval"], proba)}

    if not args.skip_mdrm:
        p_mdrm, mdrm_info = fit_mdrm(data)
        models["mdrm_t"] = {"metrics": metrics_for(data["y_eval"], p_mdrm), "model_info": mdrm_info}

    p_eegnet, eegnet_info = fit_eegnet_ts(data, args)
    models["eegnet_ts"] = {"metrics": metrics_for(data["y_eval"], p_eegnet), "model_info": eegnet_info}
    models.update(fit_edl_grid(data, args))

    result = {
        "protocol": "BCI_IV_2a_minimal_calibration_sanity",
        "subject": subject,
        "classes": data["classes"],
        "paths": data["paths"],
        "settings": {
            "euclidean_alignment_for_eegnet_edl": ea_audit(args.ea),
            "edl_temperature_scaling": {
                "enabled": bool(args.edl_temperature),
                "fit_split": "validation" if args.edl_temperature else None,
                "eval_used_for_fit": False,
                "vacuity_temperature_scaled": False,
            },
        },
        "models": models,
    }
    result["sanity_flags"] = sanity_flags(result)
    return result


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def combined_variant_name(model_name: str, ea: bool, edl_temperature: bool) -> str:
    if not ea and not edl_temperature:
        return model_name
    suffixes = []
    if ea:
        suffixes.append("ea")
    if model_name.startswith("eegnet_edl") and edl_temperature:
        suffixes.append("edltemp")
    return f"{model_name}_{'_'.join(suffixes)}"


def merge_subject_conditions(raw: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    models = dict(raw["models"])
    for model_name, model in variant["models"].items():
        if not model_name.startswith("eegnet_edl"):
            continue
        models[combined_variant_name(model_name, ea=True, edl_temperature=True)] = model

    result = {
        **raw,
        "settings": {
            "protocol_conditions": [
                {
                    "name": "raw_minimal",
                    "euclidean_alignment_for_eegnet_edl": ea_audit(False),
                    "edl_temperature_scaling": {
                        "enabled": False,
                        "fit_split": None,
                        "eval_used_for_fit": False,
                        "vacuity_temperature_scaled": False,
                    },
                },
                {
                    "name": "ea_edl_temperature",
                    "euclidean_alignment_for_eegnet_edl": ea_audit(True),
                    "edl_temperature_scaling": {
                        "enabled": True,
                        "fit_split": "validation",
                        "eval_used_for_fit": False,
                        "vacuity_temperature_scaled": False,
                    },
                },
            ]
        },
        "models": models,
    }
    result["sanity_flags"] = sanity_flags(result)
    return result


def run_subject_edl_only(subject: int, args: argparse.Namespace) -> dict[str, Any]:
    data = prepare_bci_subject(subject, args)
    models = fit_edl_grid(data, args)
    result = {
        "protocol": "BCI_IV_2a_minimal_calibration_sanity",
        "subject": subject,
        "classes": data["classes"],
        "paths": data["paths"],
        "settings": {
            "euclidean_alignment_for_eegnet_edl": ea_audit(args.ea),
            "edl_temperature_scaling": {
                "enabled": bool(args.edl_temperature),
                "fit_split": "validation" if args.edl_temperature else None,
                "eval_used_for_fit": False,
                "vacuity_temperature_scaled": False,
            },
        },
        "models": models,
    }
    result["sanity_flags"] = sanity_flags(result)
    return result


def write_minimal_summaries(results: list[dict[str, Any]], results_dir: Path) -> None:
    rows = []
    for result in results:
        for model_name, model in result["models"].items():
            info = model.get("model_info", {})
            ea = info.get("euclidean_alignment", {})
            temp_enabled = bool(info.get("temperature_scaling", {}).get("enabled", "temperature" in info))
            row = {
                "subject": result["subject"],
                "model": model_name,
                "variant": model_variant(model_name, bool(ea.get("used", False)), temp_enabled),
                "ea": bool(ea.get("used", False)),
                "edl_temperature": temp_enabled if model_name.startswith("eegnet_edl") else "",
                **model["metrics"],
                "best_epoch": info.get("best_epoch", ""),
                "best_val_brier": info.get("best_val_brier", ""),
                "vacuity_wrong_minus_correct": info.get("vacuity_wrong_minus_correct", ""),
                "temperature": info.get("temperature", ""),
                "temperature_at_upper_clamp": info.get("temperature_at_upper_clamp", ""),
            }
            rows.append(row)

    subject_path = results_dir / "minimal_edl_subject_metrics.csv"
    fields = [
        "subject",
        "model",
        "variant",
        "ea",
        "edl_temperature",
        "accuracy",
        "ECE",
        "Brier",
        "sel_acc_60",
        "coverage_60",
        "best_epoch",
        "best_val_brier",
        "vacuity_wrong_minus_correct",
        "temperature",
        "temperature_at_upper_clamp",
    ]
    subject_path.parent.mkdir(parents=True, exist_ok=True)
    with subject_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = results_dir / "minimal_edl_summary.csv"
    metric_fields = [
        "accuracy",
        "ECE",
        "Brier",
        "sel_acc_60",
        "coverage_60",
        "best_epoch",
        "best_val_brier",
        "vacuity_wrong_minus_correct",
        "temperature",
    ]
    summary_fields = ["variant", "model", "ea", "edl_temperature"]
    summary_fields.extend(f"{metric}_{stat}" for metric in metric_fields for stat in ("mean", "std"))
    summary_fields.append("temperature_upper_clamp_count")
    groups: dict[tuple[str, str, bool, Any], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["variant"], row["model"], row["ea"], row["edl_temperature"])
        groups.setdefault(key, []).append(row)
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for (variant, model, ea, edl_temperature), group in sorted(groups.items()):
            out = {"variant": variant, "model": model, "ea": ea, "edl_temperature": edl_temperature}
            for metric in metric_fields:
                values = np.array([float(row[metric]) for row in group if row[metric] not in ("", None)], dtype=float)
                out[f"{metric}_mean"] = float(values.mean()) if values.size else ""
                out[f"{metric}_std"] = float(values.std(ddof=0)) if values.size else ""
            out["temperature_upper_clamp_count"] = sum(row["temperature_at_upper_clamp"] is True for row in group)
            writer.writerow(out)


def model_variant(model_name: str, ea: bool, edl_temperature: bool) -> str:
    if "_ea" in model_name or "_edltemp" in model_name:
        return model_name
    return combined_variant_name(model_name, ea, edl_temperature)


def aggregate_rows(results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    subject_rows = []
    for result in results:
        for model_name, model in result["models"].items():
            info = model.get("model_info", {})
            subject_rows.append(
                {
                    "subject": int(result["subject"]),
                    "model": model_name,
                    **model["metrics"],
                    "best_epoch": info.get("best_epoch"),
                    "best_val_brier": info.get("best_val_brier"),
                    "vacuity_wrong_minus_correct": info.get("vacuity_wrong_minus_correct"),
                    "temperature": info.get("temperature"),
                    "temperature_at_upper_clamp": info.get("temperature_at_upper_clamp"),
                    "sanity_flags": [
                        flag
                        for flag in result.get("sanity_flags", [])
                        if flag.startswith(f"{model_name}:")
                    ],
                }
            )

    summary = []
    for model_name in sorted({row["model"] for row in subject_rows}):
        rows = [row for row in subject_rows if row["model"] == model_name]
        out = {"model": model_name, "n_subjects": len(rows)}
        for metric in ("accuracy", "ECE", "Brier", "sel_acc_60", "coverage_60"):
            values = np.array([float(row[metric]) for row in rows], dtype=float)
            out[metric] = {"mean": float(values.mean()), "std": float(values.std(ddof=0))}
        for metric in ("best_epoch", "best_val_brier", "vacuity_wrong_minus_correct", "temperature"):
            values = np.array([float(row[metric]) for row in rows if row[metric] is not None], dtype=float)
            out[metric] = {
                "mean": float(values.mean()) if values.size else None,
                "std": float(values.std(ddof=0)) if values.size else None,
            }
        out["temperature_upper_clamp_count"] = int(sum(row["temperature_at_upper_clamp"] is True for row in rows))
        out["sanity_flag_count"] = int(sum(len(row["sanity_flags"]) for row in rows))
        summary.append(out)
    return subject_rows, summary


def paired_delta_summary(subject_rows: list[dict[str, Any]], baseline: str) -> dict[str, Any]:
    metrics = ("accuracy", "ECE", "Brier", "sel_acc_60")
    by_model = {row["model"] for row in subject_rows}
    out = {}
    for model_name in sorted(by_model):
        if model_name == baseline:
            continue
        model_rows = {row["subject"]: row for row in subject_rows if row["model"] == model_name}
        base_rows = {row["subject"]: row for row in subject_rows if row["model"] == baseline}
        subjects = sorted(set(model_rows) & set(base_rows))
        if not subjects:
            continue
        out[model_name] = {}
        for metric in metrics:
            deltas = np.array([float(model_rows[s][metric]) - float(base_rows[s][metric]) for s in subjects], dtype=float)
            out[model_name][metric] = {"mean": float(deltas.mean()), "std": float(deltas.std(ddof=0))}
    return out


def verdict_from_summary(summary: list[dict[str, Any]]) -> tuple[str, str | None]:
    edl_models = [row for row in summary if row["model"].startswith("eegnet_edl")]
    if not edl_models:
        return "failed", None
    vote = next((row for row in summary if row["model"] == "lda_svm_vote"), None)
    eegnet = next((row for row in summary if row["model"] == "eegnet_ts"), None)
    candidates = [
        row
        for row in edl_models
        if row["vacuity_wrong_minus_correct"]["mean"] is not None
        and row["vacuity_wrong_minus_correct"]["mean"] > 0.0
        and row["Brier"]["mean"] < 0.75
    ]
    if not candidates:
        return "failed", None
    best = min(candidates, key=lambda row: (row["Brier"]["mean"], -row["accuracy"]["mean"], row["ECE"]["mean"]))
    if vote and best["accuracy"]["mean"] >= vote["accuracy"]["mean"] and best["Brier"]["mean"] <= vote["Brier"]["mean"]:
        return "confirmed", str(best["model"])
    if eegnet and (
        best["Brier"]["mean"] <= eegnet["Brier"]["mean"]
        or best["ECE"]["mean"] <= eegnet["ECE"]["mean"]
        or best["accuracy"]["mean"] >= eegnet["accuracy"]["mean"]
    ):
        return "promising-but-mixed", str(best["model"])
    return "failed", str(best["model"])


def write_summary_json(path: Path, results: list[dict[str, Any]]) -> dict[str, Any]:
    subject_rows, summary = aggregate_rows(results)
    verdict, main_model = verdict_from_summary(summary)
    payload = {
        "protocol": "BCI_IV_2a_A0xT_to_A0xE_minimal_calibration_first",
        "n_subjects": len(results),
        "metrics": summary,
        "paired_deltas_vs_eegnet_ts": paired_delta_summary(subject_rows, "eegnet_ts"),
        "paired_deltas_vs_lda_svm_vote": paired_delta_summary(subject_rows, "lda_svm_vote"),
        "sanity_flags_by_subject": {str(result["subject"]): result.get("sanity_flags", []) for result in results},
        "verdict": verdict,
        "main_edl_model": main_model,
        "evaluation_rules": {
            "higher_is_better": ["accuracy", "sel_acc_60"],
            "lower_is_better": ["ECE", "Brier"],
            "vacuity_wrong_minus_correct": "necessary_but_not_sufficient_support_for_uncertainty",
        },
    }
    write_json(path, payload)
    return payload


def _fmt_mean_std(summary_row: dict[str, Any], metric: str) -> str:
    value = summary_row[metric]
    return f"{value['mean']:.4f} ({value['std']:.4f})"


def _delta_text(delta: dict[str, Any], metric: str) -> str:
    value = delta[metric]
    return f"{value['mean']:+.4f} ({value['std']:.4f})"


def write_readout(path: Path, summary_payload: dict[str, Any]) -> None:
    metrics = summary_payload["metrics"]
    lines = [
        "# Minimal EEGNet-EDL Readout",
        "",
        "Protocol: BCI IV-2a A0xT -> A0xE, with train/validation split inside A0xT only.",
        "",
        "## Aggregate Metrics",
        "",
        "| model | accuracy | ECE | Brier | sel_acc_60 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in metrics:
        lines.append(
            f"| {row['model']} | {_fmt_mean_std(row, 'accuracy')} | {_fmt_mean_std(row, 'ECE')} | "
            f"{_fmt_mean_std(row, 'Brier')} | {_fmt_mean_std(row, 'sel_acc_60')} |"
        )

    for title, key in (
        ("Paired Deltas vs EEGNet+TS", "paired_deltas_vs_eegnet_ts"),
        ("Paired Deltas vs LDA+SVM Vote", "paired_deltas_vs_lda_svm_vote"),
    ):
        lines.extend(["", f"## {title}", "", "| model | accuracy | ECE | Brier | sel_acc_60 |", "|---|---:|---:|---:|---:|"])
        for model_name, delta in summary_payload[key].items():
            lines.append(
                f"| {model_name} | {_delta_text(delta, 'accuracy')} | {_delta_text(delta, 'ECE')} | "
                f"{_delta_text(delta, 'Brier')} | {_delta_text(delta, 'sel_acc_60')} |"
            )

    lines.extend(["", "## Per-Subject Sanity Flags", "", "| subject | flags |", "|---:|---|"])
    for subject, flags in summary_payload["sanity_flags_by_subject"].items():
        lines.append(f"| {subject} | {'; '.join(flags) if flags else 'none'} |")

    lines.extend(["", "## Vacuity Wrong Minus Correct", "", "| model | mean (std) |", "|---|---:|"])
    for row in metrics:
        value = row["vacuity_wrong_minus_correct"]
        if value["mean"] is not None:
            lines.append(f"| {row['model']} | {value['mean']:.4f} ({value['std']:.4f}) |")

    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"Verdict: {summary_payload['verdict']}.",
            f"Main EDL model: {summary_payload['main_edl_model'] or 'none'}.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def condition_args(args: argparse.Namespace, *, ea: bool, edl_temperature: bool) -> argparse.Namespace:
    cloned = argparse.Namespace(**vars(args))
    cloned.ea = ea
    cloned.edl_temperature = edl_temperature
    cloned.include_variants = False
    return cloned


def main() -> int:
    args = parse_args()
    subjects = args.subject or list(range(1, 10))
    results = []
    for subject in subjects:
        if args.include_variants:
            raw = run_subject(subject, condition_args(args, ea=False, edl_temperature=False))
            variant = run_subject_edl_only(subject, condition_args(args, ea=True, edl_temperature=True))
            result = merge_subject_conditions(raw, variant)
        else:
            result = run_subject(subject, args)
        results.append(result)
        out = args.results_dir / f"minimal_edl_bci4_2a_subject{subject}.json"
        write_json(out, result)
        print(f"wrote {out}")
        if result["sanity_flags"]:
            print("sanity flags:")
            for flag in result["sanity_flags"]:
                print(f"  - {flag}")
    write_minimal_summaries(results, args.results_dir)
    summary_payload = write_summary_json(args.results_dir / "minimal_edl_summary.json", results)
    write_readout(args.results_dir / "minimal_edl_readout.md", summary_payload)
    print(f"wrote {args.results_dir / 'minimal_edl_subject_metrics.csv'}")
    print(f"wrote {args.results_dir / 'minimal_edl_summary.csv'}")
    print(f"wrote {args.results_dir / 'minimal_edl_summary.json'}")
    print(f"wrote {args.results_dir / 'minimal_edl_readout.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
