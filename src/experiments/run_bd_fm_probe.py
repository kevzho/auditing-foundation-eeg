#!/usr/bin/env python3
"""Validation-locked evaluation of braindecode-hosted EEG foundation models.

Companion to ``run_fm_probe.py``, which evaluates the hand-ported LaBraM. This
runner covers models loaded from braindecode's reference implementations with
author-released weights, so a claim about *foundation models* rests on more than
one architecture and more than one person's port.

Selection discipline is unchanged: the training session splits into fit and
selection subsets; probe regularization, feature pooling, the scalar
temperature, and the fine-tuning checkpoint are all chosen on the selection
subset. Held-out data is read once, for the reported row.

Usage::

    python src/experiments/run_bd_fm_probe.py --model cbramod --mode frozen
    python src/experiments/run_bd_fm_probe.py --model cbramod --mode finetune --epochs 60
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments.run_calibration_workflow import (  # noqa: E402
    _split_train_val_indices,
    configure_determinism,
    metric_bundle,
    platform_fingerprint,
    set_all_seeds,
)
from experiments.run_classical_anchor_competence import (  # noqa: E402
    fit_temperature_from_logits,
    proba_from_logits,
)
from experiments.run_fm_probe import (  # noqa: E402
    C_GRID,
    DATASET_LABELS,
    DEFAULT_SEED,
    HELDOUT_SPLIT,
    SELECTION_SPLIT,
    encode_labels,
    freeze_lower,
    load_fm_split,
    probe_logits,
    resume_completed_subjects,
    write_probability_npz,
)
from models.braindecode_fms import (  # noqa: E402
    as_fm_spec,
    build_model,
    count_blocks,
    extract_features,
    frozen_modules,
    get_spec,
    layer_id,
)
from models.fm_adapters import prepare_for_fm  # noqa: E402

POOLINGS = ("mean", "flatten")


def build_param_groups(
    model: nn.Module, lr: float, weight_decay: float, layer_decay: float
) -> list[dict]:
    """Layer-wise learning-rate decay over braindecode's module names."""
    n_blocks = count_blocks(model)
    groups: dict[tuple[int, bool], dict] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        decay = not (param.ndim <= 1 or name.endswith(".bias"))
        layer = layer_id(name, n_blocks)
        key = (layer, decay)
        if key not in groups:
            groups[key] = {
                "params": [],
                "weight_decay": weight_decay if decay else 0.0,
                "lr": lr * (layer_decay ** (n_blocks + 1 - layer)),
                "layer_id": layer,
            }
        groups[key]["params"].append(param)
    return list(groups.values())


def freeze_lower_bd(model: nn.Module, n_frozen_blocks: int) -> int:
    if n_frozen_blocks <= 0:
        return 0
    n_blocks = count_blocks(model)
    frozen = 0
    for name, param in model.named_parameters():
        if layer_id(name, n_blocks) <= n_frozen_blocks:
            param.requires_grad_(False)
            frozen += 1
    return frozen


def run_frozen(
    spec, prepared: dict, y: dict, args: argparse.Namespace, device: str
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Frozen trunk + logistic probe, with pooling and C chosen on selection."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    n_times = prepared["fit"].X.shape[-1]
    n_classes = int(len(np.unique(y["fit"])))
    model, report = build_model(
        spec, prepared["fit"].ch_names, n_times, n_classes,
        device=device, pretrained=not args.random_init,
    )

    def probe(c: float):
        return make_pipeline(
            StandardScaler(), LogisticRegression(C=c, max_iter=2000, random_state=args.seed)
        )

    best = {"brier": np.inf, "pooling": None, "C": None, "feats": None}
    for pooling in POOLINGS:
        feats = {
            name: extract_features(
                model, spec, p.X, pooling=pooling, device=device, batch_size=args.batch_size
            )
            for name, p in prepared.items()
        }
        for c in C_GRID:
            fitted = probe(c).fit(feats["fit"], y["fit"])
            brier = metric_bundle(
                y["selection"], fitted.predict_proba(feats["selection"])
            )["Brier"]
            if brier < best["brier"]:
                best = {"brier": brier, "pooling": pooling, "C": c, "feats": feats}

    feats = best["feats"]
    fitted = probe(best["C"]).fit(feats["fit"], y["fit"])
    audit = {
        "adaptation": "frozen_linear_probe",
        "selected_C": float(best["C"]),
        "selected_pooling": best["pooling"],
        "selection_Brier_at_selected_C": float(best["brier"]),
        "feature_dim": int(feats["fit"].shape[1]),
        **report,
    }
    return probe_logits(feats["selection"], fitted), probe_logits(feats["heldout"], fitted), audit


def run_finetune(
    spec, prepared: dict, y: dict, args: argparse.Namespace, device: str
) -> tuple[np.ndarray, np.ndarray, dict]:
    """End-to-end fine-tuning; checkpoint selected on the selection subset."""
    n_times = prepared["fit"].X.shape[-1]
    n_classes = int(len(np.unique(y["fit"])))
    model, report = build_model(
        spec, prepared["fit"].ch_names, n_times, n_classes,
        device=device, pretrained=not args.random_init,
    )

    inputs = {
        name: torch.from_numpy(np.asarray(p.X, dtype=np.float32)).to(device)
        / spec.amplitude_divisor
        for name, p in prepared.items()
    }
    targets = {k: torch.from_numpy(v).to(device) for k, v in y.items()}

    n_frozen = freeze_lower_bd(model, int(args.freeze_blocks))
    still = frozen_modules(model)
    groups = build_param_groups(model, args.lr, args.weight_decay, args.layer_decay)
    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    top_layer = max(g["layer_id"] for g in groups)
    is_head = [g["layer_id"] == top_layer for g in optimizer.param_groups]
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    n_fit = inputs["fit"].shape[0]
    batch = min(args.finetune_batch_size, n_fit)
    steps_per_epoch = max(1, int(np.ceil(n_fit / batch)))
    total_steps = steps_per_epoch * max(args.epochs, 1)
    warmup_steps = min(int(args.warmup_frac * total_steps), max(total_steps - 1, 0))

    def lr_scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))

    best_state, best_brier, best_epoch = None, np.inf, -1
    # Full per-epoch trace. The argmin alone cannot show *how* a run converges,
    # and the initialisation claim is a claim about the shape of this curve.
    val_brier_curve: list[float] = []
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        for module in still:  # frozen blocks stay deterministic
            module.eval()
        encoder_frozen = epoch < args.head_warmup_epochs
        order = torch.randperm(n_fit, device=device)
        for start in range(0, n_fit, batch):
            idx = order[start : start + batch]
            scale = lr_scale(global_step)
            for group, base, head in zip(optimizer.param_groups, base_lrs, is_head):
                group["lr"] = 0.0 if (encoder_frozen and not head) else base * scale
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs["fit"][idx]), targets["fit"][idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            global_step += 1

        model.eval()
        with torch.no_grad():
            logits = model(inputs["selection"]).cpu().numpy()
        brier = metric_bundle(y["selection"], proba_from_logits(logits, 1.0))["Brier"]
        val_brier_curve.append(float(brier))
        if brier < best_brier:
            best_brier, best_epoch = brier, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if args.log_epochs and (epoch + 1) % args.log_epochs == 0:
            print(f"    epoch {epoch + 1}/{args.epochs} selection Brier {brier:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        selection_logits = model(inputs["selection"]).cpu().numpy()
        heldout_logits = model(inputs["heldout"]).cpu().numpy()

    audit = {
        "adaptation": "finetune_end_to_end",
        "selected_epoch": int(best_epoch),
        "selection_Brier_at_selected_epoch": float(best_brier),
        "val_brier_curve": val_brier_curve,
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "layer_decay": float(args.layer_decay),
        "weight_decay": float(args.weight_decay),
        "label_smoothing": float(args.label_smoothing),
        "finetune_batch_size": int(batch),
        "steps_per_epoch": int(steps_per_epoch),
        "total_steps": int(total_steps),
        "warmup_steps": int(warmup_steps),
        "freeze_blocks": int(args.freeze_blocks),
        "n_frozen_tensors": int(n_frozen),
        "n_frozen_modules_in_eval": len(still),
        "head_warmup_epochs": int(args.head_warmup_epochs),
        "n_trainable_parameters": int(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        ),
        **report,
    }
    return selection_logits, heldout_logits, audit


def run_subject(
    subject: int, args: argparse.Namespace, device: str
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    spec = get_spec(args.model)
    fm_spec = as_fm_spec(spec)

    train, provenance = load_fm_split(args.fm_dir, subject, "train", args.dataset)
    heldout, _ = load_fm_split(args.fm_dir, subject, "eval", args.dataset)
    band = tuple(provenance.get("bandpass_hz", (0.1, 75.0)))

    fit_idx, sel_idx = _split_train_val_indices(np.asarray(train["y"]), args.seed)
    classes, (y_fit, y_sel, y_held) = encode_labels(
        np.asarray(train["y"])[fit_idx], np.asarray(train["y"])[sel_idx], np.asarray(heldout["y"])
    )
    y = {"fit": y_fit, "selection": y_sel, "heldout": y_held}

    rows: list[dict[str, Any]] = []
    probas: dict[str, np.ndarray] = {
        "y_val": np.asarray(y["selection"], dtype=int),
        "y_eval": np.asarray(y["heldout"], dtype=int),
    }
    for n_patches in args.patches:
        set_all_seeds(args.seed)
        keep = n_patches * spec.patch_samples

        prepared = {}
        for name, X in (
            ("fit", np.asarray(train["X"])[fit_idx]),
            ("selection", np.asarray(train["X"])[sel_idx]),
            ("heldout", np.asarray(heldout["X"])),
        ):
            p = prepare_for_fm(
                X, train["ch_names"], train["sfreq"], fm_spec,
                source_bandpass=band, strict_band=not args.allow_band_mismatch,
            )
            if p.X.shape[-1] < keep:
                raise SystemExit(
                    f"{n_patches} patches needs {keep} samples but only {p.X.shape[-1]} available"
                )
            p.X = p.X[..., :keep]
            p.n_patches = n_patches
            prepared[name] = p

        runner = run_frozen if args.mode == "frozen" else run_finetune
        sel_logits, held_logits, audit = runner(spec, prepared, y, args, device)

        temperature, at_clamp = fit_temperature_from_logits(
            sel_logits, y["selection"], args.temperature_max
        )
        sel_proba = proba_from_logits(sel_logits, temperature)
        held_proba = proba_from_logits(held_logits, temperature)
        val_metrics = metric_bundle(y["selection"], sel_proba)
        held_metrics = metric_bundle(y["heldout"], held_proba)
        uncal = metric_bundle(y["heldout"], proba_from_logits(held_logits, 1.0))

        key = f"{spec.name}_{args.mode}"
        if args.random_init:
            key += "_randinit"
        if len(args.patches) > 1:
            key += f"_p{n_patches}"
        if args.tag:
            key += f"_{args.tag}"
        probas[f"{key}__val_proba"] = np.asarray(sel_proba, dtype=float)
        probas[f"{key}__eval_proba"] = np.asarray(held_proba, dtype=float)
        rows.append(
            {
                "dataset": args.dataset,
                "dataset_label": DATASET_LABELS.get(args.dataset, args.dataset),
                "subject": subject,
                "n_classes": int(len(classes)),
                "experiment_key": key,
                "experiment": f"{spec.name}_{args.mode}",
                "val_accuracy": val_metrics["accuracy"],
                "val_Brier": val_metrics["Brier"],
                "val_ECE": val_metrics["ECE"],
                "val_NLL": val_metrics["NLL"],
                "heldout_accuracy": held_metrics["accuracy"],
                "heldout_Brier": held_metrics["Brier"],
                "heldout_ECE": held_metrics["ECE"],
                "heldout_NLL": held_metrics["NLL"],
                "heldout_Brier_uncalibrated": uncal["Brier"],
                "heldout_ECE_uncalibrated": uncal["ECE"],
                "chance_accuracy": float(1.0 / len(classes)),
                "chance_Brier": float((len(classes) - 1) / len(classes)),
                "temperature": float(temperature),
                "temperature_at_clamp": bool(at_clamp),
                "n_patches": int(n_patches),
                "window_seconds": float(keep / spec.sfreq),
                "selection_split": SELECTION_SPLIT,
                "eval_labels_used_for_selection": False,
                "audit_json": json.dumps(audit),
            }
        )
        print(
            f"  s{subject} {key}: heldout acc {held_metrics['accuracy']:.4f} "
            f"Brier {held_metrics['Brier']:.4f} (uncal {uncal['Brier']:.4f}) T={temperature:.3f}"
        )
    return rows, probas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="cbramod")
    ap.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    ap.add_argument("--dataset", default="bci4_2a")
    ap.add_argument("--mode", choices=["frozen", "finetune"], default="frozen")
    ap.add_argument("--patches", type=int, nargs="+", default=[4])
    ap.add_argument("--fm-dir", type=Path, default=Path("data") / "fm")
    ap.add_argument(
        "--allow-band-mismatch",
        action="store_true",
        help=(
            "Accept input outside the model's pretraining band. Off by default because a "
            "silent band mismatch confounds the audit. Enable only for the deliberate "
            "narrowband control, where feeding 8-30 Hz input IS the experiment."
        ),
    )
    ap.add_argument("--out-dir", type=Path, default=Path("results") / "fm_probe")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--temperature-max", type=float, default=20.0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--layer-decay", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--finetune-batch-size", type=int, default=16)
    ap.add_argument("--warmup-frac", type=float, default=0.1)
    ap.add_argument("--clip-grad", type=float, default=1.0)
    ap.add_argument("--freeze-blocks", type=int, default=0)
    ap.add_argument("--head-warmup-epochs", type=int, default=0)
    ap.add_argument(
        "--random-init",
        action="store_true",
        help="Identical architecture, random weights. The control that separates "
        "'pretraining does not transfer' from 'this architecture cannot learn "
        "from a few hundred trials'.",
    )
    ap.add_argument("--tag", default=None)
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip subjects already completed by an interrupted run of the same "
        "configuration. Resumption is refused if the recorded arguments differ.",
    )
    ap.add_argument("--log-epochs", type=int, default=0)
    ap.add_argument("--deterministic", dest="deterministic", action="store_true", default=True)
    ap.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    configure_determinism(bool(args.deterministic))
    print(
        f"device={device} model={args.model} mode={args.mode} patches={args.patches} "
        f"deterministic={bool(args.deterministic)}"
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"fm_probe_{args.model}_{args.mode}"
    if args.dataset != "bci4_2a":
        stem += f"_{args.dataset}"
    if args.random_init:
        stem += "_randinit"
    if args.tag:
        stem += f"_{args.tag}"
    if args.seed != DEFAULT_SEED:
        # Seed 42 is the canonical run and keeps its established filename;
        # replicate seeds are suffixed so they sit alongside it instead of
        # overwriting it. experiment_key is deliberately NOT suffixed -- it
        # names the method, and the seed is provenance, recorded in the run
        # JSON and in this filename.
        stem += f"_seed{args.seed}"
    out_csv = args.out_dir / f"{stem}_subject_metrics.csv"
    partial_csv = args.out_dir / f"{stem}_subject_metrics.csv.partial"
    fingerprint_path = args.out_dir / f"{stem}_run.json.partial"
    fingerprint = {
        k: v
        for k, v in vars(args).items()
        if k not in {"out_dir", "fm_dir", "checkpoint", "device", "resume"}
    }
    fingerprint = json.loads(json.dumps(fingerprint, default=str))

    # Each subject is flushed as it finishes, so an interrupted run loses one
    # subject rather than the whole config. The rows accumulate under a
    # .partial name and are renamed only on completion, so a truncated file can
    # never be mistaken for a finished n=9 result by a downstream script.
    rows = resume_completed_subjects(partial_csv, fingerprint_path, fingerprint) if args.resume else []
    done = {int(r["subject"]) for r in rows}
    fingerprint_path.write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")

    mode = "a" if rows else "w"
    writer: csv.DictWriter | None = None
    with partial_csv.open(mode, newline="", encoding="utf-8") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        for subject in args.subjects:
            if subject in done:
                continue
            subject_rows, subject_probas = run_subject(subject, args, device)
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(subject_rows[0].keys()))
                writer.writeheader()
            writer.writerows(subject_rows)
            handle.flush()
            write_probability_npz(
                out_dir=args.out_dir,
                stem=stem,
                dataset=args.dataset,
                subject=subject,
                payload=subject_probas,
            )
            rows.extend(subject_rows)

    partial_csv.replace(out_csv)
    fingerprint_path.unlink(missing_ok=True)
    (args.out_dir / f"{stem}_run.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "mode": args.mode,
                "dataset": args.dataset,
                "random_init": bool(args.random_init),
                "tag": args.tag,
                "patches": args.patches,
                "subjects": args.subjects,
                "seed": args.seed,
                "lr": args.lr,
                "epochs": args.epochs,
                "freeze_blocks": args.freeze_blocks,
                "head_warmup_epochs": args.head_warmup_epochs,
                "heldout_split": HELDOUT_SPLIT,
                "platform": platform_fingerprint(device),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # float() rather than direct use: rows recovered by --resume come back from
    # the CSV as strings, and np.mean on a string dtype raises.
    accs = [float(r["heldout_accuracy"]) for r in rows]
    briers = [float(r["heldout_Brier"]) for r in rows]
    print(f"\nmean held-out accuracy {np.mean(accs):.4f}  Brier {np.mean(briers):.4f}  (n={len(rows)})")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
