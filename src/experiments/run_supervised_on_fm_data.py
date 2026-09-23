#!/usr/bin/env python3
"""Supervised baseline trained on the *foundation-model* input, as a control.

The audit compares foundation models against a small supervised convnet, but the
two arms are fed differently by design: the convnet gets 8--30 Hz narrowband at
250 Hz, the foundation models get 0.1--75 Hz broadband at 200 Hz, because that is
what each was built for. A reviewer is entitled to ask whether the gap is really
about the models or just about the filter.

This closes that: the same ShallowConvNet, trained on the identical broadband
200 Hz arrays the foundation models see. If it still reaches roughly its usual
accuracy, preprocessing is not the confound and the gap is attributable to the
models. If it collapses too, the comparison has to be reframed.

Selection discipline matches the rest of the project: the checkpoint and the
scalar temperature are chosen on the selection subset drawn from the training
session; the held-out session is read once.

Usage::

    python src/experiments/run_supervised_on_fm_data.py
    python src/experiments/run_supervised_on_fm_data.py --dataset bnci2014_004
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
    ArchConfig,
    CroppedShallowConvNet,
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
    DATASET_LABELS,
    DEFAULT_SEED,
    SELECTION_SPLIT,
    encode_labels,
    load_fm_split,
)


def standardize(train: np.ndarray, *others: np.ndarray):
    """Per-channel z-scoring with statistics from the fit split only.

    Broadband EEG in volts spans orders of magnitude more dynamic range than the
    narrowband arrays this convnet normally sees; without scaling it does not
    train at all. Statistics come from the fit split so no held-out information
    reaches the model.
    """
    mean = train.mean(axis=(0, 2), keepdims=True)
    std = train.std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    return [((a - mean) / std).astype(np.float32) for a in (train, *others)]


#: Supervised architectures available as comparators.
#:
#: ShallowConvNet is the project's inherited baseline. The rest are the
#: state-of-the-art motor-imagery architectures this manuscript cites, taken
#: from braindecode's reference implementations. Including them matters because
#: the paper's own finding is that baseline quality decides a benchmark's
#: verdict -- comparing foundation models only against a 2017 reference
#: architecture would be the exact mistake the paper documents.
BRAINDECODE_ARCHS = {"atcnet": "ATCNet", "eegconformer": "EEGConformer", "eegnet": "EEGNet"}
ARCHS = ("shallow_convnet",) + tuple(BRAINDECODE_ARCHS)

#: Learning-rate grid for the braindecode architectures, selected per subject on
#: validation. Their published architecture *is* their tuned configuration, so
#: only the optimiser is searched; leaving the learning rate unsearched would
#: hand them the same disadvantage this paper criticises.
LR_GRID = (1e-2, 1e-3, 3e-4)


def build_arch(name, n_chans, n_times, n_outputs, sfreq, arch_cfg):
    if name == "shallow_convnet":
        return CroppedShallowConvNet(
            n_channels=n_chans, n_times=n_times, n_classes=n_outputs,
            arch=ArchConfig(**arch_cfg),
        )
    import braindecode.models as bd
    return getattr(bd, BRAINDECODE_ARCHS[name])(
        n_chans=n_chans, n_times=n_times, n_outputs=n_outputs, sfreq=float(sfreq)
    )


def config_grid(name: str, tune: bool, lr_grid=None):
    """Per-subject search space, always scored on the selection subset."""
    if name == "shallow_convnet":
        grid = ARCH_GRID if tune else (dict(ARCH_GRID[-1]),)
        return [{"arch": dict(a), "lr": None} for a in grid]
    lrs = tuple(lr_grid or LR_GRID) if tune else (1e-3,)
    return [{"arch": {}, "lr": lr} for lr in lrs]


#: Architecture grid for broadband input, searched per subject on validation.
#: The default temporal_kernel=64 was chosen for 8-30 Hz at 250 Hz. Broadband
#: 0.1-75 Hz at 200 Hz has both a different sample spacing and much more
#: high-frequency content, so reusing that kernel would understate what a
#: properly configured supervised model achieves -- and the headline claim turns
#: on exactly that number.
ARCH_GRID = (
    {"temporal_kernel": 16, "temporal_filters": 40},
    {"temporal_kernel": 32, "temporal_filters": 40},
    {"temporal_kernel": 51, "temporal_filters": 40},  # 256 ms at 200 Hz
    {"temporal_kernel": 64, "temporal_filters": 16},  # the narrowband default
)


def train_one(cfg: dict, X, y, device: str, args: argparse.Namespace, sfreq: float):
    """Train one configuration; return (model, selection Brier, best epoch)."""
    set_all_seeds(args.seed)
    model = build_arch(
        args.arch, X["fit"].shape[1], X["fit"].shape[2],
        int(len(np.unique(y["fit"]))), sfreq, cfg.get("arch", {}),
    ).to(device)
    lr = cfg.get("lr") or args.lr
    tensors = {k: torch.from_numpy(v).to(device) for k, v in X.items()}
    targets = {k: torch.from_numpy(v).to(device) for k, v in y.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    n_fit = tensors["fit"].shape[0]
    batch = min(args.batch_size, n_fit)

    best_state, best_brier, best_epoch = None, np.inf, -1
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(n_fit, device=device)
        for start in range(0, n_fit, batch):
            idx = order[start : start + batch]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(tensors["fit"][idx]), targets["fit"][idx])
            loss.backward()
            optimizer.step()
            if hasattr(model, "apply_max_norm"):
                model.apply_max_norm()
        model.eval()
        with torch.no_grad():
            logits = model(tensors["selection"]).cpu().numpy()
        brier = metric_bundle(y["selection"], proba_from_logits(logits, 1.0))["Brier"]
        if brier < best_brier:
            best_brier, best_epoch = brier, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, float(best_brier), int(best_epoch)


def run_subject(subject: int, args: argparse.Namespace, device: str) -> dict[str, Any]:
    train, provenance = load_fm_split(args.fm_dir, subject, "train", args.dataset)
    heldout, _ = load_fm_split(args.fm_dir, subject, "eval", args.dataset)

    fit_idx, sel_idx = _split_train_val_indices(np.asarray(train["y"]), args.seed)
    classes, (y_fit, y_sel, y_held) = encode_labels(
        np.asarray(train["y"])[fit_idx], np.asarray(train["y"])[sel_idx], np.asarray(heldout["y"])
    )

    keep = int(args.seconds * float(train["sfreq"]))
    X_fit, X_sel, X_held = standardize(
        np.asarray(train["X"])[fit_idx][..., :keep],
        np.asarray(train["X"])[sel_idx][..., :keep],
        np.asarray(heldout["X"])[..., :keep],
    )

    n_classes = int(len(classes))
    X = {"fit": X_fit, "selection": X_sel, "heldout": X_held}
    y = {"fit": y_fit, "selection": y_sel, "heldout": y_held}

    # Architecture selected per subject on the selection subset -- the same
    # freedom the foundation-model probe gets for its C and pooling, so neither
    # arm is handicapped by a hyperparameter chosen for the other's input.
    sfreq = float(train["sfreq"])
    grid = config_grid(args.arch, args.tune, getattr(args, "lr_grid", None))
    best = {"brier": np.inf, "cfg": None, "model": None, "epoch": -1}
    for cfg in grid:
        model, brier, epoch = train_one(cfg, X, y, device, args, sfreq)
        if brier < best["brier"]:
            best = {"brier": brier, "cfg": cfg, "model": model, "epoch": epoch}

    model = best["model"]
    tensors = {k: torch.from_numpy(v).to(device) for k, v in X.items()}
    model.eval()
    with torch.no_grad():
        sel_logits = model(tensors["selection"]).cpu().numpy()
        held_logits = model(tensors["heldout"]).cpu().numpy()

    temperature, at_clamp = fit_temperature_from_logits(sel_logits, y_sel, args.temperature_max)
    val_metrics = metric_bundle(y_sel, proba_from_logits(sel_logits, temperature))
    held_metrics = metric_bundle(y_held, proba_from_logits(held_logits, temperature))
    uncal = metric_bundle(y_held, proba_from_logits(held_logits, 1.0))

    audit = {
        "adaptation": f"supervised_{args.arch}",
        "architecture": args.arch,
        "selected_epoch": int(best["epoch"]),
        "selection_Brier_at_selected_epoch": float(best["brier"]),
        "selected_config": best["cfg"],
        "arch_grid_searched": bool(args.tune),
        "epochs": int(args.epochs),
        # `lr` is the CLI default and is kept for continuity with runs already on
        # disk. It is *not* what trained the reported model whenever the grid
        # searched a learning rate -- `selected_lr` is. Recording only the former
        # made an audit field disagree with the optimiser, which is the kind of
        # provenance gap this project treats as a defect.
        "lr": float(args.lr),
        "selected_lr": float(best["cfg"].get("lr") or args.lr),
        "lr_grid": [float(v) for v in args.lr_grid],
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "source_bandpass_hz": provenance.get("bandpass_hz"),
        "source_sfreq_hz": provenance.get("sfreq_hz"),
    }
    row = {
        "dataset": args.dataset,
        "dataset_label": DATASET_LABELS.get(args.dataset, args.dataset),
        "subject": subject,
        "n_classes": n_classes,
        "experiment_key": (
            f"{args.arch}_{args.band}"
            + (f"_{args.tag}" if args.tag else "")
            + ("_tuned" if args.tune else "")
        ),
        "experiment": f"{args.arch}_supervised_{args.band}",
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
        "chance_accuracy": float(1.0 / n_classes),
        "chance_Brier": float((n_classes - 1) / n_classes),
        "temperature": float(temperature),
        "temperature_at_clamp": bool(at_clamp),
        "n_patches": -1,
        "window_seconds": float(args.seconds),
        "selection_split": SELECTION_SPLIT,
        "eval_labels_used_for_selection": False,
        "audit_json": json.dumps(audit),
    }
    print(
        f"  s{subject}: heldout acc {held_metrics['accuracy']:.4f} "
        f"Brier {held_metrics['Brier']:.4f} T={temperature:.3f}"
    )
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="bci4_2a")
    ap.add_argument("--arch", default="shallow_convnet", choices=list(ARCHS))
    ap.add_argument(
        "--band",
        default="broadband",
        help="Label for the input band, used in the experiment key and filename. "
        "Use 'narrowband' when --fm-dir points at the 8-30 Hz arrays.",
    )
    ap.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    ap.add_argument("--fm-dir", type=Path, default=Path("data") / "fm")
    ap.add_argument("--out-dir", type=Path, default=Path("results") / "fm_probe")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument(
        "--tune",
        action="store_true",
        help="Search ARCH_GRID per subject on validation. Without it the model "
        "keeps the architecture tuned for narrowband input, which understates "
        "what a supervised baseline reaches on broadband.",
    )
    ap.add_argument(
        "--tag",
        default=None,
        help="Distinguishing label for an arm that differs from another only in "
        "its input, e.g. --tag ica. Without it two such arms share an "
        "experiment_key and a filename, and the second silently overwrites the "
        "first -- exactly the confound the arm was created to measure.",
    )
    ap.add_argument(
        "--lr-grid",
        type=float,
        nargs="+",
        default=list(LR_GRID),
        help="Learning rates searched per subject on validation, for the "
        "braindecode architectures. Extend it when the selected value lands on "
        "an edge: a grid whose endpoint keeps winning has been truncated, not "
        "searched.",
    )
    ap.add_argument("--temperature-max", type=float, default=20.0)
    ap.add_argument("--deterministic", dest="deterministic", action="store_true", default=True)
    ap.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    configure_determinism(bool(args.deterministic))
    print(f"device={device} dataset={args.dataset} supervised-on-broadband control")

    rows = [run_subject(s, args, device) for s in args.subjects]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"supervised_{args.arch}_{args.band}"
    if args.tag:
        stem += f"_{args.tag}"
    if args.tune:
        stem += "_tuned"
    if args.dataset != "bci4_2a":
        stem += f"_{args.dataset}"
    if args.seed != DEFAULT_SEED:
        # Replicate seeds sit alongside the canonical seed-42 run rather than
        # overwriting it. See run_fm_probe.py for the same convention.
        stem += f"_seed{args.seed}"
    out_csv = args.out_dir / f"{stem}_subject_metrics.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / f"{stem}_run.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "arch": args.arch,
                "band": args.band,
                "subjects": args.subjects,
                "seed": args.seed,
                "epochs": args.epochs,
                "platform": platform_fingerprint(device),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    accs = [r["heldout_accuracy"] for r in rows]
    print(f"\nmean held-out accuracy {np.mean(accs):.4f} (n={len(rows)})")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
