#!/usr/bin/env python3
"""Validation-locked evaluation of a pretrained foundation-model encoder on MI.

Emits the same row schema as ``run_calibration_workflow.py`` so every existing
analysis script -- confidence shift, transfer gap, multiplicity, risk coverage --
works on the output unchanged.

Selection discipline is identical to the rest of the project. The training
session is split into fit/selection subsets; the probe's regularization, the
scalar temperature, and (for fine-tuning) the checkpoint are all chosen on the
selection subset. The held-out session is read once, for the final row.

Adaptation modes
----------------
``frozen``    linear probe on frozen encoder features. Weakest use of an FM, but
              it isolates what the pretrained representation already contains.
``finetune``  encoder updated end to end with a fresh classification head. This
              is what practitioners actually do, so a fair comparison against a
              supervised baseline needs it.

Usage::

    python src/experiments/run_fm_probe.py --subjects 1 --mode frozen
    python src/experiments/run_fm_probe.py --mode finetune --epochs 30
    python src/experiments/run_fm_probe.py --mode frozen --patches 2 3 4   # window sweep
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

from eeg_montage import load_npz_with_channels  # noqa: E402
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
from models.fm_adapters import LABRAM_SPEC, prepare_for_fm  # noqa: E402
from models.labram_probe import (  # noqa: E402
    LABRAM_AMPLITUDE_DIVISOR,
    channel_indices,
    extract_features,
    load_labram,
)

SELECTION_SPLIT = "A0xT_validation"
#: Display names used in the results CSVs, matching the supervised arm.
DATASET_LABELS = {
    "bci4_2a": "BCI_IV_2a",
    "bnci2014_001": "BNCI2014_001",
    "bnci2014_004": "BNCI2014_004",
    "bci_iiia": "BCI_IIIa",
    "lee2019_mi": "Lee2019_MI",
    "zhou2016": "Zhou2016",
}
HELDOUT_SPLIT = "A0xE_final_report_only"
C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)
#: Canonical seed. Runs at other seeds are replicates and are filename-suffixed.
DEFAULT_SEED = 42


def load_fm_split(
    fm_dir: Path, subject: int, split: str, dataset: str = "bci4_2a"
) -> tuple[dict, dict]:
    path = Path(fm_dir) / f"{dataset}_subject{subject}_{split}.npz"
    if not path.exists():
        raise SystemExit(
            f"missing {path}; run src/preprocess_fm.py (BCI IV-2a) or "
            f"src/preprocess_fm_moabb.py (MOABB datasets) first"
        )
    data = load_npz_with_channels(path, dataset=dataset)
    raw = np.load(path, allow_pickle=False)
    provenance = json.loads(str(raw["provenance"])) if "provenance" in raw.files else {}
    return data, provenance


def encode_labels(y_train: np.ndarray, *others: np.ndarray):
    classes = np.unique(np.asarray(y_train))
    lookup = {int(c): i for i, c in enumerate(classes)}
    out = [np.array([lookup[int(v)] for v in np.asarray(y_train)], dtype=np.int64)]
    for other in others:
        out.append(np.array([lookup[int(v)] for v in np.asarray(other)], dtype=np.int64))
    return classes, out


def probe_logits(features: np.ndarray, model) -> np.ndarray:
    """Decision function as logits, shaped (n, n_classes) even for 2 classes."""
    scores = model.decision_function(features)
    if scores.ndim == 1:  # binary: sklearn returns a single margin
        scores = np.column_stack([-scores, scores])
    return np.asarray(scores, dtype=np.float64)


def run_frozen(
    prepared: dict, y: dict, args: argparse.Namespace, device: str
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Frozen encoder + logistic probe. Returns (selection_logits, heldout_logits, audit)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model, report = load_labram(
        device=device, checkpoint=args.checkpoint, pretrained=not args.random_init
    )
    feats = {
        name: extract_features(
            model, p.X, p.ch_names, device=device, batch_size=args.batch_size
        ).features
        for name, p in prepared.items()
    }

    # Regularization chosen on the selection subset only.
    best_c, best_brier = None, np.inf
    for c in C_GRID:
        pipe = make_pipeline(
            StandardScaler(), LogisticRegression(C=c, max_iter=2000, random_state=args.seed)
        ).fit(feats["fit"], y["fit"])
        proba = pipe.predict_proba(feats["selection"])
        brier = metric_bundle(y["selection"], proba)["Brier"]
        if brier < best_brier:
            best_c, best_brier = c, brier

    probe = make_pipeline(
        StandardScaler(), LogisticRegression(C=best_c, max_iter=2000, random_state=args.seed)
    ).fit(feats["fit"], y["fit"])
    audit = {
        "adaptation": "frozen_linear_probe",
        "selected_C": float(best_c),
        "selection_Brier_at_selected_C": float(best_brier),
        "checkpoint": report["checkpoint"],
        "n_parameters": report["n_parameters"],
        "feature_dim": int(feats["fit"].shape[1]),
    }
    return probe_logits(feats["selection"], probe), probe_logits(feats["heldout"], probe), audit


class LaBraMClassifier(nn.Module):
    """Pretrained encoder with a fresh linear head, for end-to-end fine-tuning."""

    def __init__(self, encoder, embed_dim: int, n_classes: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(embed_dim, n_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, input_chans):
        return self.head(self.encoder.forward_features(x, input_chans=input_chans))


def _as_model_input(X: np.ndarray, n_patches: int, patch: int, device: str) -> torch.Tensor:
    t = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device) / LABRAM_AMPLITUDE_DIVISOR
    return t.reshape(t.shape[0], t.shape[1], n_patches, patch)


def _layer_id(name: str, n_blocks: int) -> int:
    """Depth index for layer-wise learning-rate decay.

    Embeddings sit at 0, transformer block ``i`` at ``i+1``, and the head above
    everything. Mirrors the reference ``layer_decay=0.9`` schedule, which keeps a
    fresh head from destroying pretrained features early in training.
    """
    if name.startswith("head") or name.startswith("encoder.fc_norm") or name.startswith("encoder.norm"):
        return n_blocks + 1
    if name.startswith("encoder.blocks."):
        return int(name.split(".")[2]) + 1
    return 0  # patch_embed, cls_token, pos_embed, time_embed


def freeze_lower(model: nn.Module, n_frozen_blocks: int) -> int:
    """Freeze the embeddings and the lowest ``n_frozen_blocks`` transformer blocks.

    Partial fine-tuning is the standard remedy when a small target set destroys
    pretrained low-level features. Returns the number of frozen tensors.
    """
    if n_frozen_blocks <= 0:
        return 0
    n_blocks = len(model.encoder.blocks)
    frozen = 0
    for name, param in model.named_parameters():
        layer = _layer_id(name, n_blocks)
        if layer <= n_frozen_blocks:  # layer 0 = embeddings, block i = i+1
            param.requires_grad_(False)
            frozen += 1
    return frozen


def frozen_modules(model: nn.Module) -> list[nn.Module]:
    """Modules whose parameters are all frozen; kept in eval mode while training.

    Dropout inside a block that is not learning only injects noise into features
    that cannot adapt to it. It also avoids a PyTorch limitation: with nothing
    upstream requiring grad, MPS selects a fused attention kernel that raises
    NotImplementedError on dropout.
    """
    out = []
    for module in model.modules():
        params = list(module.parameters(recurse=True))
        if params and all(not p.requires_grad for p in params):
            out.append(module)
    return out


def build_param_groups(
    model: nn.Module, lr: float, weight_decay: float, layer_decay: float
) -> list[dict]:
    n_blocks = len(model.encoder.blocks)
    groups: dict[tuple[int, bool], dict] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # No weight decay on biases or any 1-D (norm / embedding) parameter.
        decay = not (param.ndim <= 1 or name.endswith(".bias"))
        layer = _layer_id(name, n_blocks)
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


def run_finetune(
    prepared: dict, y: dict, args: argparse.Namespace, device: str
) -> tuple[np.ndarray, np.ndarray, dict]:
    """End-to-end fine-tuning. Checkpoint selected on the selection subset."""
    encoder, report = load_labram(
        device=device, checkpoint=args.checkpoint, pretrained=not args.random_init
    )
    n_classes = int(len(np.unique(y["fit"])))
    model = LaBraMClassifier(encoder, embed_dim=200, n_classes=n_classes).to(device)

    patch = LABRAM_SPEC.patch_samples
    n_patches = prepared["fit"].n_patches
    input_chans = channel_indices(prepared["fit"].ch_names)
    inputs = {k: _as_model_input(p.X, n_patches, patch, device) for k, p in prepared.items()}
    targets = {k: torch.from_numpy(v).to(device) for k, v in y.items()}

    n_frozen = freeze_lower(model, int(getattr(args, "freeze_blocks", 0)))
    still = frozen_modules(model)
    groups = build_param_groups(model, args.lr, args.weight_decay, args.layer_decay)
    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    # Head-only warmup (LP-FT): a randomly initialised head produces large
    # gradients that flow straight into pretrained weights on the first steps.
    # Holding the encoder still until the head is sane is the standard fix.
    top_layer = max(g["layer_id"] for g in groups)
    is_head = [g["layer_id"] == top_layer for g in optimizer.param_groups]
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    n_fit = inputs["fit"].shape[0]
    batch = min(args.finetune_batch_size, n_fit)
    steps_per_epoch = max(1, int(np.ceil(n_fit / batch)))
    total_steps = steps_per_epoch * max(args.epochs, 1)
    warmup_steps = min(int(args.warmup_frac * total_steps), max(total_steps - 1, 0))

    def lr_scale(step: int) -> float:
        """Linear warmup then cosine decay, per optimizer step.

        Stepping per batch rather than per epoch matters here: a subject has only
        a few hundred trials, so an epoch is a handful of updates and an
        epoch-wise schedule would barely move.
        """
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))

    head_warmup_epochs = int(getattr(args, "head_warmup_epochs", 0))
    best_state, best_brier, best_epoch = None, np.inf, -1
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        for module in still:  # frozen blocks stay deterministic
            module.eval()
        encoder_frozen = epoch < head_warmup_epochs
        order = torch.randperm(n_fit, device=device)
        for start in range(0, n_fit, batch):
            idx = order[start : start + batch]
            scale = lr_scale(global_step)
            for group, base, head in zip(optimizer.param_groups, base_lrs, is_head):
                group["lr"] = 0.0 if (encoder_frozen and not head) else base * scale
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs["fit"][idx], input_chans), targets["fit"][idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            global_step += 1

        # Checkpoint selection on the selection subset -- never on held-out data.
        model.eval()
        with torch.no_grad():
            logits = model(inputs["selection"], input_chans).cpu().numpy()
        brier = metric_bundle(y["selection"], proba_from_logits(logits, 1.0))["Brier"]
        if brier < best_brier:
            best_brier, best_epoch = brier, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if args.log_epochs and (epoch + 1) % args.log_epochs == 0:
            print(f"    epoch {epoch + 1}/{args.epochs} selection Brier {brier:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        selection_logits = model(inputs["selection"], input_chans).cpu().numpy()
        heldout_logits = model(inputs["heldout"], input_chans).cpu().numpy()

    audit = {
        "adaptation": "finetune_end_to_end",
        "selected_epoch": int(best_epoch),
        "selection_Brier_at_selected_epoch": float(best_brier),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "layer_decay": float(args.layer_decay),
        "weight_decay": float(args.weight_decay),
        "label_smoothing": float(args.label_smoothing),
        "finetune_batch_size": int(batch),
        "steps_per_epoch": int(steps_per_epoch),
        "total_steps": int(total_steps),
        "warmup_steps": int(warmup_steps),
        "freeze_blocks": int(getattr(args, "freeze_blocks", 0)),
        "n_frozen_tensors": int(n_frozen),
        "n_frozen_modules_in_eval": len(still),
        "head_warmup_epochs": head_warmup_epochs,
        "n_trainable_parameters": int(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        ),
        "checkpoint": report["checkpoint"],
        "n_parameters": report["n_parameters"],
    }
    return selection_logits, heldout_logits, audit


def write_probability_npz(
    *,
    out_dir: Path,
    stem: str,
    dataset: str,
    subject: int,
    payload: dict[str, np.ndarray],
) -> Path:
    """Persist per-trial probabilities in the schema make_risk_coverage_results.py reads.

    Selective prediction needs the confidence of every trial, which the subject
    metrics CSV aggregates away. Keys are `{experiment_key}__val_proba` and
    `__eval_proba`; the reader refuses any file whose metadata does not prove
    eval_labels_used_for_selection=false, so that field is written explicitly
    rather than left implicit.
    """
    path = out_dir / f"{stem}_s{subject}_probabilities.npz"
    metadata = {
        "dataset": dataset,
        "subject": subject,
        "selection_split": SELECTION_SPLIT,
        "heldout_split": HELDOUT_SPLIT,
        "validation_discipline": {"eval_labels_used_for_selection": False},
    }
    np.savez_compressed(path, metadata_json=json.dumps(metadata), **payload)
    return path


def resume_completed_subjects(
    partial_csv: Path, fingerprint_path: Path, fingerprint: dict[str, Any]
) -> list[dict[str, Any]]:
    """Rows already on disk from an interrupted run of this exact configuration.

    Returns [] whenever resuming would be unsafe -- no partial file, no
    fingerprint, or a fingerprint that does not match the current arguments --
    in which case the caller starts fresh. Refusing to resume across differing
    arguments is the whole point: silently appending subjects run under one
    recipe to subjects run under another would fabricate a configuration that
    was never executed.
    """
    if not partial_csv.exists() or not fingerprint_path.exists():
        return []
    try:
        stored = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if stored != fingerprint:
        print("  resume declined: run arguments differ from the interrupted run")
        return []
    with partial_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if rows:
        done = sorted({int(r["subject"]) for r in rows})
        print(f"  resuming: subjects {done} already complete")
    return rows


def run_subject(
    subject: int, args: argparse.Namespace, device: str
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
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
        spec = LABRAM_SPEC
        keep = n_patches * spec.patch_samples

        prepared = {}
        for name, X in (
            ("fit", np.asarray(train["X"])[fit_idx]),
            ("selection", np.asarray(train["X"])[sel_idx]),
            ("heldout", np.asarray(heldout["X"])),
        ):
            p = prepare_for_fm(
                X, train["ch_names"], train["sfreq"], spec,
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
        sel_logits, held_logits, audit = runner(prepared, y, args, device)

        # Temperature fit on the selection subset only.
        temperature, at_clamp = fit_temperature_from_logits(
            sel_logits, y["selection"], args.temperature_max
        )
        sel_proba = proba_from_logits(sel_logits, temperature)
        held_proba = proba_from_logits(held_logits, temperature)
        val_metrics = metric_bundle(y["selection"], sel_proba)
        held_metrics = metric_bundle(y["heldout"], held_proba)
        uncal = metric_bundle(y["heldout"], proba_from_logits(held_logits, 1.0))

        key = f"labram_{args.mode}"
        if args.random_init:
            key += "_randinit"
        if len(args.patches) > 1:
            key += f"_p{n_patches}"
        if getattr(args, "tag", None):
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
                "experiment": f"labram_base_{args.mode}",
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
    ap.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    ap.add_argument("--dataset", default="bci4_2a")
    ap.add_argument("--mode", choices=["frozen", "finetune"], default="frozen")
    ap.add_argument("--patches", type=int, nargs="+", default=[4], help="Window length in 1 s patches.")
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
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--temperature-max", type=float, default=20.0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--layer-decay", type=float, default=0.9, help="Reference default.")
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument(
        "--finetune-batch-size",
        type=int,
        default=16,
        help="Smaller than the reference 64: a subject has only a few hundred trials, "
        "so batch 64 yields too few optimizer steps to fine-tune a transformer.",
    )
    ap.add_argument("--warmup-frac", type=float, default=0.1)
    ap.add_argument("--clip-grad", type=float, default=1.0)
    ap.add_argument("--deterministic", dest="deterministic", action="store_true", default=True)
    ap.add_argument(
        "--no-deterministic",
        dest="deterministic",
        action="store_false",
        help="Required on MPS, which has no deterministic index_put kernel. The run's "
        "platform fingerprint records that determinism was off, so such results are "
        "self-identifying and must not be mixed with deterministic ones.",
    )
    ap.add_argument("--log-epochs", type=int, default=0)
    ap.add_argument(
        "--freeze-blocks",
        type=int,
        default=0,
        help="Freeze embeddings and the lowest N transformer blocks (partial fine-tuning).",
    )
    ap.add_argument(
        "--head-warmup-epochs",
        type=int,
        default=0,
        help="Train the classifier head alone for N epochs before unfreezing the encoder (LP-FT).",
    )
    ap.add_argument(
        "--random-init",
        action="store_true",
        help="Identical architecture, random weights. The control that separates "
        "'pretraining does not transfer' from 'this architecture cannot learn "
        "from a few hundred trials'.",
    )
    ap.add_argument(
        "--tag",
        default=None,
        help="Suffix for experiment_key and output filenames, so sweep configurations "
        "are written side by side instead of overwriting each other.",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip subjects already completed by an interrupted run of the same "
        "configuration. Resumption is refused if the recorded arguments differ.",
    )
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    configure_determinism(bool(args.deterministic))
    print(
        f"device={device} mode={args.mode} patches={args.patches} "
        f"deterministic={bool(args.deterministic)}"
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"fm_probe_{args.mode}"
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
                "finetune_batch_size": args.finetune_batch_size,
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
