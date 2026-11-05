#!/usr/bin/env python3
"""Leave-one-subject-out frozen probing of EEG foundation models.

Each fold fits on the other subjects' training sessions and evaluates on the
held-out subject's evaluation session. Probe regularization and temperature
are selected using training subjects only. Random-init runs use the same
architecture and data.

Usage::

    python src/experiments/run_fm_loso.py --model cbramod
    python src/experiments/run_fm_loso.py --model labram --random-init
"""

from __future__ import annotations

import argparse
import atexit
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

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
    load_fm_split,
    probe_logits,
    resume_completed_subjects,
)
from models.fm_adapters import LABRAM_SPEC, prepare_for_fm  # noqa: E402

POOLINGS = ("mean", "flatten")


def _prepare(X, ch_names, sfreq, spec, band, keep):
    p = prepare_for_fm(X, ch_names, sfreq, spec, source_bandpass=band)
    if p.X.shape[-1] < keep:
        raise SystemExit(f"need {keep} samples, have {p.X.shape[-1]}")
    p.X = p.X[..., :keep]
    return p


def extract_all(args, spec, spec_bd, device: str, keep: int, cache_dir: Path | None = None) -> dict[int, dict]:
    """Frozen features for every subject, computed once and reused across folds.

    With ``cache_dir``, feature arrays are stored as read-only memmaps. This
    keeps the 54-subject, 62-channel flatten pooling within memory limits.
    """
    cache: dict[int, dict] = {}
    if args.model == "labram":
        from models.labram_probe import extract_features, load_labram

        model, report = load_labram(device=device, pretrained=not args.random_init)
    else:
        from models.braindecode_fms import build_model, extract_features as bd_extract

        model = report = None  # built lazily: it needs n_times and n_chans

    for subject in args.subjects:
        train, provenance = load_fm_split(args.fm_dir, subject, "train", args.dataset)
        heldout, _ = load_fm_split(args.fm_dir, subject, "eval", args.dataset)
        band = tuple(provenance.get("bandpass_hz", (0.1, 75.0)))

        fit_idx, sel_idx = _split_train_val_indices(np.asarray(train["y"]), args.seed)
        parts = {
            "fit": np.asarray(train["X"])[fit_idx],
            "selection": np.asarray(train["X"])[sel_idx],
            "heldout": np.asarray(heldout["X"]),
        }
        prepared = {
            k: _prepare(v, train["ch_names"], train["sfreq"], spec, band, keep)
            for k, v in parts.items()
        }

        if args.model == "labram":
            feats = {
                k: {
                    "mean": extract_features(
                        model, p.X, p.ch_names, device=device, batch_size=args.batch_size
                    ).features
                }
                for k, p in prepared.items()
            }
        else:
            if model is None:
                model, report = build_model(
                    spec_bd,
                    prepared["fit"].ch_names,
                    prepared["fit"].X.shape[-1],
                    2,  # head is unused for trunk features
                    device=device,
                    pretrained=not args.random_init,
                )
            feats = {
                k: {
                    pool_name: bd_extract(
                        model, spec_bd, p.X, pooling=pool_name,
                        device=device, batch_size=args.batch_size,
                    )
                    for pool_name in POOLINGS
                }
                for k, p in prepared.items()
            }

        if cache_dir is not None:
            spilled: dict[str, dict[str, np.ndarray]] = {}
            for split, pools in feats.items():
                spilled[split] = {}
                for pool_name, arr in pools.items():
                    path = cache_dir / f"s{subject}_{split}_{pool_name}.npy"
                    np.save(path, np.ascontiguousarray(arr))
                    spilled[split][pool_name] = np.load(path, mmap_mode="r")
            feats = spilled

        cache[subject] = {
            "features": feats,
            "y_raw": {
                "fit": np.asarray(train["y"])[fit_idx],
                "selection": np.asarray(train["y"])[sel_idx],
                "heldout": np.asarray(heldout["y"]),
            },
        }
        print(f"  extracted subject {subject}")
    cache["_report"] = report
    return cache


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="cbramod", choices=["cbramod", "labram"])
    ap.add_argument("--dataset", default="bci4_2a")
    ap.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    ap.add_argument("--patches", type=int, default=4)
    ap.add_argument("--fm-dir", type=Path, default=Path("data") / "fm")
    ap.add_argument("--out-dir", type=Path, default=Path("results") / "fm_probe")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature-max", type=float, default=20.0)
    ap.add_argument("--random-init", action="store_true")
    ap.add_argument("--tag", default=None)
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip folds already completed by an interrupted run of the same "
        "configuration. Resumption is refused if the recorded arguments differ.",
    )
    ap.add_argument("--deterministic", dest="deterministic", action="store_true", default=True)
    ap.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    configure_determinism(bool(args.deterministic))
    set_all_seeds(args.seed)

    if args.model == "labram":
        spec = LABRAM_SPEC
        spec_bd = None
        poolings = ("mean",)
    else:
        from models.braindecode_fms import as_fm_spec, get_spec

        spec_bd = get_spec(args.model)
        spec = as_fm_spec(spec_bd)
        poolings = POOLINGS

    keep = args.patches * spec.patch_samples
    print(f"device={device} model={args.model} dataset={args.dataset} loso patches={args.patches}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"fm_loso_{args.model}"
    if args.dataset != "bci4_2a":
        stem += f"_{args.dataset}"
    if args.random_init:
        stem += "_randinit"
    if args.tag:
        stem += f"_{args.tag}"
    out_csv = args.out_dir / f"{stem}_subject_metrics.csv"
    partial_csv = args.out_dir / f"{stem}_subject_metrics.csv.partial"
    fingerprint_path = args.out_dir / f"{stem}_run.json.partial"

    # A fold's training pool depends on the complete subject list.
    fingerprint = {
        k: v
        for k, v in vars(args).items()
        if k not in {"out_dir", "fm_dir", "checkpoint", "device", "resume"}
    }
    fingerprint = json.loads(json.dumps(fingerprint, default=str))

    rows: list[dict[str, Any]] = (
        resume_completed_subjects(partial_csv, fingerprint_path, fingerprint)
        if args.resume
        else []
    )
    done = {int(r["subject"]) for r in rows}
    if done:
        print(f"  resuming: {len(done)} fold(s) already on disk, {len(args.subjects) - len(done)} to go")
    fingerprint_path.write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")

    # Use a stable path so the next run can clear files left by SIGKILL.
    cache_dir = args.out_dir / f".{stem}_features"
    if cache_dir.exists():
        print(f"  clearing stale feature cache from an interrupted run: {cache_dir}")
        shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.mkdir(parents=True)
    atexit.register(shutil.rmtree, cache_dir, True)

    cache = extract_all(args, spec, spec_bd, device, keep, cache_dir=cache_dir)
    report = cache.pop("_report")

    handle = partial_csv.open("a" if rows else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys())) if rows else None
    for held_subject in args.subjects:
        if held_subject in done:
            continue
        pool_subjects = [s for s in args.subjects if s != held_subject]

        # Labels share one encoding across the pool and the held-out subject.
        all_raw = np.concatenate(
            [cache[s]["y_raw"]["fit"] for s in pool_subjects]
            + [cache[held_subject]["y_raw"]["heldout"]]
        )
        classes = np.unique(all_raw)
        lookup = {int(c): i for i, c in enumerate(classes)}
        enc = lambda arr: np.array([lookup[int(v)] for v in arr], dtype=np.int64)  # noqa: E731

        y_fit = np.concatenate([enc(cache[s]["y_raw"]["fit"]) for s in pool_subjects])
        y_sel = np.concatenate([enc(cache[s]["y_raw"]["selection"]) for s in pool_subjects])
        y_held = enc(cache[held_subject]["y_raw"]["heldout"])

        # Materialize one pooling at a time to bound peak memory.
        best = {"brier": np.inf, "pooling": None, "C": None}
        for pooling in poolings:
            stack = {
                "fit": np.concatenate(
                    [cache[s]["features"]["fit"][pooling] for s in pool_subjects]
                ),
                "selection": np.concatenate(
                    [cache[s]["features"]["selection"][pooling] for s in pool_subjects]
                ),
            }
            for c in C_GRID:
                pipe = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(C=c, max_iter=2000, random_state=args.seed),
                ).fit(stack["fit"], y_fit)
                brier = metric_bundle(y_sel, pipe.predict_proba(stack["selection"]))["Brier"]
                if brier < best["brier"]:
                    best = {"brier": brier, "pooling": pooling, "C": c}
            del stack, pipe

        # Rebuild only the selected pooling after the search.
        feats = {
            "fit": np.concatenate(
                [cache[s]["features"]["fit"][best["pooling"]] for s in pool_subjects]
            ),
            "selection": np.concatenate(
                [cache[s]["features"]["selection"][best["pooling"]] for s in pool_subjects]
            ),
            "heldout": np.asarray(cache[held_subject]["features"]["heldout"][best["pooling"]]),
        }
        probe = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=best["C"], max_iter=2000, random_state=args.seed),
        ).fit(feats["fit"], y_fit)
        sel_logits = probe_logits(feats["selection"], probe)
        held_logits = probe_logits(feats["heldout"], probe)

        temperature, at_clamp = fit_temperature_from_logits(sel_logits, y_sel, args.temperature_max)
        val_metrics = metric_bundle(y_sel, proba_from_logits(sel_logits, temperature))
        held_metrics = metric_bundle(y_held, proba_from_logits(held_logits, temperature))
        uncal = metric_bundle(y_held, proba_from_logits(held_logits, 1.0))

        key = f"{args.model}_loso" + ("_randinit" if args.random_init else "")
        if args.tag:
            key += f"_{args.tag}"
        audit = {
            "adaptation": "frozen_linear_probe_loso",
            "pool_subjects": pool_subjects,
            "n_fit_trials": int(len(y_fit)),
            "n_selection_trials": int(len(y_sel)),
            "selected_C": float(best["C"]),
            "selected_pooling": best["pooling"],
            "selection_Brier_at_selected_C": float(best["brier"]),
            **(report or {}),
        }
        rows.append(
            {
                "dataset": args.dataset,
                "dataset_label": DATASET_LABELS.get(args.dataset, args.dataset),
                "subject": held_subject,
                "n_classes": int(len(classes)),
                "experiment_key": key,
                "experiment": f"{args.model}_frozen_loso",
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
                "n_patches": int(args.patches),
                "window_seconds": float(keep / spec.sfreq),
                "selection_split": "pooled_training_subjects_validation",
                "eval_labels_used_for_selection": False,
                "audit_json": json.dumps(audit),
            }
        )
        if writer is None:
            writer = csv.DictWriter(handle, fieldnames=list(rows[-1].keys()))
            writer.writeheader()
        writer.writerow(rows[-1])
        handle.flush()

        del feats, probe
        print(
            f"  held-out s{held_subject}: acc {held_metrics['accuracy']:.4f} "
            f"Brier {held_metrics['Brier']:.4f} (pool n={len(y_fit)}) T={temperature:.3f}"
        )

    handle.close()
    # Publish the CSV only after all folds finish.
    partial_csv.replace(out_csv)
    fingerprint_path.unlink(missing_ok=True)
    (args.out_dir / f"{stem}_run.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "dataset": args.dataset,
                "mode": "frozen_loso",
                "random_init": bool(args.random_init),
                "patches": args.patches,
                "subjects": args.subjects,
                "seed": args.seed,
                "platform": platform_fingerprint(device),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Resumed rows come from CSV and contain strings.
    accs = [float(r["heldout_accuracy"]) for r in rows]
    briers = [float(r["heldout_Brier"]) for r in rows]
    print(f"\nmean held-out accuracy {np.mean(accs):.4f}  Brier {np.mean(briers):.4f}  (n={len(rows)})")
    shutil.rmtree(cache_dir, ignore_errors=True)
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
