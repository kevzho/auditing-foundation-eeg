#!/usr/bin/env python3
"""Leave-one-subject-out frozen probing of EEG foundation models.

Answers the strongest objection to the per-subject results: *foundation models
are meant for the pooled regime, so of course they fail on ~230 calibration
trials.* Here each fold pools eight subjects' training sessions -- roughly eight
times the data -- and evaluates on the ninth subject's held-out session. If
pretraining's value only appears with more data, this is where it shows up.

Selection discipline is unchanged and is the reason this is not simply
"train on 8, test on 1": probe regularisation and the scalar temperature are
chosen on a selection split drawn from the *pooled training subjects*, never
from the held-out subject. The held-out subject's evaluation session is read
once, for the reported row.

The random-init control runs the identical architecture with random weights on
the identical pooled data, so the pooled-regime comparison isolates what
pretraining contributes rather than what the architecture contributes.

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
    encode_labels,
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


def extract_all(args, spec, device: str, keep: int, cache_dir: Path | None = None) -> dict[int, dict]:
    """Frozen features for every subject, computed once and reused across folds.

    Nine folds each need eight subjects' features; recomputing per fold would do
    the same forward passes eight times over.

    When ``cache_dir`` is given, each subject's features are written there and
    returned as read-only memmaps instead of resident arrays. The arrays are
    bit-identical either way -- this changes where the bytes live, not what
    they are.

    Why it is not optional at 54 subjects. CBraMod's ``flatten`` pooling is
    ``patches * channels * d_model`` wide, which for Lee2019_MI's 62-channel
    montage is 49600 floats per trial: 79.7 MB per subject, **4.30 GB** for 54
    of them, on a 17 GB machine. Fold 1 then concatenates 53 subjects into a
    1.7 GB fit matrix, which scikit-learn's lbfgs upcasts to float64 and the
    scaler copies again. The n=54 control died there twice. At nine subjects
    the same cache is 0.7 GB and none of this mattered, which is why the design
    survived this long.

    Dropping ``flatten`` would have been the cheap fix and is not available:
    validation selects it on 54 of 54 folds of the pretrained arm, so a
    control without it would be answering a different question.
    """
    cache: dict[int, dict] = {}
    if args.model == "labram":
        from models.labram_probe import channel_indices, extract_features, load_labram

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

    global spec_bd
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

    # Output paths are resolved before the first fold, not after the last.
    # Deriving them at the end meant nothing could be written until every fold
    # had succeeded: the 54-subject control was killed at fold 25 and all 24
    # completed folds went with it, because they existed only in `rows`.
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

    # Same discipline as run_bd_fm_probe: resumption is refused unless the
    # recorded arguments match, because appending folds computed under one
    # configuration to folds computed under another fabricates a run that never
    # happened. Note the fold identity depends on `subjects` as a whole -- each
    # fold pools every other subject -- so a changed subject list must not
    # resume, and the fingerprint covers it.
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

    # Features spill to a scratch directory beside the results rather than
    # staying resident; see extract_all. Removed on the way out, including on
    # failure, so an interrupted run does not leave gigabytes behind on a
    # volume that has been down to single-digit GB free.
    # The path is deterministic, not a mkdtemp name, and it is cleared on the
    # way *in* as well as on the way out.
    #
    # Cleanup-at-exit is not sufficient for this script and it is worth being
    # explicit about why: `atexit` does not run on SIGKILL, and being killed is
    # this job's characteristic failure -- twice for memory, on a volume that
    # has been down to single-digit GB free. A random temp name plus an exit
    # hook means every kill strands another 4.3 GB that nothing will ever
    # collect, and the third kill is then caused by the first two.
    #
    # Clearing a known path at startup is the only cleanup that survives a
    # kill, because it runs in the next process rather than the dead one. The
    # cost is re-extracting features after an interrupted run; that is ~20
    # minutes, against fold results which `--resume` still preserves.
    cache_dir = args.out_dir / f".{stem}_features"
    if cache_dir.exists():
        print(f"  clearing stale feature cache from an interrupted run: {cache_dir}")
        shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.mkdir(parents=True)
    atexit.register(shutil.rmtree, cache_dir, True)  # best effort for clean exits

    cache = extract_all(args, spec, device, keep, cache_dir=cache_dir)
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

        # One pooling is materialised at a time and dropped before the next is
        # built. Holding every pooling's stack simultaneously doubled the peak
        # for no benefit: only the winner is used after selection, and the
        # winner is rebuilt below. Selection is unchanged -- every (pooling, C)
        # pair is still scored on the same validation split, in the same order.
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

        # The winning pooling is rebuilt rather than retained. One extra
        # concatenate per fold buys a peak that does not scale with the number
        # of poolings, which is the difference between fitting in memory and
        # not at 54 subjects.
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
        # Flushed per fold, so a kill costs one fold rather than the run.
        if writer is None:
            writer = csv.DictWriter(handle, fieldnames=list(rows[-1].keys()))
            writer.writeheader()
        writer.writerow(rows[-1])
        handle.flush()

        # `stacked` holds every pooling's concatenated fit matrix for this
        # fold -- with 62 channels and the "flatten" pooling that is the run's
        # memory peak, and at 54 subjects it is ~6x what the n=9 runs held.
        # Dropping it before the next fold builds its own is what keeps peak
        # usage flat across folds instead of letting it creep until the kernel
        # intervenes, which is how fold 25 died.
        del feats, probe
        print(
            f"  held-out s{held_subject}: acc {held_metrics['accuracy']:.4f} "
            f"Brier {held_metrics['Brier']:.4f} (pool n={len(y_fit)}) T={temperature:.3f}"
        )

    handle.close()
    # Renamed only once every fold is in, so a truncated file can never be
    # mistaken downstream for a finished n=54 result.
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

    # Coerced because resumed folds are read back from CSV as strings while
    # folds computed in this process are floats. Leaving that unresolved made
    # the summary raise *after* a correct CSV had already been written, so a
    # successful run reported itself as a failure -- and under `set -e` would
    # have taken the next arm in the script down with it.
    accs = [float(r["heldout_accuracy"]) for r in rows]
    briers = [float(r["heldout_Brier"]) for r in rows]
    print(f"\nmean held-out accuracy {np.mean(accs):.4f}  Brier {np.mean(briers):.4f}  (n={len(rows)})")
    shutil.rmtree(cache_dir, ignore_errors=True)
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
