#!/usr/bin/env python3
"""Gate A: can a frozen pretrained LaBraM encoder decode motor imagery at all?

This is a feasibility check, not a result. It answers one question -- does the
whole chain (broadband preprocessing -> FM adapter -> pretrained encoder ->
linear probe) run end to end and produce above-chance decoding -- before any
effort goes into the uncertainty audit that depends on it.

Selection discipline is preserved even here: the probe's regularization strength
is chosen on a split carved out of the *training* session, and the held-out
evaluation session is touched once, for the final numbers.

Usage:
    python src/scripts/gate_a_labram_probe.py --subject 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eeg_montage import load_npz_with_channels  # noqa: E402
from models.fm_adapters import LABRAM_SPEC, prepare_for_fm  # noqa: E402
from models.labram_probe import extract_features, load_labram  # noqa: E402


def multiclass_brier(proba: np.ndarray, y: np.ndarray, classes: np.ndarray) -> float:
    onehot = np.zeros_like(proba)
    for i, label in enumerate(y):
        onehot[i, int(np.where(classes == label)[0][0])] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def load_split(fm_dir: Path, subject: int, split: str):
    path = fm_dir / f"bci4_2a_subject{subject}_{split}.npz"
    if not path.exists():
        raise SystemExit(f"missing {path}; run src/preprocess_fm.py first")
    data = load_npz_with_channels(path, dataset="bci4_2a")
    raw = np.load(path, allow_pickle=False)
    provenance = json.loads(str(raw["provenance"])) if "provenance" in raw.files else {}
    return data, provenance


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--fm-dir", type=Path, default=Path("data") / "fm")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train, train_prov = load_split(args.fm_dir, args.subject, "train")
    evaluation, _ = load_split(args.fm_dir, args.subject, "eval")
    band = tuple(train_prov.get("bandpass_hz", (0.1, 75.0)))
    print(f"subject {args.subject}: train {train['X'].shape}, eval {evaluation['X'].shape}, band {band}")

    prepared = {
        name: prepare_for_fm(
            split["X"], split["ch_names"], split["sfreq"], LABRAM_SPEC, source_bandpass=band
        )
        for name, split in (("train", train), ("eval", evaluation))
    }
    print(
        f"adapted -> {prepared['train'].X.shape} @ {LABRAM_SPEC.sfreq:g} Hz, "
        f"{prepared['train'].n_patches} patches"
    )

    model, report = load_labram(device=args.device)
    print(f"loaded {report['n_parameters'] / 1e6:.1f}M params from {Path(report['checkpoint']).name}")

    feats = {
        name: extract_features(
            model, p.X, p.ch_names, device=args.device, batch_size=args.batch_size
        )
        for name, p in prepared.items()
    }
    print(f"features: train {feats['train'].features.shape}, eval {feats['eval'].features.shape}")

    y_train = np.asarray(train["y"]).astype(int)
    y_eval = np.asarray(evaluation["y"]).astype(int)

    # Regularization chosen on the training session only.
    best_c, best_score = None, -np.inf
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    for c in (0.001, 0.01, 0.1, 1.0, 10.0):
        pipe = make_pipeline(
            StandardScaler(), LogisticRegression(C=c, max_iter=2000, random_state=args.seed)
        )
        score = float(
            np.mean(cross_val_score(pipe, feats["train"].features, y_train, cv=cv, scoring="accuracy"))
        )
        print(f"  C={c:<7g} train-session CV accuracy {score:.4f}")
        if score > best_score:
            best_c, best_score = c, score

    probe = make_pipeline(
        StandardScaler(), LogisticRegression(C=best_c, max_iter=2000, random_state=args.seed)
    ).fit(feats["train"].features, y_train)

    # Held-out session, touched once.
    proba = probe.predict_proba(feats["eval"].features)
    classes = probe.named_steps["logisticregression"].classes_
    accuracy = float(np.mean(probe.predict(feats["eval"].features) == y_eval))
    brier = multiclass_brier(proba, y_eval, classes)
    n_classes = len(classes)

    print()
    print(f"selected C={best_c} (train-session CV {best_score:.4f})")
    print(f"HELD-OUT accuracy {accuracy:.4f}   (chance {1.0 / n_classes:.4f})")
    print(f"HELD-OUT Brier    {brier:.4f}   (chance {(n_classes - 1) / n_classes:.4f})")
    verdict = "PASS - above chance" if accuracy > 1.0 / n_classes else "FAIL - at or below chance"
    print(f"GATE A: {verdict}")


if __name__ == "__main__":
    main()
