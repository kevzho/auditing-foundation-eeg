"""Preprocess BCI Competition IV 2a GDF files into epoch arrays."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import mne
import numpy as np

from eeg_montage import BCI4_2A_CHANNELS, BCI4_2A_SFREQ, verify_bci4_2a_order

from config import (
    DATA_DIR,
    EPOCH_DIR,
    FILTER_HIGH,
    FILTER_LOW,
    ICA_N_COMPONENTS,
    RANDOM_SEED,
    SUBJECTS_TEST,
    SUBJECTS_TRAIN,
    TMAX,
    TMIN,
)

EVENT_ID = {
    "Left Hand": 769,
    "Right Hand": 770,
    "Feet": 771,
    "Tongue": 772,
}
EVAL_CUE = 783


def _normalize_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1).astype(int)
    unique = set(labels.tolist())
    if unique.issubset({0, 1, 2, 3}):
        return labels + 769
    if unique.issubset({1, 2, 3, 4}):
        return labels + 768
    if unique.issubset(set(EVENT_ID.values())):
        return labels
    raise ValueError(f"Unsupported label values: {sorted(unique)}")


def _load_eval_labels(subject: str, labels_dir: Path) -> np.ndarray:
    stems = (
        subject,
        f"{subject}_labels",
        f"{subject}_true_labels",
        f"true_labels_{subject}",
    )
    candidates = []
    search_dirs = (labels_dir, labels_dir / "labels", labels_dir / "true_labels")
    for base in search_dirs:
        for stem in stems:
            candidates.extend(base / f"{stem}{suffix}" for suffix in (".npy", ".csv", ".txt", ".mat"))

    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".npy":
            return _normalize_labels(np.load(path))
        if path.suffix in {".csv", ".txt"}:
            return _normalize_labels(np.loadtxt(path, delimiter="," if path.suffix == ".csv" else None))
        if path.suffix == ".mat":
            from scipy.io import loadmat

            mat = loadmat(path)
            vectors = [
                np.asarray(v).reshape(-1)
                for k, v in mat.items()
                if not k.startswith("__") and np.asarray(v).size >= 1
            ]
            if not vectors:
                raise ValueError(f"No label-like arrays found in {path}")
            vectors.sort(key=lambda v: (-v.size, str(v.dtype)))
            return _normalize_labels(vectors[0])

    raise FileNotFoundError(
        f"{subject}.gdf contains only unknown evaluation cues ({EVAL_CUE}), so true labels are required.\n"
        f"Place labels in {labels_dir}, {labels_dir / 'labels'}, or {labels_dir / 'true_labels'} as one of: "
        f"{', '.join(p.name for p in candidates[:4])} ...\n"
        "Accepted label values are 1-4, 0-3, or event codes 769-772."
    )


def _mark_eog(raw: mne.io.BaseRaw) -> None:
    eog = [ch for ch in raw.ch_names if "EOG" in ch.upper()]
    if not eog and raw.info["nchan"] >= 25:
        eog = raw.ch_names[-3:]
    if eog:
        raw.set_channel_types({ch: "eog" for ch in eog})


def preprocess_subject(
    subject: str,
    data_dir: Path = DATA_DIR,
    out_dir: Path = EPOCH_DIR,
    labels_dir: Path = DATA_DIR,
    npz_dir: Path | None = None,
) -> None:
    """Create NPY epoch arrays, and optionally the workflow NPZ file."""

    gdf_path = data_dir / f"{subject}.gdf"
    if not gdf_path.exists():
        raise FileNotFoundError(f"Missing raw file: {gdf_path}")

    print(f"[preprocess] {subject}")
    raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose=False)
    _mark_eog(raw)
    raw.filter(
        FILTER_LOW,
        FILTER_HIGH,
        method="iir",
        iir_params={"order": 5, "ftype": "butter"},
        verbose=False,
    )

    ica = mne.preprocessing.ICA(
        n_components=ICA_N_COMPONENTS,
        random_state=RANDOM_SEED,
        max_iter="auto",
        verbose=False,
    )
    ica.fit(raw, verbose=False)
    eog_picks = mne.pick_types(raw.info, eog=True)
    if len(eog_picks) > 0:
        eog_name = raw.ch_names[int(eog_picks[0])]
        bads, _ = ica.find_bads_eog(raw, ch_name=eog_name, verbose=False)
        ica.exclude = list(bads)
    ica.apply(raw, verbose=False)

    # Confirm the GDF's EEG order matches the canonical positional mapping before
    # any channel identity is attached to the arrays. Raises on mismatch.
    verify_bci4_2a_order(raw.ch_names)

    raw.pick("eeg")
    events, annotation_ids = mne.events_from_annotations(raw, verbose=False)

    if subject.endswith("E"):
        if str(EVAL_CUE) not in annotation_ids:
            raise KeyError(f"{subject} is missing expected evaluation cue {EVAL_CUE}")
        labels = _load_eval_labels(subject, labels_dir=labels_dir)
        cue_code = annotation_ids[str(EVAL_CUE)]
        cue_events = events[events[:, 2] == cue_code]
        epoch_event_id = {"Unknown": cue_code}
        if labels.shape[0] != cue_events.shape[0]:
            raise ValueError(
                f"{subject}: labels length ({labels.shape[0]}) does not match evaluation cues ({cue_events.shape[0]})"
            )
    else:
        missing = [str(code) for code in EVENT_ID.values() if str(code) not in annotation_ids]
        if missing:
            raise KeyError(f"{subject} is missing class annotations: {missing}")
        epoch_event_id = {name: annotation_ids[str(code)] for name, code in EVENT_ID.items()}
        cue_events = events[np.isin(events[:, 2], list(epoch_event_id.values()))]
        label_by_event = {annotation_ids[str(code)]: code for code in EVENT_ID.values()}
        labels = np.array([label_by_event[int(event_code)] for event_code in cue_events[:, 2]], dtype=int)

    epochs = mne.Epochs(
        raw,
        cue_events,
        epoch_event_id,
        tmin=TMIN,
        tmax=TMAX,
        baseline=None,
        preload=True,
        reject={"eeg": 100e-6},
        event_repeated="drop",
        verbose=False,
    )

    X = epochs.get_data(copy=True).astype(np.float32, copy=False)
    y = labels[epochs.selection].astype(np.int16, copy=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"X_{subject}.npy", X)
    np.save(out_dir / f"y_{subject}.npy", y)

    if npz_dir is not None:
        _save_workflow_npz(subject, X, y, npz_dir, sfreq=float(epochs.info["sfreq"]))


def _save_workflow_npz(
    subject: str,
    X: np.ndarray,
    y: np.ndarray,
    npz_dir: Path,
    sfreq: float = BCI4_2A_SFREQ,
) -> Path:
    if len(subject) != 4 or not subject.startswith("A") or subject[-1] not in {"T", "E"}:
        raise ValueError(f"Expected a BCI IV-2a subject like A01T or A01E, got {subject!r}")
    subject_id = int(subject[1:3])
    split = "train" if subject.endswith("T") else "eval"
    path = npz_dir / f"bci4_2a_subject{subject_id}_{split}.npz"
    npz_dir.mkdir(parents=True, exist_ok=True)
    # ch_names and sfreq are part of the input contract for channel-embedding
    # foundation models; X and y stay first so existing readers are unaffected.
    np.savez_compressed(
        path,
        X=X,
        y=y,
        ch_names=np.asarray(BCI4_2A_CHANNELS, dtype="<U8"),
        sfreq=np.asarray(float(sfreq), dtype=np.float64),
    )
    return path


def preprocess_many(
    subjects: Iterable[str],
    data_dir: Path = DATA_DIR,
    out_dir: Path = EPOCH_DIR,
    labels_dir: Path = DATA_DIR,
    npz_dir: Path | None = None,
) -> None:
    for subject in subjects:
        preprocess_subject(
            subject,
            data_dir=data_dir,
            out_dir=out_dir,
            labels_dir=labels_dir,
            npz_dir=npz_dir,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess BCI IV 2a train and evaluation sessions.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=EPOCH_DIR)
    parser.add_argument("--labels-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--npz-dir",
        type=Path,
        default=None,
        help="Also write bci4_2a_subject{n}_{train,eval}.npz files for the main workflow.",
    )
    parser.add_argument("--sessions", choices=("all", "T", "E"), default="all")
    args = parser.parse_args()

    if args.sessions == "T":
        subjects = SUBJECTS_TRAIN
    elif args.sessions == "E":
        subjects = SUBJECTS_TEST
    else:
        subjects = (*SUBJECTS_TRAIN, *SUBJECTS_TEST)

    for subject in subjects:
        if subject.endswith("E"):
            _load_eval_labels(subject, labels_dir=args.labels_dir)
    preprocess_many(subjects, data_dir=args.data_dir, out_dir=args.out_dir, labels_dir=args.labels_dir, npz_dir=args.npz_dir)


if __name__ == "__main__":
    main()
