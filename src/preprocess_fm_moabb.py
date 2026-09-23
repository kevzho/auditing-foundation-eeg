#!/usr/bin/env python3
"""Broadband preprocessing of MOABB datasets for the foundation-model arm.

Companion to ``preprocess_fm.py``, which handles the local BCI IV-2a GDF files.
The reasoning is the same and worth restating: the supervised pipeline
band-passes to 8--30 Hz, which is correct for CSP and small convnets and wrong
for models pretrained on ~0.1--75 Hz broadband EEG. Narrow-band input would put
every trial outside the pretraining distribution, so "the foundation model
fails" could not be distinguished from "we filtered away what it was trained
on".

    band-pass 0.1--75 Hz  ->  notch 50 Hz  ->  resample 200 Hz

Session split follows the dataset's own labelling: sessions whose name contains
``train`` form the calibration session, those containing ``test`` the held-out
one. This is the same rule ``run_calibration_workflow.moabb_protocol_masks``
applies, so the foundation-model arm and the supervised arm are split
identically and their numbers are comparable.

Outputs ``data/fm/{dataset}_subject{N}_{train,eval}.npz`` with ``X``, ``y``,
``ch_names``, ``sfreq`` and a ``provenance`` JSON blob -- the schema the probe
runners already read.

Usage::

    python src/preprocess_fm_moabb.py --dataset BNCI2014_004
    python src/preprocess_fm_moabb.py --dataset BNCI2014_004 --subjects 1 2 3
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

FM_SFREQ = 200.0
FM_BANDPASS_HZ = (0.1, 75.0)
FM_NOTCH_HZ = 50.0
FM_NPZ_DIR = Path("data") / "fm"

#: Named band presets, so one pass over a downloaded subject can write every
#: band a study needs.
#:
#: This matters for datasets large enough to require purging the source files
#: after processing: Lee2019_MI is ~1.2 GB per subject and 54 subjects will not
#: fit on disk at once, so the raw files are deleted as we go. Writing only one
#: band per pass would make a second band cost a complete ~59 GB re-download.
BAND_PRESETS = {
    "broadband": {"bandpass": (0.1, 75.0), "sfreq": 200.0, "npz_dir": Path("data") / "fm"},
    "narrowband": {"bandpass": (8.0, 30.0), "sfreq": 250.0, "npz_dir": Path("data") / "narrowband"},
}

#: Number of classes per dataset. MOABB's LeftRightImagery paradigm is the right
#: one for 2-class sets; MotorImagery with n_classes for the rest.
DATASET_CLASSES = {
    "BNCI2014_001": 4,
    "BNCI2014_004": 2,
    "Zhou2016": 3,
    # 54 subjects, 62 channels, 2 sessions. The large-n, dense-montage set:
    # it breaks the n=9 attainable-p floor and tests whether BNCI2014_004's
    # failed pooling result was really caused by having only three electrodes.
    "Lee2019_MI": 2,
}


#: Constructor arguments for datasets whose defaults hide part of the recording.
#: Lee2019_MI defaults to ``test_run=None`` for MI, which exposes only the
#: training run of session 1 -- a quarter of the data, silently.
DATASET_KWARGS = {
    "Lee2019_MI": dict(train_run=True, test_run=True),
}

#: Datasets where MOABB's session filter silently discards half the recording.
#:
#: ``Lee2019_MI``'s loader names sessions ``str(session - 1)``, emitting "0" and
#: "1", while ``BaseDataset.get_data`` filters those keys against
#: ``_selected_sessions``, which holds the 1-indexed ``(1, 2)``. The intersection
#: keeps only "1", so a caller who asks for nothing unusual receives one session
#: and no warning -- which would silently turn this project's cross-session
#: protocol into a within-session one, the exact evaluation weakness the
#: manuscript criticises. Clearing the filter restores both sessions.
DATASETS_NEEDING_SESSION_UNFILTER = {"Lee2019_MI"}


def dataset_slug(name: str) -> str:
    return str(name).lower()


def _prepare_env(data_dir: Path) -> None:
    """Point MNE and MOABB at the project's data directory before import."""
    resolved = str(Path(data_dir).resolve())
    os.environ.setdefault("MNE_DATA", resolved)
    os.environ.setdefault("MOABB_RESULTS", resolved)


def build_paradigm(dataset_name: str, bandpass=FM_BANDPASS_HZ, sfreq=FM_SFREQ):
    from moabb.paradigms import LeftRightImagery, MotorImagery

    n_classes = DATASET_CLASSES.get(dataset_name, 4)
    common = dict(fmin=float(bandpass[0]), fmax=float(bandpass[1]), resample=float(sfreq))
    if n_classes == 2:
        return LeftRightImagery(**common)
    try:
        return MotorImagery(n_classes=n_classes, **common)
    except TypeError:
        return MotorImagery(**common)


def apply_notch(X: np.ndarray, sfreq: float, notch: float | None) -> np.ndarray:
    """Notch the epoched array.

    MOABB's paradigm applies the band-pass and resampling but no notch, so it is
    done here to keep this dataset's preprocessing identical to the BCI IV-2a
    arm. Only harmonics strictly below Nyquist are used.

    ``spectrum_fit`` rather than the default FIR: these are ~4.5 s epochs, and a
    default 50 Hz FIR notch is longer than the signal, which MNE warns produces
    distortion. The sinusoidal fit method is built for short segments and needs
    no filter length. Applying the notch to continuous data instead would be
    preferable, but MOABB's paradigm epochs internally and exposes no hook.
    """
    if not notch:
        return X
    import mne

    freqs = [f for f in np.arange(notch, sfreq / 2.0, notch)]
    if not freqs:
        return X
    return mne.filter.notch_filter(
        X.astype(np.float64, copy=False),
        Fs=float(sfreq),
        freqs=freqs,
        method="spectrum_fit",
        filter_length="auto",
        verbose=False,
    ).astype(np.float32, copy=False)


def split_sessions(sessions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calibration vs held-out mask, from session labels.

    Two naming conventions occur. The BNCI sets label sessions ``0train`` /
    ``3test``, so the split reads straight off the name. Lee2019_MI labels them
    ``1`` / ``2`` and puts the train/test distinction in the *run* column
    instead -- but for a session-shift study the run distinction is not what we
    want. There the correct split is chronological: the first recording day
    calibrates, the last is held out, which is the protocol used everywhere else
    in this project.
    """
    text = np.array([str(s).lower() for s in sessions])
    train = np.char.find(text, "train") >= 0
    evaluation = np.char.find(text, "test") >= 0
    if train.any() and evaluation.any():
        return train, evaluation

    ordered = sorted(set(text))
    if len(ordered) < 2:
        raise SystemExit(
            f"need at least two sessions to form a held-out split; saw {ordered}"
        )
    first, last = ordered[0], ordered[-1]
    return text == first, text == last


def preprocess_subject_bands(
    dataset_name: str,
    subject: int,
    bands: list[str],
    notch: float | None = FM_NOTCH_HZ,
    purge_raw: bool = False,
) -> list[Path]:
    """Write one subject in every requested band, then optionally purge.

    All bands are produced from the same downloaded files before anything is
    deleted; otherwise a second band costs a second full download.
    """
    written: list[Path] = []
    for band in bands:
        preset = BAND_PRESETS[band]
        written += preprocess_subject(
            dataset_name, subject,
            npz_dir=preset["npz_dir"], notch=notch,
            bandpass=preset["bandpass"], sfreq=preset["sfreq"],
        )
    if purge_raw:
        from moabb import datasets as moabb_datasets

        dataset = getattr(moabb_datasets, dataset_name)(**DATASET_KWARGS.get(dataset_name, {}))
        freed = purge_raw_subject(dataset, subject)
        print(f"[preprocess_fm_moabb] purged {freed / 1e9:.2f} GB of source files for s{subject}")
    return written


def preprocess_subject(
    dataset_name: str,
    subject: int,
    npz_dir: Path = FM_NPZ_DIR,
    notch: float | None = FM_NOTCH_HZ,
    bandpass=FM_BANDPASS_HZ,
    sfreq: float = FM_SFREQ,
    purge_raw: bool = False,
) -> list[Path]:
    from moabb import datasets as moabb_datasets

    dataset = getattr(moabb_datasets, dataset_name)(**DATASET_KWARGS.get(dataset_name, {}))
    if dataset_name in DATASETS_NEEDING_SESSION_UNFILTER:
        dataset._selected_sessions = None
    paradigm = build_paradigm(dataset_name, bandpass, sfreq)
    X, y, meta = paradigm.get_data(dataset=dataset, subjects=[int(subject)])
    X = np.asarray(X, dtype=np.float32)  # volts

    # The paradigm does not return channel names, so read them from the dataset's
    # own info. Channel identity is not optional here: a foundation model with a
    # montage vocabulary indexes its positional embeddings by electrode name.
    raw_map = dataset.get_data([int(subject)])
    first_session = next(iter(raw_map[int(subject)].values()))
    first_run = next(iter(first_session.values()))
    picks = [
        name
        for name, kind in zip(first_run.ch_names, first_run.get_channel_types())
        if kind == "eeg"
    ]
    ch_names = picks
    if len(ch_names) != X.shape[1]:
        raise SystemExit(
            f"{dataset_name} s{subject}: {len(ch_names)} EEG channel names for "
            f"{X.shape[1]} data channels"
        )

    X = apply_notch(X, sfreq, notch)

    sessions = meta["session"].to_numpy()
    train_mask, eval_mask = split_sessions(sessions)

    slug = dataset_slug(dataset_name)
    npz_dir = Path(npz_dir)
    npz_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for split, mask in (("train", train_mask), ("eval", eval_mask)):
        classes, encoded = np.unique(y[mask], return_inverse=True)
        provenance = {
            "pipeline": "preprocess_fm_moabb",
            "dataset": dataset_name,
            "bandpass_hz": [float(bandpass[0]), float(bandpass[1])],
            "notch_hz": float(notch) if notch else None,
            "sfreq_hz": float(sfreq),
            "ica_applied": False,
            "reject_uv": None,
            "units": "V",
            "n_trials": int(mask.sum()),
            "sessions": sorted({str(s) for s in sessions[mask]}),
            "class_names": [str(c) for c in classes],
        }
        path = npz_dir / f"{slug}_subject{int(subject)}_{split}.npz"
        np.savez_compressed(
            path,
            X=X[mask],
            # Integer codes, matching the GDF arm; class_names keeps the mapping.
            y=encoded.astype(np.int16, copy=False),
            ch_names=np.asarray(ch_names, dtype="<U16"),
            sfreq=np.asarray(float(sfreq), dtype=np.float64),
            provenance=np.asarray(json.dumps(provenance)),
        )
        print(
            f"[preprocess_fm_moabb] {dataset_name} s{subject} {split}: "
            f"{X[mask].shape} {provenance['sessions']} -> {path}"
        )
        written.append(path)

    if purge_raw:
        freed = purge_raw_subject(dataset, subject)
        print(f"[preprocess_fm_moabb] purged {freed / 1e9:.2f} GB of source files for s{subject}")
    return written


def purge_raw_subject(dataset, subject: int) -> int:
    """Delete one subject's source files after its npz is written.

    Driven by ``dataset.data_path(subject)``, which is MOABB's own record of
    exactly which files back this subject -- never by globbing a directory,
    which risks deleting another dataset's cache.

    Needed because Lee2019_MI is ~1.1 GB per subject on disk; 54 subjects would
    want roughly 59 GB against the ~29 GB free here. Processing subject by
    subject and purging as we go holds peak usage at one subject.
    """
    freed = 0
    for path in dataset.data_path(int(subject)):
        path = Path(path)
        if path.is_file():
            freed += path.stat().st_size
            path.unlink()
    return freed


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", default="BNCI2014_004")
    ap.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    ap.add_argument("--data-dir", type=Path, default=Path("data") / "moabb")
    ap.add_argument("--npz-dir", type=Path, default=FM_NPZ_DIR)
    ap.add_argument("--no-notch", dest="notch", action="store_const", const=None,
                    default=FM_NOTCH_HZ)
    ap.add_argument(
        "--bandpass", type=float, nargs=2, default=list(FM_BANDPASS_HZ),
        metavar=("FMIN", "FMAX"),
        help="Band-pass in Hz. Default is the broadband foundation-model band; "
        "pass 8 30 to build the narrowband arm used by classical/CNN decoders.",
    )
    ap.add_argument("--sfreq", type=float, default=FM_SFREQ)
    ap.add_argument(
        "--bands",
        nargs="+",
        default=None,
        choices=sorted(BAND_PRESETS),
        help="Write these named bands in a single pass over each downloaded "
        "subject. Overrides --bandpass/--sfreq/--npz-dir.",
    )
    ap.add_argument(
        "--purge-raw",
        action="store_true",
        help="Delete each subject's downloaded source files once its npz is "
        "written. Needed for large datasets (Lee2019_MI is ~1.1 GB/subject).",
    )
    args = ap.parse_args()

    _prepare_env(args.data_dir)
    import mne

    mne.set_log_level("ERROR")

    for subject in args.subjects:
        if args.bands:
            preprocess_subject_bands(
                args.dataset, subject, list(args.bands),
                notch=args.notch, purge_raw=args.purge_raw,
            )
        else:
            preprocess_subject(
                args.dataset, subject, npz_dir=args.npz_dir, notch=args.notch,
                bandpass=tuple(args.bandpass), sfreq=args.sfreq,
                purge_raw=args.purge_raw,
            )


if __name__ == "__main__":
    main()
