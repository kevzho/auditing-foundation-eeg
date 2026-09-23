#!/usr/bin/env python3
"""Broadband preprocessing for the foundation-model arm.

Separate from ``preprocess.py`` on purpose. That pipeline band-passes to
8--30 Hz -- the mu/beta motor-imagery band -- which is right for CSP and small
convnets and wrong for LaBraM and BIOT, which pretrain on ~0.1--75 Hz broadband
EEG. Feeding narrow-band data to a broadband-pretrained model places every trial
outside the pretraining distribution, so a finding of "foundation models are
poorly calibrated" could not be distinguished from an artifact of the filter.

This module reproduces the published foundation-model recipe instead:

    band-pass 0.1--75 Hz  ->  notch 50 Hz  ->  resample 200 Hz

Deliberate omission
-------------------
ICA eye-artifact removal is **off by default**, unlike ``preprocess.py``. The
published recipes do not include it, and every extra transform is one more way
the input can differ from pretraining. ``--ica`` enables it for an explicit
sensitivity check; when enabled, that fact is recorded in the array provenance so
the two variants can never be silently mixed.

Outputs ``data/fm/bci4_2a_subject{N}_{train,eval}.npz`` with ``X``, ``y``,
``ch_names``, ``sfreq`` and a ``provenance`` JSON blob. The existing arrays under
``data/`` are left untouched, so current results stay reproducible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mne
import numpy as np

from config import DATA_DIR, ICA_N_COMPONENTS, RANDOM_SEED, SUBJECTS_TEST, SUBJECTS_TRAIN, TMAX, TMIN
from eeg_montage import BCI4_2A_CHANNELS, verify_bci4_2a_order
from models.fm_adapters import FM_BANDPASS, FM_NOTCH
from preprocess import EVAL_CUE, EVENT_ID, _load_eval_labels, _mark_eog

FM_SFREQ = 200.0
FM_NPZ_DIR = Path("data") / "fm"

# preprocess.py drops trials above 100 uV peak-to-peak. That threshold assumes
# 8-30 Hz data; on broadband signal, normal drift and blinks routinely exceed it
# and would discard most of the recording. Rejection is therefore disabled by
# default here and left to the model, matching the pretraining recipes.
DEFAULT_REJECT_UV = None


def preprocess_subject_fm(
    subject: str,
    data_dir: Path = DATA_DIR,
    labels_dir: Path = DATA_DIR,
    npz_dir: Path = FM_NPZ_DIR,
    *,
    use_ica: bool = False,
    reject_uv: float | None = DEFAULT_REJECT_UV,
    bandpass: tuple[float, float] = FM_BANDPASS,
    notch: float | None = FM_NOTCH,
    target_sfreq: float = FM_SFREQ,
) -> Path:
    """Write one broadband, FM-ready epoch array for a BCI IV-2a session."""
    gdf_path = Path(data_dir) / f"{subject}.gdf"
    if not gdf_path.exists():
        raise FileNotFoundError(f"Missing raw file: {gdf_path}")

    print(f"[preprocess_fm] {subject}")
    raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose=False)
    _mark_eog(raw)

    # FIR here rather than the IIR used for the narrow MI band: a 0.1 Hz
    # high-pass needs a very long transition band, where IIR designs become
    # numerically fragile.
    raw.filter(
        float(bandpass[0]),
        float(bandpass[1]),
        method="fir",
        phase="zero",
        fir_design="firwin",
        verbose=False,
    )
    if notch:
        # Only apply harmonics that are below Nyquist for the *current* rate.
        freqs = [f for f in np.arange(notch, raw.info["sfreq"] / 2.0, notch)]
        if freqs:
            raw.notch_filter(freqs, verbose=False)

    if use_ica:
        ica = mne.preprocessing.ICA(
            n_components=ICA_N_COMPONENTS, random_state=RANDOM_SEED, max_iter="auto", verbose=False
        )
        ica.fit(raw, verbose=False)
        eog_picks = mne.pick_types(raw.info, eog=True)
        if len(eog_picks) > 0:
            bads, _ = ica.find_bads_eog(raw, ch_name=raw.ch_names[int(eog_picks[0])], verbose=False)
            ica.exclude = list(bads)
        ica.apply(raw, verbose=False)

    verify_bci4_2a_order(raw.ch_names)
    raw.pick("eeg")

    # Resample before epoching so epoch boundaries land on true sample edges.
    if not np.isclose(raw.info["sfreq"], target_sfreq):
        raw.resample(float(target_sfreq), verbose=False)

    events, annotation_ids = mne.events_from_annotations(raw, verbose=False)
    if subject.endswith("E"):
        if str(EVAL_CUE) not in annotation_ids:
            raise KeyError(f"{subject} is missing expected evaluation cue {EVAL_CUE}")
        labels = _load_eval_labels(subject, labels_dir=Path(labels_dir))
        cue_code = annotation_ids[str(EVAL_CUE)]
        cue_events = events[events[:, 2] == cue_code]
        epoch_event_id = {"Unknown": cue_code}
        if labels.shape[0] != cue_events.shape[0]:
            raise ValueError(
                f"{subject}: labels ({labels.shape[0]}) != evaluation cues ({cue_events.shape[0]})"
            )
    else:
        missing = [str(c) for c in EVENT_ID.values() if str(c) not in annotation_ids]
        if missing:
            raise KeyError(f"{subject} is missing class annotations: {missing}")
        epoch_event_id = {name: annotation_ids[str(code)] for name, code in EVENT_ID.items()}
        cue_events = events[np.isin(events[:, 2], list(epoch_event_id.values()))]
        label_by_event = {annotation_ids[str(c)]: c for c in EVENT_ID.values()}
        labels = np.array([label_by_event[int(e)] for e in cue_events[:, 2]], dtype=int)

    reject = {"eeg": float(reject_uv) * 1e-6} if reject_uv else None
    epochs = mne.Epochs(
        raw,
        cue_events,
        epoch_event_id,
        tmin=TMIN,
        tmax=TMAX,
        baseline=None,
        preload=True,
        reject=reject,
        event_repeated="drop",
        verbose=False,
    )

    X = epochs.get_data(copy=True).astype(np.float32, copy=False)  # volts
    y = labels[epochs.selection].astype(np.int16, copy=False)

    provenance = {
        "pipeline": "preprocess_fm",
        "bandpass_hz": [float(bandpass[0]), float(bandpass[1])],
        "notch_hz": float(notch) if notch else None,
        "sfreq_hz": float(epochs.info["sfreq"]),
        "ica_applied": bool(use_ica),
        "reject_uv": float(reject_uv) if reject_uv else None,
        "tmin_s": float(TMIN),
        "tmax_s": float(TMAX),
        "units": "V",
        "n_trials": int(X.shape[0]),
        "n_dropped": int(len(cue_events) - X.shape[0]),
    }

    subject_id = int(subject[1:3])
    split = "train" if subject.endswith("T") else "eval"
    npz_dir = Path(npz_dir)
    npz_dir.mkdir(parents=True, exist_ok=True)
    path = npz_dir / f"bci4_2a_subject{subject_id}_{split}.npz"
    np.savez_compressed(
        path,
        X=X,
        y=y,
        ch_names=np.asarray(BCI4_2A_CHANNELS, dtype="<U8"),
        sfreq=np.asarray(float(epochs.info["sfreq"]), dtype=np.float64),
        provenance=np.asarray(json.dumps(provenance)),
    )
    print(
        f"[preprocess_fm] {subject}: {X.shape} @ {provenance['sfreq_hz']:g} Hz "
        f"({provenance['n_dropped']} dropped) -> {path}"
    )
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", nargs="+", default=list(SUBJECTS_TRAIN) + list(SUBJECTS_TEST))
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--labels-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--npz-dir", type=Path, default=FM_NPZ_DIR)
    ap.add_argument("--ica", action="store_true", help="Apply ICA eye-artifact removal (off by default).")
    ap.add_argument("--reject-uv", type=float, default=None, help="Peak-to-peak rejection in uV. Off by default.")
    ap.add_argument(
        "--bandpass",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=list(FM_BANDPASS),
        help=(
            "Band-pass in Hz. Defaults to the FM pretraining band. Set 8 30 to build the "
            "narrowband control that asks whether the MI band is missing from the input or "
            "present but discarded by the encoder. Sampling rate is unchanged either way, so "
            "the band is the only difference from the broadband arrays."
        ),
    )
    args = ap.parse_args()

    for subject in args.subjects:
        preprocess_subject_fm(
            subject,
            data_dir=args.data_dir,
            labels_dir=args.labels_dir,
            npz_dir=args.npz_dir,
            use_ica=bool(args.ica),
            reject_uv=args.reject_uv,
            bandpass=(float(args.bandpass[0]), float(args.bandpass[1])),
        )


if __name__ == "__main__":
    main()
