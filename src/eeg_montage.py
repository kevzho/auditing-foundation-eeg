#!/usr/bin/env python3
"""Channel identity for EEG arrays.

Foundation models such as LaBraM and BIOT take an explicit channel-order list and
look up a learned embedding per electrode, so channel identity is part of their
input contract rather than metadata. The arrays this project produced before
2026-08-09 carry only ``X`` and ``y`` -- names were dropped at
``preprocess.py`` and at ``raw.pick("eeg")`` in the workflow -- so this module
supplies the canonical names for legacy arrays and normalizes names to 10-20
form for new ones.

BCI IV-2a naming
----------------
The raw GDF files label only five electrodes and number the rest::

    EEG-Fz, EEG-0 ... EEG-5, EEG-C3, EEG-6, EEG-Cz, EEG-7, EEG-C4,
    EEG-8 ... EEG-14, EEG-Pz, EEG-15, EEG-16

The competition description gives the 22-electrode layout as rows of
1/5/7/5/3/1. Mapping that layout onto the file order places Fz at index 0, C3 at
7, Cz at 9, C4 at 11 and Pz at 19 -- which is exactly where the five *named*
channels fall in the GDF. That agreement on all five anchors is what validates
the mapping below.
"""

from __future__ import annotations

import numpy as np

# Canonical 10-20 names for the 22 BCI IV-2a EEG channels, in file order.
BCI4_2A_CHANNELS: tuple[str, ...] = (
    "Fz",
    "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2",
    "POz",
)

# Raw GDF names, in file order, for cross-checking a load against the mapping.
BCI4_2A_GDF_NAMES: tuple[str, ...] = (
    "EEG-Fz",
    "EEG-0", "EEG-1", "EEG-2", "EEG-3", "EEG-4",
    "EEG-5", "EEG-C3", "EEG-6", "EEG-Cz", "EEG-7", "EEG-C4", "EEG-8",
    "EEG-9", "EEG-10", "EEG-11", "EEG-12", "EEG-13",
    "EEG-14", "EEG-Pz", "EEG-15",
    "EEG-16",
)

# The five electrodes the GDF names explicitly, and where they must land.
BCI4_2A_ANCHORS: dict[int, str] = {0: "Fz", 7: "C3", 9: "Cz", 11: "C4", 19: "Pz"}

BCI4_2A_SFREQ = 250.0

# Canonical channel sets per dataset key, for arrays that lack stored names.
LEGACY_CHANNELS: dict[str, tuple[str, ...]] = {
    "bci4_2a": BCI4_2A_CHANNELS,
}
LEGACY_SFREQ: dict[str, float] = {
    "bci4_2a": BCI4_2A_SFREQ,
}


def normalize_channel_name(name: str) -> str:
    """Reduce a recorded channel label to bare 10-20 form.

    ``EEG-C3`` -> ``C3``, ``eeg c3`` -> ``C3``, ``C3.`` -> ``C3``. Numeric GDF
    placeholders such as ``EEG-7`` have no 10-20 equivalent and are returned
    stripped of the prefix; callers should prefer the positional mapping in
    :data:`BCI4_2A_CHANNELS` for those.
    """
    text = str(name).strip()
    for prefix in ("EEG-", "EEG ", "eeg-", "eeg ", "EEG_", "eeg_"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    text = text.replace(".", "").strip()
    if not text:
        return str(name).strip()
    # 10-20 convention: leading letters upper, trailing z lower (e.g. FCz, POz).
    if text[-1] in {"z", "Z"} and len(text) > 1:
        return text[:-1].upper() + "z"
    return text.upper()


def verify_bci4_2a_order(gdf_names: list[str]) -> None:
    """Fail loudly if a GDF's EEG channel order does not match the mapping.

    Guards against a silently re-ordered or differently-exported file turning
    into wrong electrode identities downstream.
    """
    eeg = [n for n in gdf_names if n.upper().startswith("EEG")]
    if len(eeg) != len(BCI4_2A_CHANNELS):
        raise ValueError(f"expected {len(BCI4_2A_CHANNELS)} EEG channels, got {len(eeg)}: {eeg}")
    for index, expected in BCI4_2A_ANCHORS.items():
        found = normalize_channel_name(eeg[index])
        if found != expected:
            raise ValueError(
                f"channel order mismatch at index {index}: expected {expected}, found {eeg[index]!r}. "
                "The positional mapping in BCI4_2A_CHANNELS cannot be trusted for this file."
            )


def channels_for(dataset: str, n_channels: int) -> tuple[str, ...]:
    """Canonical names for a legacy array with no stored channel identity."""
    known = LEGACY_CHANNELS.get(str(dataset).lower())
    if known is None:
        raise KeyError(
            f"no canonical channel list for dataset {dataset!r}; "
            "re-run preprocessing so ch_names is stored in the array"
        )
    if len(known) != int(n_channels):
        raise ValueError(
            f"{dataset}: canonical list has {len(known)} channels but array has {n_channels}"
        )
    return known


def load_npz_with_channels(path, dataset: str | None = None) -> dict:
    """Load a workflow NPZ, filling in channel identity when it is absent.

    New arrays carry ``ch_names`` and ``sfreq``. Arrays written before
    2026-08-09 do not; for those the canonical list is substituted and
    ``ch_names_source`` records that it was inferred rather than recorded.
    """
    from pathlib import Path

    path = Path(path)
    loaded = np.load(path, allow_pickle=False)
    out: dict = {"X": loaded["X"], "y": loaded["y"]}
    if dataset is None:
        dataset = path.name.split("_subject")[0]

    if "ch_names" in loaded.files:
        out["ch_names"] = tuple(str(c) for c in loaded["ch_names"])
        out["ch_names_source"] = "recorded"
    else:
        out["ch_names"] = channels_for(dataset, out["X"].shape[1])
        out["ch_names_source"] = "inferred_canonical"

    if "sfreq" in loaded.files:
        out["sfreq"] = float(loaded["sfreq"])
        out["sfreq_source"] = "recorded"
    else:
        out["sfreq"] = float(LEGACY_SFREQ.get(str(dataset).lower(), float("nan")))
        out["sfreq_source"] = "inferred_canonical"

    out["dataset"] = dataset
    return out
