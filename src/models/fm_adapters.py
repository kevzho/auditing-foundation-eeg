#!/usr/bin/env python3
"""Convert this project's EEG epochs into what pretrained foundation models expect.

Each foundation model has a hard input contract -- sampling rate, patch length,
amplitude units, and an explicit channel-order list feeding a learned per-electrode
embedding. Getting any of these wrong does not error; it silently produces
embeddings from data unlike anything in pretraining, which would confound an
uncertainty audit with an input-format artifact.

.. warning::

   **The existing workflow arrays are not usable for foundation models.**
   ``config.FILTER_LOW/FILTER_HIGH`` band-pass to 8--30 Hz (the mu/beta MI band),
   while LaBraM and BIOT pretrain on ~0.1--75 Hz broadband EEG. Feeding 8--30 Hz
   data to a broadband-pretrained model puts every trial far outside the
   pretraining distribution, so any "foundation models are poorly calibrated"
   finding would be unfalsifiable -- it could just be the filter.

   The FM arm therefore needs its own preprocessing pass from the raw GDF/MOABB
   sources with :data:`FM_BANDPASS`. :func:`check_band_compatibility` enforces
   this rather than letting it pass silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import resample_poly

FM_BANDPASS: tuple[float, float] = (0.1, 75.0)
FM_NOTCH: float = 50.0


@dataclass(frozen=True)
class FMSpec:
    """Input contract for one pretrained foundation model."""

    name: str
    sfreq: float
    patch_samples: int
    units: str  # "uV" or "V"
    bandpass: tuple[float, float]
    notch: float | None
    checkpoint: str = ""
    notes: str = ""

    @property
    def patch_seconds(self) -> float:
        return self.patch_samples / self.sfreq


LABRAM_SPEC = FMSpec(
    name="labram_base_patch200_200",
    sfreq=200.0,
    patch_samples=200,  # 1 s at 200 Hz
    units="uV",
    bandpass=(0.1, 75.0),
    notch=50.0,
    checkpoint="labram-base.pth",
    notes="Requires an explicit channel-order list; unknown electrodes have no embedding.",
)

BIOT_SPEC = FMSpec(
    name="biot_eeg_prest_16",
    sfreq=200.0,
    patch_samples=200,
    units="uV",
    bandpass=(0.1, 75.0),
    notch=50.0,
    checkpoint="EEG-PREST-16-channels.ckpt",
    notes="Pretrained on a 16-channel montage; channel tags accompany each patch.",
)

FM_SPECS: dict[str, FMSpec] = {
    "labram": LABRAM_SPEC,
    "biot": BIOT_SPEC,
}


class BandMismatchError(ValueError):
    """Raised when source data is band-limited more narrowly than the FM expects."""


def check_band_compatibility(
    source_bandpass: tuple[float, float],
    spec: FMSpec,
    *,
    tolerance_hz: float = 2.0,
    strict: bool = True,
) -> list[str]:
    """Reject source data whose passband is far narrower than the model's.

    Returns the list of problems found. With ``strict`` (the default) a non-empty
    list raises, because silently proceeding produces a confounded audit.
    """
    low, high = float(source_bandpass[0]), float(source_bandpass[1])
    want_low, want_high = spec.bandpass
    problems: list[str] = []
    if low > want_low + tolerance_hz:
        problems.append(
            f"source high-passes at {low} Hz but {spec.name} pretrains from {want_low} Hz: "
            f"all content below {low} Hz is missing"
        )
    if high < want_high - tolerance_hz:
        problems.append(
            f"source low-passes at {high} Hz but {spec.name} pretrains to {want_high} Hz: "
            f"all content above {high} Hz is missing"
        )
    if problems and strict:
        raise BandMismatchError(
            "; ".join(problems)
            + f". Re-preprocess with FM_BANDPASS={FM_BANDPASS} instead of reusing the MI-band arrays."
        )
    return problems


def to_microvolts(X: np.ndarray, source_units: str) -> np.ndarray:
    """Scale to microvolts. MNE stores volts; foundation models expect uV."""
    units = str(source_units).lower()
    if units in {"uv", "µv", "microvolt", "microvolts"}:
        return np.asarray(X, dtype=np.float32)
    if units in {"v", "volt", "volts"}:
        return (np.asarray(X, dtype=np.float64) * 1e6).astype(np.float32)
    raise ValueError(f"unrecognized units {source_units!r}; expected 'V' or 'uV'")


def infer_units(X: np.ndarray) -> str:
    """Guess amplitude units from magnitude.

    Scalp EEG is tens of microvolts, i.e. ~1e-5 V. A standard deviation below
    1e-3 means the array is almost certainly in volts.
    """
    scale = float(np.nanstd(np.asarray(X, dtype=np.float64)))
    return "V" if scale < 1e-3 else "uV"


def resample_epochs(X: np.ndarray, source_sfreq: float, target_sfreq: float) -> np.ndarray:
    """Polyphase resample along the time axis.

    ``resample_poly`` is used rather than Fourier resampling because it does not
    assume periodicity across the epoch boundary.
    """
    source, target = float(source_sfreq), float(target_sfreq)
    if np.isclose(source, target):
        return np.asarray(X, dtype=np.float32)
    from math import gcd

    a, b = int(round(target)), int(round(source))
    divisor = gcd(a, b)
    up, down = a // divisor, b // divisor
    out = resample_poly(np.asarray(X, dtype=np.float64), up, down, axis=-1)
    return out.astype(np.float32)


def crop_to_patches(
    X: np.ndarray,
    patch_samples: int,
    *,
    align: str = "start",
) -> np.ndarray:
    """Trim the time axis to a whole number of patches.

    ``align='start'`` keeps the beginning of the window, which for cue-locked MI
    epochs keeps the post-cue onset where the discriminative signal lives.
    ``'center'`` and ``'end'`` are provided for windows defined differently.
    """
    n_times = X.shape[-1]
    n_patches = n_times // int(patch_samples)
    if n_patches < 1:
        raise ValueError(
            f"epoch has {n_times} samples, shorter than one {patch_samples}-sample patch"
        )
    keep = n_patches * int(patch_samples)
    if keep == n_times:
        return np.asarray(X, dtype=np.float32)
    if align == "start":
        start = 0
    elif align == "end":
        start = n_times - keep
    elif align == "center":
        start = (n_times - keep) // 2
    else:
        raise ValueError(f"align must be start/center/end, got {align!r}")
    return np.asarray(X[..., start : start + keep], dtype=np.float32)


@dataclass
class PreparedEpochs:
    """FM-ready epochs plus the provenance needed to reproduce them."""

    X: np.ndarray
    ch_names: tuple[str, ...]
    sfreq: float
    n_patches: int
    spec_name: str
    provenance: dict = field(default_factory=dict)


def prepare_for_fm(
    X: np.ndarray,
    ch_names,
    source_sfreq: float,
    spec: FMSpec,
    *,
    source_units: str | None = None,
    source_bandpass: tuple[float, float] | None = None,
    align: str = "start",
    strict_band: bool = True,
) -> PreparedEpochs:
    """Convert ``(trials, channels, samples)`` epochs into a model's input contract.

    Order matters: unit conversion, then resample, then crop. Resampling before
    cropping keeps the crop boundary on a true patch edge at the target rate.
    """
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"expected (trials, channels, samples), got shape {X.shape}")
    ch_names = tuple(str(c) for c in ch_names)
    if len(ch_names) != X.shape[1]:
        raise ValueError(f"{len(ch_names)} channel names for {X.shape[1]} channels")

    if source_bandpass is not None:
        check_band_compatibility(source_bandpass, spec, strict=strict_band)

    units = source_units or infer_units(X)
    out = to_microvolts(X, units)
    out = resample_epochs(out, source_sfreq, spec.sfreq)
    out = crop_to_patches(out, spec.patch_samples, align=align)
    n_patches = out.shape[-1] // spec.patch_samples

    return PreparedEpochs(
        X=out,
        ch_names=ch_names,
        sfreq=spec.sfreq,
        n_patches=n_patches,
        spec_name=spec.name,
        provenance={
            "source_sfreq": float(source_sfreq),
            "source_units": units,
            "source_bandpass": list(source_bandpass) if source_bandpass else None,
            "source_samples": int(X.shape[-1]),
            "target_sfreq": float(spec.sfreq),
            "target_samples": int(out.shape[-1]),
            "patch_samples": int(spec.patch_samples),
            "n_patches": int(n_patches),
            "align": align,
        },
    )
