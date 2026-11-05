"""MIRepNet input contract, expressed in the same terms as the other adapters.

MIRepNet is the first EEG foundation model pretrained specifically for motor
imagery, as opposed to the general-purpose clinical pretraining behind LaBraM
and CBraMod. Auditing it answers a question the current audit cannot: whether
the four-class deficit is a property of pretraining breadth rather than of
pretraining as such.

Contract, read from the upstream repository (staraink/MIRepNet, MIT):

    sample_rate = 250                       dataset.py
    X_resampled = X[:, :, :1000]            dataset.py  -> 4.0 s window
    unified channel template                utils/channel_list.py

Two consequences worth stating before any number is reported.

1.  At 250 Hz and 4 s, MIRepNet's contract is close to this project's
    *narrowband* supervised pathway rather than the 200 Hz broadband pathway
    that LaBraM and CBraMod require. The matched-input confound is therefore
    much smaller for MIRepNet, which makes it a cleaner comparison and means
    the decomposition in the paper does not transfer to it unchanged.

2.  The upstream preprocessing applies "frequency filtering" whose cutoffs are
    not stated in the repository README or in dataset.py. This module therefore
    refuses to run until the band is supplied explicitly, rather than guessing
    and silently comparing across mismatched inputs. Read the value out of the
    paper or the preprocessing source, then pass it in.
"""
from __future__ import annotations


import numpy as np

from .fm_adapters import BandMismatchError, FMSpec, PreparedEpochs, prepare_for_fm

#: Upstream constants. MIRepNet crops to a fixed 4 s window rather than tiling
#: patches, so the whole window is one "patch" in this project's terms.
MIREPNET_SFREQ = 250.0
MIREPNET_SAMPLES = 1000

#: Sentinel. A real (low, high) pair must be supplied by the caller.
BAND_UNVERIFIED: tuple[float, float] | None = None


class BandUnverifiedError(BandMismatchError):
    """Raised when MIRepNet's filter band has not been confirmed upstream."""


def mirepnet_spec(bandpass: tuple[float, float], notch: float | None = 50.0) -> FMSpec:
    """Build the spec once the upstream filter band has actually been checked.

    Passing a band is deliberate friction. The other adapters hard-code a band
    because it is documented; this one is not, and a wrong band would produce a
    deficit attributable to the input rather than to the model.
    """
    lo, hi = float(bandpass[0]), float(bandpass[1])
    if not (0.0 <= lo < hi):
        raise ValueError(f"implausible bandpass {bandpass}")
    return FMSpec(
        name="mirepnet",
        sfreq=MIREPNET_SFREQ,
        patch_samples=MIREPNET_SAMPLES,
        units="uV",
        bandpass=(lo, hi),
        notch=notch,
        checkpoint="weight/MIRepNet.pth",
        notes=(
            "Motor-imagery-specific pretraining. 4 s window at 250 Hz. Channel "
            "template alignment is part of the contract; see align_to_template."
        ),
    )


def align_to_template(
    X: np.ndarray,
    ch_names,
    template,
    *,
    fill: float = 0.0,
) -> tuple[np.ndarray, tuple[str, ...], dict]:
    """Project epochs onto MIRepNet's channel template.

    ``template`` must be the list imported from the upstream
    ``utils.channel_list`` module, not a transcription of it. Channels present
    in the recording are copied into their template slot; template positions the
    recording does not have are filled and reported, because a silently padded
    montage is the kind of thing that looks like a model result.

    Matching is case-insensitive on the electrode name only. No spatial
    interpolation is performed: this maps what exists and declares what does
    not.
    """
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"expected (trials, channels, samples), got {X.shape}")
    have = {str(c).upper(): i for i, c in enumerate(ch_names)}
    template = tuple(str(c).upper() for c in template)
    if len(have) != X.shape[1]:
        raise ValueError(f"{len(have)} unique channel names for {X.shape[1]} channels")

    out = np.full((X.shape[0], len(template), X.shape[2]), fill, dtype=X.dtype)
    matched, missing = [], []
    for slot, name in enumerate(template):
        idx = have.get(name)
        if idx is None:
            missing.append(name)
        else:
            out[:, slot, :] = X[:, idx, :]
            matched.append(name)
    unused = sorted(set(have) - set(template))

    provenance = {
        "template_size": len(template),
        "matched": len(matched),
        "missing_from_recording": missing,
        "recording_channels_unused": unused,
        "fill_value": float(fill),
    }
    return out, template, provenance


def prepare_for_mirepnet(
    X: np.ndarray,
    ch_names,
    source_sfreq: float,
    *,
    bandpass: tuple[float, float] | None = BAND_UNVERIFIED,
    template=None,
    source_bandpass: tuple[float, float] | None = None,
    strict_band: bool = True,
    align: str = "start",
) -> PreparedEpochs:
    """Full MIRepNet preparation: template alignment, then the shared pipeline.

    Raises if the upstream band has not been supplied, and if no channel
    template is given.
    """
    if bandpass is None:
        raise BandUnverifiedError(
            "MIRepNet's filter band is not documented in the upstream README or "
            "dataset.py. Read it from the paper or the preprocessing source and "
            "pass bandpass=(low, high). Guessing it would make any deficit "
            "attributable to the input rather than the model."
        )
    if template is None:
        raise ValueError(
            "Pass the channel template imported from MIRepNet's "
            "utils/channel_list.py (e.g. use_channels_names). Do not transcribe "
            "it; import it, so an upstream change cannot silently desync."
        )

    aligned, aligned_names, ch_prov = align_to_template(X, ch_names, template)
    spec = mirepnet_spec(bandpass)
    prepared = prepare_for_fm(
        aligned,
        aligned_names,
        source_sfreq,
        spec,
        source_units=None,
        source_bandpass=source_bandpass,
        align=align,
        strict_band=strict_band,
    )
    prepared.provenance["channel_alignment"] = ch_prov
    prepared.provenance["upstream"] = {
        "repo": "staraink/MIRepNet",
        "license": "MIT",
        "sample_rate": MIREPNET_SFREQ,
        "window_samples": MIREPNET_SAMPLES,
        "band_source": "supplied by caller; not documented upstream",
    }
    return prepared
