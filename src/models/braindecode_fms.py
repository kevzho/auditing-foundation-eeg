"""Braindecode-backed EEG foundation models, behind one interface.

Why braindecode rather than another hand port: a single-model result cannot
support a claim about *foundation models*, and hand-porting each release is both
slow and a liability -- every negative result invites "you ported it wrong".
Braindecode ships reference implementations with author-released weights on the
Hugging Face hub, so the models here are the community's implementations, not
ours. The one exception is LaBraM, which this project also ports directly in
``models/labram_probe``; running both is a deliberate cross-check.

Input contract
--------------
All models here take microvolts divided by 100 at 200 Hz in 200-sample patches
-- CBraMod's dataset loader returns ``data/100`` exactly as LaBraM's does. That
means ``models.fm_adapters.prepare_for_fm`` output feeds these unchanged, and
the amplitude divisor stays in one place.

Frozen features
---------------
``forward_features`` exposes two poolings. Mean pooling over channels and
patches gives a 200-dim embedding directly comparable to LaBraM's mean-pooled
output; flattening gives the geometry CBraMod's own downstream heads use. Which
one is better is not obvious a priori, so the runner selects between them on the
validation subset -- never on held-out data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence

import numpy as np
import torch
import torch.nn as nn

#: Shared with LaBraM. Both models were pretrained on microvolts scaled by 1/100.
FM_AMPLITUDE_DIVISOR = 100.0

Pooling = Literal["mean", "flatten"]


@dataclass(frozen=True)
class BDFMSpec:
    """Everything needed to build one braindecode foundation model."""

    name: str
    repo: str
    builder: str  # attribute name in braindecode.models
    sfreq: float = 200.0
    patch_samples: int = 200
    amplitude_divisor: float = FM_AMPLITUDE_DIVISOR
    bandpass: tuple[float, float] = (0.1, 75.0)
    notch: float | None = 50.0
    #: Models whose pretrained positional tables fix the montage need chs_info
    #: with 3-D locations rather than a bare channel count.
    needs_locations: bool = False
    extra_kwargs: tuple[tuple[str, Any], ...] = ()


BD_FM_SPECS: dict[str, BDFMSpec] = {
    "cbramod": BDFMSpec(
        name="cbramod",
        repo="braindecode/cbramod-pretrained",
        builder="CBraMod",
        # CBraMod is pretrained on TUEG; its asymmetric conditional positional
        # encoding is generated convolutionally, so it accepts an arbitrary
        # channel count without a montage lookup table.
        needs_locations=False,
    ),
}


class UnsupportedFMError(ValueError):
    """Raised for a model name with no registered spec."""


def as_fm_spec(spec: BDFMSpec):
    """Adapt to ``fm_adapters.FMSpec`` so resampling and the band guard are shared.

    Preprocessing must not fork per model: the band-compatibility check that
    stops 8-30 Hz data reaching a broadband-pretrained encoder lives in
    ``fm_adapters`` and has to apply here too.
    """
    from models.fm_adapters import FMSpec

    return FMSpec(
        name=spec.name,
        sfreq=spec.sfreq,
        patch_samples=spec.patch_samples,
        units="uV",
        bandpass=spec.bandpass,
        notch=spec.notch,
        checkpoint=spec.repo,
    )


def get_spec(name: str) -> BDFMSpec:
    try:
        return BD_FM_SPECS[name]
    except KeyError:
        raise UnsupportedFMError(
            f"unknown braindecode foundation model {name!r}; "
            f"registered: {sorted(BD_FM_SPECS)}"
        ) from None


def build_chs_info(ch_names: Sequence[str], sfreq: float) -> list[dict]:
    """Channel dicts carrying 3-D positions from the standard 10-05 montage.

    Only needed by models that interpolate onto a fixed pretrained montage;
    building it eagerly for every model would make MNE a hard import here.
    """
    import mne

    info = mne.create_info([str(n) for n in ch_names], float(sfreq), "eeg")
    info.set_montage(
        mne.channels.make_standard_montage("standard_1005"), match_case=False
    )
    chs = [dict(c) for c in info["chs"]]
    missing = [
        c["ch_name"] for c in chs if not np.any(np.asarray(c["loc"][:3], dtype=float))
    ]
    if missing:
        raise ValueError(f"no montage position for channels: {missing}")
    return chs


def build_model(
    spec: BDFMSpec,
    ch_names: Sequence[str],
    n_times: int,
    n_outputs: int,
    device: str = "cpu",
    pretrained: bool = True,
) -> tuple[nn.Module, dict]:
    """Instantiate a model. Returns (model, audit report).

    ``pretrained=False`` builds the identical architecture with random weights.
    That is the control a negative transfer result needs: without it, "the
    foundation model performs at chance" cannot be separated from "this
    architecture with this much target data performs at chance".
    """
    import braindecode.models as bd_models

    cls = getattr(bd_models, spec.builder)
    kwargs: dict[str, Any] = {
        "n_outputs": int(n_outputs),
        "n_times": int(n_times),
        "sfreq": float(spec.sfreq),
        **dict(spec.extra_kwargs),
    }
    if spec.needs_locations:
        kwargs["chs_info"] = build_chs_info(ch_names, spec.sfreq)
    else:
        kwargs["n_chans"] = int(len(ch_names))

    model = (cls.from_pretrained(spec.repo, **kwargs) if pretrained else cls(**kwargs)).to(device)
    report = {
        "model": spec.name,
        "builder": spec.builder,
        "checkpoint": spec.repo if pretrained else "random_init",
        "pretrained": bool(pretrained),
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "n_times": int(n_times),
        "n_chans": int(len(ch_names)),
        "amplitude_divisor": float(spec.amplitude_divisor),
    }
    return model, report


# --- feature extraction -----------------------------------------------------


def _cbramod_trunk(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Everything up to but excluding the classification head."""
    h = model.rearrange(x)
    h = model.patch_embedding(h)
    h = model.encoder(h)
    return model.proj_out(h)


#: Registered per model because the head boundary differs by architecture.
_TRUNKS: dict[str, Callable[[nn.Module, torch.Tensor], torch.Tensor]] = {
    "cbramod": _cbramod_trunk,
}


def pool(h: torch.Tensor, pooling: Pooling) -> torch.Tensor:
    """Reduce a (batch, chans, patches, dim) trunk output to (batch, features)."""
    if pooling == "mean":
        return h.mean(dim=tuple(range(1, h.ndim - 1)))
    if pooling == "flatten":
        return h.flatten(1)
    raise ValueError(f"unknown pooling {pooling!r}")


@torch.no_grad()
def extract_features(
    model: nn.Module,
    spec: BDFMSpec,
    X_microvolts: np.ndarray,
    *,
    pooling: Pooling = "mean",
    device: str = "cpu",
    batch_size: int = 64,
) -> np.ndarray:
    """Frozen trunk features for ``(n_trials, n_chans, n_times)`` microvolt data."""
    trunk = _TRUNKS.get(spec.name)
    if trunk is None:
        raise UnsupportedFMError(f"no trunk defined for {spec.name!r}")
    model.eval()
    out = []
    X = np.asarray(X_microvolts, dtype=np.float32)
    for start in range(0, X.shape[0], batch_size):
        batch = torch.from_numpy(X[start : start + batch_size]).to(device)
        batch = batch / spec.amplitude_divisor
        out.append(pool(trunk(model, batch), pooling).float().cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, 0), dtype=np.float32)


# --- fine-tuning ------------------------------------------------------------


def layer_id(name: str, n_blocks: int) -> int:
    """Depth index for layer-wise learning-rate decay, braindecode naming.

    Mirrors ``run_fm_probe._layer_id`` but for braindecode's module names:
    the patch embedding sits at 0, encoder block ``i`` at ``i+1``, and the
    projection and head above everything.
    """
    if name.startswith("final_layer") or name.startswith("proj_out"):
        return n_blocks + 1
    if name.startswith("encoder.layers."):
        return int(name.split(".")[2]) + 1
    return 0  # patch_embedding and any positional parameters


def frozen_modules(model: nn.Module) -> list[nn.Module]:
    """Modules whose parameters are all frozen.

    Partial fine-tuning should not run dropout inside a block that is not
    learning: the stochasticity only injects noise into features that cannot
    adapt to it. Keeping these in eval mode is the standard choice, and it also
    sidesteps a concrete PyTorch limitation -- with no upstream tensor requiring
    grad, MPS selects a fused attention kernel that raises
    ``NotImplementedError`` on dropout.
    """
    out = []
    for module in model.modules():
        params = list(module.parameters(recurse=True))
        if params and all(not p.requires_grad for p in params):
            out.append(module)
    return out


def count_blocks(model: nn.Module) -> int:
    encoder = getattr(model, "encoder", None)
    layers = getattr(encoder, "layers", None)
    if layers is None:
        raise UnsupportedFMError("model has no encoder.layers to count")
    return len(layers)
