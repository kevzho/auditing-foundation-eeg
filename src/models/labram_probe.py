#!/usr/bin/env python3
"""Frozen-feature extraction from the pretrained LaBraM encoder.

Wraps the reference implementation in ``third_party/LaBraM`` so the rest of this
project can treat LaBraM as a feature extractor without inheriting its training
harness. Three parts of its input contract are silent failure modes -- getting
any of them wrong yields plausible-looking embeddings computed from data the
model never saw in pretraining:

1. **Amplitude.** ``engine_for_finetuning`` divides by 100 *after* converting to
   microvolts, so the model expects uV/100. Volts (MNE's default) are off by 1e8.
2. **Channel identity.** ``pos_embed`` is indexed by position in a fixed
   vocabulary, with index 0 reserved for the CLS token. Names must be uppercase
   (``FZ``, not ``Fz``), and an electrode outside the vocabulary has no embedding.
3. **Shape.** Input is ``(B, channels, patches, 200)`` -- patched along time, not
   a flat time series.

The vocabulary is vendored from ``third_party/LaBraM/utils.py`` rather than
imported because that module pulls in ``tensorboardX`` for training-time logging
that is irrelevant here. :func:`verify_vendored_vocab` checks the copy against
the source at runtime so the two cannot drift apart silently.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

DEFAULT_LABRAM_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "LaBraM"
DEFAULT_CHECKPOINT = "checkpoints/labram-base.pth"

# LaBraM applies this after conversion to microvolts (engine_for_finetuning.py).
LABRAM_AMPLITUDE_DIVISOR = 100.0

# Hyperparameters matching run_class_finetuning.py's argparse defaults. timm
# >=1.0 no longer supplies a default for init_values, and the reference code does
# `if init_values > 0`, so omitting it raises TypeError on None.
LABRAM_MODEL_KWARGS = dict(
    drop_rate=0.0,
    drop_path_rate=0.1,
    attn_drop_rate=0.0,
    drop_block_rate=None,
    use_mean_pooling=True,
    init_scale=0.001,
    use_rel_pos_bias=False,
    use_abs_pos_emb=True,
    init_values=0.1,
    qkv_bias=False,
)

# Vendored from third_party/LaBraM/utils.py :: standard_1020.
STANDARD_1020: tuple[str, ...] = (
    "FP1", "FPZ", "FP2",
    "AF9", "AF7", "AF5", "AF3", "AF1", "AFZ", "AF2", "AF4", "AF6", "AF8", "AF10",
    "F9", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "F10",
    "FT9", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "FT10",
    "T9", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "T10",
    "TP9", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "TP10",
    "P9", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "P10",
    "PO9", "PO7", "PO5", "PO3", "PO1", "POZ", "PO2", "PO4", "PO6", "PO8", "PO10",
    "O1", "OZ", "O2", "O9", "CB1", "CB2",
    "IZ", "O10", "T3", "T5", "T4", "T6", "M1", "M2", "A1", "A2",
    "CFC1", "CFC2", "CFC3", "CFC4", "CFC5", "CFC6", "CFC7", "CFC8",
    "CCP1", "CCP2", "CCP3", "CCP4", "CCP5", "CCP6", "CCP7", "CCP8",
    "T1", "T2", "FTT9h", "TTP7h", "TPP9h", "FTT10h", "TPP8h", "TPP10h",
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
)


class ChannelNotInVocabularyError(KeyError):
    """An electrode has no pretrained positional embedding."""


def labram_root(root: str | Path | None = None) -> Path:
    return Path(root or os.environ.get("LABRAM_ROOT", DEFAULT_LABRAM_ROOT))


def verify_vendored_vocab(root: str | Path | None = None) -> None:
    """Fail if the vendored vocabulary has drifted from the reference source.

    A shifted index silently changes which electrode every embedding refers to,
    so this is checked rather than assumed.
    """
    utils_py = labram_root(root) / "utils.py"
    if not utils_py.exists():
        raise FileNotFoundError(f"LaBraM source not found at {utils_py}")
    text = utils_py.read_text(encoding="utf-8")
    start = text.index("standard_1020 = [")
    body = text[start : text.index("]", start) + 1]
    namespace: dict = {}
    exec(body, namespace)  # noqa: S102 - a literal list assignment from a pinned file
    reference = tuple(namespace["standard_1020"])
    if reference != STANDARD_1020:
        raise RuntimeError(
            "vendored STANDARD_1020 no longer matches third_party/LaBraM/utils.py; "
            "channel indices would be wrong"
        )


#: Case-insensitive view of the vocabulary, built once.
#:
#: Six of the 136 vendored entries carry a lowercase 10-05 suffix -- FTT9h,
#: TTP7h, TPP9h, FTT10h, TPP8h, TPP10h. Upstream's ``get_input_chans`` looks
#: names up verbatim, so those six resolve there and must resolve here too.
#: Uppercasing the query before an exact match against a mixed-case vocabulary
#: made them permanently unreachable: every one of them raised
#: ChannelNotInVocabularyError no matter how the name was spelled, including
#: when spelled exactly as the vocabulary spells it.
#:
#: This never fired on BCI IV-2a or BNCI2014_004, whose montages are plain
#: 10-20. It fired on all 54 Lee2019_MI subjects, which carry all six, and it
#: presented as "LaBraM cannot be evaluated on this dataset" rather than as a
#: lookup bug -- which would have silently cost the paper a comparison.
_VOCAB_BY_UPPER = {name.upper(): i for i, name in enumerate(STANDARD_1020)}


def channel_indices(ch_names) -> list[int]:
    """Map electrode names to LaBraM ``input_chans``.

    Index 0 is reserved for the CLS token, so vocabulary positions are offset by
    one -- matching ``utils.get_input_chans``.

    Lookup is case-insensitive, which is a superset of upstream's exact match:
    every name upstream resolves, this resolves to the same index, plus common
    case variants (``Fz`` for ``FZ``). The returned index is the vocabulary's
    own position, so the positional embedding selected is identical to the one
    upstream would select.
    """
    out = [0]
    for name in ch_names:
        key = str(name).strip()
        index = _VOCAB_BY_UPPER.get(key.upper())
        if index is None:
            raise ChannelNotInVocabularyError(
                f"electrode {name!r} (normalized {key.upper()!r}) is not in LaBraM's "
                "vocabulary; it has no pretrained positional embedding"
            )
        out.append(index + 1)
    return out


@dataclass
class LaBraMFeatures:
    """Frozen embeddings plus the provenance needed to reproduce them."""

    features: np.ndarray  # (trials, embed_dim)
    ch_names: tuple[str, ...]
    input_chans: list[int]
    n_patches: int
    provenance: dict


def load_labram(
    checkpoint: str | Path | None = None,
    root: str | Path | None = None,
    device: str = "cpu",
    *,
    verify_vocab: bool = True,
    pretrained: bool = True,
):
    """Instantiate the encoder and load pretrained weights.

    Returns ``(model, load_report)``. The report records which tensors were
    missing or discarded so an unnoticed partial load cannot masquerade as a
    pretrained model.

    ``pretrained=False`` returns the identical architecture at its random
    initialization. It is the control for a negative transfer result: it
    separates "the pretrained weights carry nothing for this task" from "this
    architecture cannot be trained on a few hundred trials".
    """
    import sys

    root_path = labram_root(root)
    if verify_vocab:
        verify_vendored_vocab(root_path)
    if str(root_path) not in sys.path:
        sys.path.insert(0, str(root_path))

    import modeling_finetune  # noqa: F401 - registers labram_* with timm
    from timm.models import create_model

    model = create_model(
        "labram_base_patch200_200", pretrained=False, num_classes=0, **LABRAM_MODEL_KWARGS
    )
    if not pretrained:
        model.eval().to(device)
        return model, {
            "checkpoint": "random_init",
            "pretrained": False,
            "missing": [],
            "discarded": [],
            "n_parameters": int(sum(p.numel() for p in model.parameters())),
        }

    ckpt_path = Path(checkpoint) if checkpoint else root_path / DEFAULT_CHECKPOINT
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    tensors = state.get("model", state)
    tensors = {
        (k[len("student.") :] if k.startswith("student.") else k): v for k, v in tensors.items()
    }
    missing, unexpected = model.load_state_dict(tensors, strict=False)

    # fc_norm is the mean-pooling head; pretraining used the CLS/lm_head path, so
    # it is legitimately absent. nn.LayerNorm initializes to weight=1, bias=0,
    # which is plain normalization -- deterministic, not random.
    unexplained = [k for k in missing if not k.startswith("fc_norm.")]
    if unexplained:
        raise RuntimeError(f"unexpected missing pretrained tensors: {unexplained}")

    model.eval().to(device)
    report = {
        "checkpoint": str(ckpt_path),
        "missing": list(missing),
        "unexpected": list(unexpected),
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "model_kwargs": dict(LABRAM_MODEL_KWARGS),
    }
    return model, report


@torch.no_grad()
def extract_features(
    model,
    X_microvolts: np.ndarray,
    ch_names,
    *,
    patch_samples: int = 200,
    device: str = "cpu",
    batch_size: int = 64,
) -> LaBraMFeatures:
    """Frozen forward pass over epochs already converted to microvolts.

    ``X_microvolts`` is ``(trials, channels, samples)`` with ``samples`` an exact
    multiple of ``patch_samples`` -- use ``fm_adapters.prepare_for_fm`` first.
    """
    X = np.asarray(X_microvolts, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError(f"expected (trials, channels, samples), got {X.shape}")
    n_trials, n_channels, n_samples = X.shape
    if len(ch_names) != n_channels:
        raise ValueError(f"{len(ch_names)} channel names for {n_channels} channels")
    if n_samples % patch_samples:
        raise ValueError(
            f"{n_samples} samples is not a multiple of the {patch_samples}-sample patch"
        )
    n_patches = n_samples // patch_samples
    input_chans = channel_indices(ch_names)

    outputs = []
    for start in range(0, n_trials, batch_size):
        batch = torch.from_numpy(X[start : start + batch_size]).to(device)
        batch = batch / LABRAM_AMPLITUDE_DIVISOR
        batch = batch.reshape(batch.shape[0], n_channels, n_patches, patch_samples)
        feats = model.forward_features(batch, input_chans=input_chans)
        outputs.append(feats.detach().cpu().numpy())

    features = np.concatenate(outputs, axis=0).astype(np.float32)
    return LaBraMFeatures(
        features=features,
        ch_names=tuple(str(c) for c in ch_names),
        input_chans=input_chans,
        n_patches=n_patches,
        provenance={
            "amplitude_divisor": LABRAM_AMPLITUDE_DIVISOR,
            "input_units": "uV",
            "patch_samples": int(patch_samples),
            "n_patches": int(n_patches),
            "n_channels": int(n_channels),
            "feature_dim": int(features.shape[1]),
            "pooling": "mean" if LABRAM_MODEL_KWARGS["use_mean_pooling"] else "cls",
        },
    )
