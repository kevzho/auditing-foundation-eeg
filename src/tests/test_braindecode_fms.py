"""Tests for the braindecode foundation-model adapter.

The failure modes worth catching here are silent ones. A wrong layer index still
trains, it just applies the wrong learning rate to the wrong depth. A pooling
that collapses the wrong axes still returns a feature matrix of plausible shape.
An amplitude divisor that drifts from LaBraM's turns a cross-model comparison
into a comparison of preprocessing.

Anything requiring the pretrained checkpoints is skipped when the network or
braindecode is unavailable, so the suite stays runnable offline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from models.braindecode_fms import (  # noqa: E402
    BD_FM_SPECS,
    FM_AMPLITUDE_DIVISOR,
    UnsupportedFMError,
    as_fm_spec,
    build_chs_info,
    get_spec,
    layer_id,
    pool,
)
from models.labram_probe import LABRAM_AMPLITUDE_DIVISOR  # noqa: E402

braindecode = pytest.importorskip("braindecode", reason="braindecode not installed")


# --- registry ---------------------------------------------------------------


def test_unknown_model_raises_with_the_registered_names():
    with pytest.raises(UnsupportedFMError) as excinfo:
        get_spec("not-a-model")
    assert "cbramod" in str(excinfo.value)


def test_amplitude_divisor_matches_labram():
    """CBraMod's loader returns data/100, exactly as LaBraM's does.

    If these ever diverge, a cross-model comparison silently becomes a
    comparison of input scaling.
    """
    assert FM_AMPLITUDE_DIVISOR == LABRAM_AMPLITUDE_DIVISOR
    for spec in BD_FM_SPECS.values():
        assert spec.amplitude_divisor == LABRAM_AMPLITUDE_DIVISOR


def test_as_fm_spec_preserves_the_input_contract():
    spec = get_spec("cbramod")
    fm = as_fm_spec(spec)
    assert fm.sfreq == spec.sfreq
    assert fm.patch_samples == spec.patch_samples
    assert fm.bandpass == spec.bandpass
    assert fm.notch == spec.notch
    assert fm.units == "uV"


# --- layer depth ------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("patch_embedding.proj.weight", 0),
        ("encoder.layers.0.self_attn_s.in_proj_weight", 1),
        ("encoder.layers.11.norm2.bias", 12),
        ("proj_out.0.weight", 13),
        ("final_layer.1.weight", 13),
    ],
)
def test_layer_id_assignment(name, expected):
    assert layer_id(name, n_blocks=12) == expected


def test_head_sits_strictly_above_every_block():
    n_blocks = 12
    head = layer_id("final_layer.1.weight", n_blocks)
    assert all(layer_id(f"encoder.layers.{i}.norm1.weight", n_blocks) < head for i in range(n_blocks))


# --- pooling ----------------------------------------------------------------


def test_mean_pooling_collapses_channels_and_patches_only():
    h = torch.randn(3, 22, 4, 200)
    out = pool(h, "mean")
    assert out.shape == (3, 200)
    assert torch.allclose(out, h.mean(dim=(1, 2)))


def test_flatten_pooling_keeps_every_element():
    h = torch.randn(3, 22, 4, 200)
    out = pool(h, "flatten")
    assert out.shape == (3, 22 * 4 * 200)


def test_unknown_pooling_raises():
    with pytest.raises(ValueError):
        pool(torch.randn(2, 3, 4, 5), "max")


# --- montage ----------------------------------------------------------------


def test_build_chs_info_gives_every_channel_a_position():
    pytest.importorskip("mne")
    from eeg_montage import BCI4_2A_CHANNELS

    chs = build_chs_info(BCI4_2A_CHANNELS, 200.0)
    assert len(chs) == len(BCI4_2A_CHANNELS)
    for c in chs:
        assert np.any(np.asarray(c["loc"][:3], dtype=float)), c["ch_name"]


def test_build_chs_info_rejects_an_unknown_electrode():
    pytest.importorskip("mne")
    with pytest.raises(Exception):
        build_chs_info(["FZ", "NOT_AN_ELECTRODE"], 200.0)
