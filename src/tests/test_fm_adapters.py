"""Tests for channel identity and foundation-model input adapters."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eeg_montage import (  # noqa: E402
    BCI4_2A_ANCHORS,
    BCI4_2A_CHANNELS,
    BCI4_2A_GDF_NAMES,
    channels_for,
    normalize_channel_name,
    verify_bci4_2a_order,
)
from models.fm_adapters import (  # noqa: E402
    LABRAM_SPEC,
    BandMismatchError,
    check_band_compatibility,
    crop_to_patches,
    infer_units,
    prepare_for_fm,
    resample_epochs,
    to_microvolts,
)


# --- channel identity -------------------------------------------------------


def test_canonical_list_has_22_unique_channels():
    assert len(BCI4_2A_CHANNELS) == 22
    assert len(set(BCI4_2A_CHANNELS)) == 22


def test_named_gdf_channels_land_on_their_anchors():
    """The five electrodes the GDF names must map to themselves."""
    for index, expected in BCI4_2A_ANCHORS.items():
        assert BCI4_2A_CHANNELS[index] == expected
        assert normalize_channel_name(BCI4_2A_GDF_NAMES[index]) == expected


def test_verify_accepts_real_gdf_order():
    names = list(BCI4_2A_GDF_NAMES) + ["EOG-left", "EOG-central", "EOG-right"]
    verify_bci4_2a_order(names)  # must not raise


def test_verify_rejects_reordered_channels():
    """A shuffled file must fail loudly rather than mislabel electrodes."""
    names = list(BCI4_2A_GDF_NAMES)
    names[0], names[7] = names[7], names[0]  # swap Fz and C3
    with pytest.raises(ValueError, match="channel order mismatch"):
        verify_bci4_2a_order(names + ["EOG-left", "EOG-central", "EOG-right"])


def test_verify_rejects_wrong_channel_count():
    with pytest.raises(ValueError, match="expected 22 EEG channels"):
        verify_bci4_2a_order(list(BCI4_2A_GDF_NAMES[:20]))


@pytest.mark.parametrize(
    "raw,expected",
    [("EEG-C3", "C3"), ("EEG-Fz", "Fz"), ("eeg cz", "Cz"), ("POz", "POz"), ("c4.", "C4")],
)
def test_normalize_channel_name(raw, expected):
    assert normalize_channel_name(raw) == expected


def test_channels_for_rejects_channel_count_mismatch():
    with pytest.raises(ValueError, match="22 channels but array has 20"):
        channels_for("bci4_2a", 20)


def test_channels_for_rejects_unknown_dataset():
    with pytest.raises(KeyError):
        channels_for("not_a_dataset", 22)


# --- units ------------------------------------------------------------------


def test_infer_units_detects_volts_and_microvolts():
    volts = np.random.default_rng(0).normal(0, 4e-6, (4, 3, 100))
    assert infer_units(volts) == "V"
    assert infer_units(volts * 1e6) == "uV"


def test_to_microvolts_scales_volts_only():
    volts = np.full((2, 2, 4), 1e-6, dtype=np.float64)
    assert np.allclose(to_microvolts(volts, "V"), 1.0)
    assert np.allclose(to_microvolts(volts, "uV"), 1e-6)


def test_to_microvolts_rejects_unknown_units():
    with pytest.raises(ValueError, match="unrecognized units"):
        to_microvolts(np.zeros((1, 1, 4)), "millivolts")


# --- resampling and cropping ------------------------------------------------


def test_resample_changes_length_by_rate_ratio():
    X = np.random.default_rng(1).normal(size=(3, 5, 1000))
    out = resample_epochs(X, 250.0, 200.0)
    assert out.shape[:2] == (3, 5)
    assert out.shape[-1] == 800


def test_resample_is_identity_at_matching_rate():
    X = np.random.default_rng(2).normal(size=(2, 3, 64)).astype(np.float32)
    assert np.allclose(resample_epochs(X, 200.0, 200.0), X)


def test_resample_preserves_a_slow_sinusoid():
    """Downsampling must not distort content well below Nyquist."""
    t = np.arange(1000) / 250.0
    X = np.sin(2 * np.pi * 10.0 * t)[None, None, :]
    out = resample_epochs(X, 250.0, 200.0)[0, 0]
    t2 = np.arange(out.size) / 200.0
    expected = np.sin(2 * np.pi * 10.0 * t2)
    # ignore edge transients from the polyphase filter
    assert np.corrcoef(out[40:-40], expected[40:-40])[0, 1] > 0.99


def test_crop_to_whole_patches():
    X = np.zeros((2, 3, 900))
    assert crop_to_patches(X, 200).shape[-1] == 800


def test_crop_alignment_selects_different_windows():
    X = np.arange(900, dtype=np.float32)[None, None, :]
    assert crop_to_patches(X, 200, align="start")[0, 0, 0] == 0
    assert crop_to_patches(X, 200, align="end")[0, 0, 0] == 100
    assert crop_to_patches(X, 200, align="center")[0, 0, 0] == 50


def test_crop_rejects_epoch_shorter_than_one_patch():
    with pytest.raises(ValueError, match="shorter than one"):
        crop_to_patches(np.zeros((1, 1, 150)), 200)


# --- band compatibility guard ----------------------------------------------


def test_mi_band_arrays_are_rejected_for_a_broadband_model():
    """The 8-30 Hz MI arrays must not silently feed a broadband-pretrained FM."""
    with pytest.raises(BandMismatchError, match="high-passes"):
        check_band_compatibility((8.0, 30.0), LABRAM_SPEC)


def test_broadband_source_passes():
    assert check_band_compatibility((0.1, 75.0), LABRAM_SPEC) == []


def test_band_problems_can_be_reported_without_raising():
    problems = check_band_compatibility((8.0, 30.0), LABRAM_SPEC, strict=False)
    assert len(problems) == 2  # both edges violated


# --- end to end -------------------------------------------------------------


def test_prepare_for_fm_matches_labram_contract():
    rng = np.random.default_rng(3)
    X = rng.normal(0, 4e-6, (16, 22, 1126))  # volts, 250 Hz, 4.504 s
    prepared = prepare_for_fm(
        X, BCI4_2A_CHANNELS, 250.0, LABRAM_SPEC, source_bandpass=(0.1, 75.0)
    )
    assert prepared.X.shape == (16, 22, 800)  # 4 s at 200 Hz
    assert prepared.n_patches == 4
    assert prepared.sfreq == 200.0
    assert prepared.X.std() > 1.0  # microvolts, not volts
    assert prepared.provenance["source_units"] == "V"


def test_prepare_for_fm_rejects_channel_name_count_mismatch():
    X = np.zeros((4, 22, 1126))
    with pytest.raises(ValueError, match="channel names"):
        prepare_for_fm(X, BCI4_2A_CHANNELS[:20], 250.0, LABRAM_SPEC, source_bandpass=(0.1, 75.0))


def test_prepare_for_fm_rejects_wrong_dimensionality():
    with pytest.raises(ValueError, match="expected \\(trials, channels, samples\\)"):
        prepare_for_fm(np.zeros((22, 1126)), BCI4_2A_CHANNELS, 250.0, LABRAM_SPEC)
