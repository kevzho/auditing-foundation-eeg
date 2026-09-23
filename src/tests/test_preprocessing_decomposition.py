"""Tests for the band-versus-model decomposition.

The risk this guards is the failure documented in docs section 8h: the split
between "the filter band did it" and "the model did it" is decided by how well
the broadband baseline is tuned, and an under-tuned arm sitting next to a tuned
one on disk would quietly halve the model's share. It also guards against
reporting shares when there is no deficit to apportion, where the ratio is a
small number divided by a smaller one and means nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import make_preprocessing_decomposition as D  # noqa: E402


def arm(key: str, accs: list[float], dataset: str = "bci4_2a") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": dataset,
            "subject": [str(i + 1) for i in range(len(accs))],
            "experiment_key": key,
            "heldout_accuracy": accs,
        }
    )


def supervised(*frames: pd.DataFrame) -> pd.DataFrame:
    # Deliberately the production annotator rather than a copy of it: a test
    # helper that derived arch/band/recipe independently would keep passing
    # after the real parser drifted.
    return D.annotate_arms(pd.concat(frames, ignore_index=True))


def test_pick_arm_prefers_the_tuned_variant():
    """An untuned arm beside a tuned one must never be the reported baseline."""
    part = supervised(
        arm("shallow_convnet_broadband", [0.50] * 5),
        arm("shallow_convnet_broadband_tuned", [0.60] * 5),
    )
    chosen = D.pick_arm(part[part["band"] == "broadband"])
    assert chosen["experiment_key"].unique().tolist() == ["shallow_convnet_broadband_tuned"]


def test_pick_arm_falls_back_when_no_tuned_arm_exists():
    part = supervised(arm("atcnet_broadband", [0.55] * 5))
    chosen = D.pick_arm(part[part["band"] == "broadband"])
    assert chosen["experiment_key"].unique().tolist() == ["atcnet_broadband"]


def test_shares_partition_the_total_gap():
    sup = supervised(
        arm("shallow_convnet_narrowband", [0.70] * 9),
        arm("shallow_convnet_broadband_tuned", [0.60] * 9),
    )
    fm = D._norm(arm("cbramod_frozen_p2", [0.40] * 9))
    out = D.decompose(sup, fm)
    row = out.iloc[0]
    assert row["total_gap"] == pytest.approx(0.30)
    assert row["band_gap"] + row["model_gap"] == pytest.approx(row["total_gap"])
    assert row["band_share"] + row["model_share"] == pytest.approx(1.0)
    assert row["model_share"] == pytest.approx(2 / 3)


def test_untuned_baseline_would_have_reported_a_larger_band_share():
    """The section 8h lesson, as an assertion rather than a paragraph."""
    tuned = supervised(
        arm("shallow_convnet_narrowband", [0.70] * 9),
        arm("shallow_convnet_broadband_tuned", [0.60] * 9),
    )
    untuned = supervised(
        arm("shallow_convnet_narrowband", [0.70] * 9),
        arm("shallow_convnet_broadband", [0.50] * 9),
    )
    fm = D._norm(arm("cbramod_frozen_p2", [0.40] * 9))
    assert D.decompose(untuned, fm).iloc[0]["band_share"] > D.decompose(tuned, fm).iloc[0]["band_share"]


def test_shares_are_withheld_when_the_baseline_does_not_lead():
    """No deficit means no deficit to apportion."""
    sup = supervised(
        arm("atcnet_narrowband", [0.40] * 9),
        arm("atcnet_broadband", [0.45] * 9),
    )
    fm = D._norm(arm("cbramod_finetune_sel", [0.60] * 9))
    row = D.decompose(sup, fm).iloc[0]
    assert not row["shares_meaningful"]
    assert pd.isna(row["band_share"]) and pd.isna(row["model_share"])


def test_architecture_without_both_bands_is_skipped():
    sup = supervised(arm("eegconformer_broadband", [0.55] * 9))
    fm = D._norm(arm("cbramod_frozen_p2", [0.40] * 9))
    assert D.decompose(sup, fm).empty


def test_decomposition_is_computed_per_architecture_and_dataset():
    sup = supervised(
        arm("shallow_convnet_narrowband", [0.70] * 9),
        arm("shallow_convnet_broadband_tuned", [0.60] * 9),
        arm("atcnet_narrowband", [0.80] * 9),
        arm("atcnet_broadband_tuned", [0.75] * 9),
        arm("atcnet_narrowband", [0.72] * 9, dataset="bnci2014_004"),
        arm("atcnet_broadband_tuned", [0.68] * 9, dataset="bnci2014_004"),
    )
    fm = D._norm(
        pd.concat(
            [arm("cbramod_frozen_p2", [0.40] * 9), arm("cbramod_frozen_p2", [0.50] * 9, dataset="bnci2014_004")],
            ignore_index=True,
        )
    )
    out = D.decompose(sup, fm)
    assert set(zip(out["dataset"], out["arch"])) == {
        ("bci4_2a", "shallow_convnet"),
        ("bci4_2a", "atcnet"),
        ("bnci2014_004", "atcnet"),
    }


def test_subjects_are_intersected_across_all_three_arms():
    """A comparator missing a subject must shrink the pairing, not drop rows."""
    sup = supervised(
        arm("atcnet_narrowband", [0.70] * 9),
        arm("atcnet_broadband_tuned", [0.60] * 5),
    )
    fm = D._norm(arm("cbramod_frozen_p2", [0.40] * 7))
    assert D.decompose(sup, fm).iloc[0]["n"] == 5


def test_random_init_controls_are_not_treated_as_foundation_models(tmp_path, monkeypatch):
    """The control is an ablation of the model, not a comparator for the band."""
    monkeypatch.setattr(D, "FM_DIR", tmp_path)
    monkeypatch.setattr(D, "SOTA_DIR", tmp_path / "absent")
    arm("cbramod_frozen_p2", [0.40] * 9).to_csv(tmp_path / "a_subject_metrics.csv", index=False)
    arm("cbramod_frozen_p2_randinit", [0.30] * 9).to_csv(tmp_path / "b_subject_metrics.csv", index=False)
    keys = set(D.load_fm_arms()["experiment_key"])
    assert keys == {"cbramod_frozen_p2"}


def test_cross_pipeline_arm_is_never_used_as_a_decomposition_baseline():
    """The band term must not absorb a change of training recipe.

    The imported ``baseline`` arm is ``baseline_ce_cropped_shallow_convnet``:
    multi-scale crops with aggregation, which is an augmentation scheme as much
    as a training loop. Pairing it against a single-window broadband arm would
    put cropping inside the number labelled "filter band".
    """
    part = supervised(
        arm("shallow_convnet_narrowband_cropped", [0.70] * 9),
        arm("shallow_convnet_narrowband_tuned", [0.63] * 9),
    )
    chosen = D.pick_arm(part[part["band"] == "narrowband"])
    assert chosen["experiment_key"].unique().tolist() == ["shallow_convnet_narrowband_tuned"]


def test_a_cross_pipeline_only_band_yields_no_decomposition():
    """Better to report nothing than to report a contaminated split."""
    sup = supervised(
        arm("shallow_convnet_narrowband_cropped", [0.70] * 9),
        arm("shallow_convnet_broadband_tuned", [0.60] * 9),
    )
    fm = D._norm(arm("cbramod_frozen_p2", [0.40] * 9))
    assert D.decompose(sup, fm).empty


def test_imported_baseline_keeps_its_pipeline_marker(tmp_path, monkeypatch):
    """Renaming it to a bare arm name is what made the confound invisible."""
    monkeypatch.setattr(D, "FM_DIR", tmp_path / "absent")
    monkeypatch.setattr(D, "SOTA_DIR", tmp_path / "absent")
    csv = tmp_path / "comparison.csv"
    arm("baseline", [0.70] * 9).to_csv(csv, index=False)
    monkeypatch.setattr(D, "COMPARISON_CSV", csv)
    arms = D.load_supervised_arms()
    assert arms["experiment_key"].tolist() == [D.NATIVE_IMPORT_KEY] * 9
    assert arms["cross_pipeline"].all()


def test_key_parsing_survives_an_unanticipated_tag():
    """A retuned arm must not silently vanish from the output.

    Suffix-stripping by regex mis-parses the first key carrying a tag the
    pattern never anticipated: `arch` becomes the whole string, the arm pairs
    with nothing, and it disappears without an error.
    """
    got = D.parse_key("atcnet_narrowband_retune_tuned")
    assert got == {
        "arch": "atcnet",
        "band": "narrowband",
        "tuned": True,
        "cross_pipeline": False,
        "variant": "retune",
    }


def test_key_parsing_handles_every_shape_on_disk():
    cases = {
        "shallow_convnet_broadband": ("shallow_convnet", "broadband", False, False, ""),
        "shallow_convnet_broadband_tuned": ("shallow_convnet", "broadband", True, False, ""),
        "shallow_convnet_narrowband_cropped": ("shallow_convnet", "narrowband", False, True, ""),
        "atcnet_narrowband_tuned": ("atcnet", "narrowband", True, False, ""),
        "eegconformer_broadband_ica_tuned": ("eegconformer", "broadband", True, False, "ica"),
    }
    for key, (arch, band, tuned, cross, variant) in cases.items():
        got = D.parse_key(key)
        assert (got["arch"], got["band"], got["tuned"], got["cross_pipeline"], got["variant"]) == (
            arch, band, tuned, cross, variant
        ), key


def test_a_retuned_arm_never_pairs_with_the_arm_it_replaces():
    """Mixing variants would decompose a grid change as if it were a band change."""
    sup = supervised(
        arm("atcnet_narrowband_tuned", [0.58] * 9),
        arm("atcnet_broadband_tuned", [0.66] * 9),
        arm("atcnet_narrowband_retune_tuned", [0.70] * 9),
        arm("atcnet_broadband_retune_tuned", [0.72] * 9),
    )
    fm = D._norm(arm("cbramod_finetune_sel", [0.48] * 9))
    out = D.decompose(sup, fm)
    by_variant = dict(zip(out["variant"], zip(out["acc_narrowband"], out["acc_broadband"])))
    assert by_variant["base"] == pytest.approx((0.58, 0.66))
    assert by_variant["retune"] == pytest.approx((0.70, 0.72))
