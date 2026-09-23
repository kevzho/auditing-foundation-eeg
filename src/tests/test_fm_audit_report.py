"""Tests for the consolidated report generator.

The risk this guards is a comparator that exists on disk but never reaches a
table. The paper's own finding is that baseline quality decides a benchmark's
verdict, so a state-of-the-art comparator silently omitted from the
FM-vs-supervised tests would understate the gap and no error would be raised.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import make_fm_audit_report as R  # noqa: E402


def test_sota_directory_is_a_declared_source():
    """SOTA comparators live outside the probe directory and must be loaded."""
    assert R.SOTA_DIR == Path("results") / "supervised_sota"


def test_load_local_supervised_returns_empty_frame_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "SOTA_DIR", tmp_path / "nothing_here")
    assert R.load_local_supervised().empty


def test_load_local_supervised_normalises_join_keys(tmp_path, monkeypatch):
    """Dataset case and subject dtype must match the foundation-model rows.

    The supervised CSV spells MOABB datasets 'BNCI2014_004' while the probe
    runners use the lowercase slug, and subject ids are int in one file and str
    in another. Either mismatch silently yields an empty intersection and no
    paired tests at all -- which has happened before in this project.
    """
    monkeypatch.setattr(R, "SOTA_DIR", tmp_path)
    pd.DataFrame(
        {
            "dataset": ["BNCI2014_004", "BNCI2014_004"],
            "subject": [1, 2],
            "experiment_key": ["atcnet_broadband", "atcnet_broadband"],
            "heldout_accuracy": [0.7, 0.8],
        }
    ).to_csv(tmp_path / "supervised_atcnet_broadband_subject_metrics.csv", index=False)

    df = R.load_local_supervised()
    assert list(df["dataset"].unique()) == ["bnci2014_004"]
    assert df["subject"].map(type).eq(str).all()


@pytest.mark.parametrize(
    "key,expected",
    [
        ("cbramod_finetune_sel", "foundation model"),
        ("cbramod_finetune_randinit_sel", "random-init control"),
        ("atcnet_narrowband_tuned", "supervised"),
        ("eegconformer_broadband_tuned", "supervised"),
        ("shallow_convnet_broadband", "supervised"),
    ],
)
def test_sota_architectures_are_classified_as_supervised(key, expected):
    """A comparator misfiled as a foundation model would be tested against
    itself and inflate the apparent number of significant results."""
    assert R.role_of(key) == expected


def test_cropped_reference_arms_are_labelled_not_dropped():
    """The strongest supervised arm has a different training recipe.

    Dropping it would understate what the foundation models must beat; leaving
    it unlabelled invites the reader to attribute a recipe difference to the
    architecture, which is the confound that invalidated the section 8h band
    term. It has to appear, and it has to say what it is.
    """
    assert R.recipe_of("baseline") == "cropped+aggregated"
    assert R.recipe_of("seed_ensemble") == "cropped+aggregated"
    assert R.recipe_of("atcnet_narrowband_tuned") == "single-window"
    assert R.recipe_of("shallow_convnet_broadband_tuned") == "single-window"


def test_recipe_is_blank_for_arms_that_are_not_supervised():
    """A foundation model has no supervised training recipe to report."""
    assert R.recipe_of("cbramod_finetune_sel") == ""
    assert R.recipe_of("cbramod_finetune_sel_randinit") == ""
