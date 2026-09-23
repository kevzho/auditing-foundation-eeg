"""Tests for MOABB broadband preprocessing.

The failure modes here are silent ones. A session split that picks the wrong
sessions still produces a well-formed npz -- it just quietly turns a
cross-session protocol into a within-session one, which is the exact evaluation
weakness this project criticises in the literature. A dataset registered with
the wrong class count is rejected loudly by MOABB, so that one is cheap to
guard too.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from preprocess_fm_moabb import (  # noqa: E402
    DATASET_CLASSES,
    DATASET_KWARGS,
    DATASETS_NEEDING_SESSION_UNFILTER,
    dataset_slug,
    split_sessions,
)


# --- session splitting ------------------------------------------------------


def test_named_sessions_split_on_train_and_test():
    sessions = np.array(["0train", "1train", "2train", "3test", "4test"])
    train, evaluation = split_sessions(sessions)
    assert list(train) == [True, True, True, False, False]
    assert list(evaluation) == [False, False, False, True, True]


def test_numeric_sessions_split_chronologically():
    """Lee2019_MI labels sessions '0' and '1' with no train/test in the name.

    The correct reading for a session-shift study is first day calibrates, last
    day is held out -- not the run-level train/test split inside a session,
    which would be a within-session evaluation.
    """
    sessions = np.array(["0"] * 200 + ["1"] * 200)
    train, evaluation = split_sessions(sessions)
    assert train.sum() == 200 and evaluation.sum() == 200
    assert train[:200].all() and evaluation[200:].all()
    assert not (train & evaluation).any(), "a trial cannot be in both splits"


def test_numeric_sessions_use_first_and_last_only():
    sessions = np.array(["0", "1", "2"])
    train, evaluation = split_sessions(sessions)
    assert list(train) == [True, False, False]
    assert list(evaluation) == [False, False, True]


def test_single_session_is_refused():
    """One session cannot yield a held-out day; failing loudly beats guessing."""
    with pytest.raises(SystemExit):
        split_sessions(np.array(["1", "1", "1"]))


def test_splits_are_disjoint_for_named_sessions():
    sessions = np.array(["0train", "3test"])
    train, evaluation = split_sessions(sessions)
    assert not (train & evaluation).any()


# --- dataset registry -------------------------------------------------------


def test_lee2019_is_registered_as_two_class():
    """MOABB rejects the dataset outright if this is wrong.

    MotorImagery(n_classes=4) against a 2-class set raises "not valid for
    paradigm" rather than silently mislabelling, so this guards a real failure.
    """
    assert DATASET_CLASSES["Lee2019_MI"] == 2


def test_lee2019_requests_both_runs():
    """Default test_run is None for MI, which exposes only the training run."""
    kwargs = DATASET_KWARGS["Lee2019_MI"]
    assert kwargs["train_run"] is True
    assert kwargs["test_run"] is True


def test_lee2019_needs_the_session_filter_cleared():
    """Its loader emits 0-indexed session names while the filter is 1-indexed.

    Without clearing it, get_data returns one session of two and says nothing.
    """
    assert "Lee2019_MI" in DATASETS_NEEDING_SESSION_UNFILTER


@pytest.mark.parametrize(
    "name,expected",
    [("BNCI2014_004", "bnci2014_004"), ("Lee2019_MI", "lee2019_mi")],
)
def test_dataset_slug_is_lowercased(name, expected):
    assert dataset_slug(name) == expected


def test_every_specially_handled_dataset_declares_its_class_count():
    """A dataset can only be misclassified silently once.

    ``n_classes`` falls back to 4 for any dataset missing from
    ``DATASET_CLASSES``, which would have turned Lee2019_MI's two classes into a
    four-class paradigm and produced empty or wrong epochs rather than an error.
    Anything already needing special handling is a dataset someone added on
    purpose, so it has no excuse for relying on that default.
    """
    special = set(DATASET_KWARGS) | set(DATASETS_NEEDING_SESSION_UNFILTER)
    assert special <= set(DATASET_CLASSES)


def test_dataset_labels_cover_every_preprocessable_dataset():
    """The runners label results from a registry the preprocessor never sees.

    A dataset preprocessed under a name the runners cannot label would fall back
    to its own slug, and the results CSV would disagree with every other arm on
    what the dataset is called -- the class of filename mismatch that has
    already broken analysis scripts in this project more than once.
    """
    from experiments.run_fm_probe import DATASET_LABELS

    assert set(DATASET_CLASSES) <= set(DATASET_LABELS.values())
