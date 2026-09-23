"""LaBraM electrode lookup must agree with upstream on every vocabulary entry.

Six of the 136 vendored names carry a lowercase 10-05 suffix (FTT9h, TTP7h,
TPP9h, FTT10h, TPP8h, TPP10h). An uppercase-then-exact-match lookup made those
six unreachable *even when spelled exactly as the vocabulary spells them*,
which surfaced as "LaBraM cannot be evaluated on Lee2019_MI" rather than as a
bug -- every one of that dataset's 54 subjects carries all six.

The regression guarded here is silent scope loss: a lookup failure that reads
like a property of the dataset costs a comparison without ever looking wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from models.labram_probe import (  # noqa: E402
    STANDARD_1020,
    ChannelNotInVocabularyError,
    channel_indices,
)

MIXED_CASE = ("FTT9h", "TTP7h", "TPP9h", "FTT10h", "TPP8h", "TPP10h")


def test_every_vocabulary_name_resolves_to_its_upstream_index():
    """Upstream does ``standard_1020.index(name) + 1`` with no normalisation.

    Checking all 136 rather than a sample: the bug affected exactly the entries
    nobody thought to sample.
    """
    for position, name in enumerate(STANDARD_1020):
        assert channel_indices([name]) == [0, position + 1], name


def test_mixed_case_entries_are_reachable():
    assert set(MIXED_CASE) <= set(STANDARD_1020), "vendored vocabulary drifted"
    for name in MIXED_CASE:
        assert channel_indices([name]) == [0, STANDARD_1020.index(name) + 1]


@pytest.mark.parametrize("spelling", ["Fz", "FZ", "fz", " Fz "])
def test_lookup_is_case_and_whitespace_insensitive(spelling):
    assert channel_indices([spelling]) == channel_indices(["FZ"])


def test_cls_token_occupies_index_zero():
    assert channel_indices(["FZ", "CZ"])[0] == 0


def test_genuine_out_of_vocabulary_names_still_raise():
    """Case-insensitivity must not degrade into silently accepting anything.

    A mis-mapped electrode is worse than a refused one: it selects a real but
    wrong positional embedding, and nothing downstream can detect it.
    """
    with pytest.raises(ChannelNotInVocabularyError):
        channel_indices(["NOTANELECTRODE"])
