"""Tests for the configured default paths.

These guard a failure the python-layout reorg introduced and nothing caught for
weeks: moving `config.py` into `src/` silently redirected every default derived
from `BASE_DIR`, so `DATA_DIR` became `src/data/BCICIV_2a_gdf`. No script
complained, because the arrays those scripts write had already been generated
under the old layout and the defaults were never exercised again -- until a new
control needed to re-read the raw recordings and failed with a path that had
never existed.

A path constant that points nowhere is not an error until someone uses it, which
is exactly why it needs a test rather than a runtime check.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import config  # noqa: E402


def test_base_dir_is_the_repository_root_not_the_package():
    """`src/` must be a child of BASE_DIR, never BASE_DIR itself."""
    assert (config.BASE_DIR / "src").is_dir()
    assert config.BASE_DIR.name != "src"


def test_input_paths_resolve_to_directories_that_exist():
    """Inputs must exist; a default that cannot be read is not a default."""
    assert config.DATA_DIR.is_dir(), config.DATA_DIR
    assert config.EPOCH_DIR.is_dir(), config.EPOCH_DIR


def test_output_paths_are_inside_the_repository():
    """Outputs need not exist yet, but must not land outside the checkout."""
    for path in (config.OUTPUT_DIR, config.EPOCH_DIR, config.DATA_DIR):
        assert config.BASE_DIR in path.parents or path == config.BASE_DIR, path
