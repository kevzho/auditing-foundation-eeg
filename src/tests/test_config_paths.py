"""Tests for repository-relative default paths."""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import config  # noqa: E402


def test_base_dir_is_the_repository_root_not_the_package():
    assert (config.BASE_DIR / "src").is_dir()
    assert config.BASE_DIR.name != "src"


def test_input_paths_resolve_from_repository_root():
    assert config.DATA_DIR == config.BASE_DIR / "data" / "BCICIV_2a_gdf"
    assert config.EPOCH_DIR == config.BASE_DIR / "analysis_results"


def test_output_paths_are_inside_the_repository():
    for path in (config.OUTPUT_DIR, config.EPOCH_DIR, config.DATA_DIR):
        assert config.BASE_DIR in path.parents or path == config.BASE_DIR, path
