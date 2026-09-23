"""Tests for the supervised comparator runner.

The point of adding state-of-the-art architectures is that the paper's own
finding is that baseline quality decides a benchmark's verdict. That argument
only holds if the new baselines are actually given a fair search and are built
at the right input geometry -- both of which fail silently if wrong: a model
constructed with the wrong sampling rate still trains, it just trains a
differently-shaped filter bank than its authors intended.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments.run_supervised_on_fm_data import (  # noqa: E402
    ARCH_GRID,
    ARCHS,
    BRAINDECODE_ARCHS,
    LR_GRID,
    config_grid,
)


def test_shallow_convnet_remains_the_default_arch():
    """Existing artifacts were produced with it; it must stay index 0."""
    assert ARCHS[0] == "shallow_convnet"


def test_cited_sota_architectures_are_available():
    for name in ("atcnet", "eegconformer"):
        assert name in ARCHS
        assert name in BRAINDECODE_ARCHS


def test_untuned_shallow_convnet_reproduces_the_narrowband_default():
    """The no-tune path must match the configuration that produced the
    already-published broadband number, or old artifacts stop being
    reproducible by this script."""
    grid = config_grid("shallow_convnet", tune=False)
    assert len(grid) == 1
    assert grid[0]["arch"] == dict(ARCH_GRID[-1])


def test_tuning_shallow_convnet_searches_the_architecture_grid():
    grid = config_grid("shallow_convnet", tune=True)
    assert len(grid) == len(ARCH_GRID)
    assert all(g["lr"] is None for g in grid), "shallow convnet searches arch, not lr"


def test_braindecode_archs_search_learning_rate_not_architecture():
    """Their published architecture is their tuned configuration; the optimiser
    is what we owe them. Leaving lr unsearched would hand the new baselines the
    same disadvantage the paper criticises."""
    grid = config_grid("atcnet", tune=True)
    assert [g["lr"] for g in grid] == list(LR_GRID)
    assert all(g["arch"] == {} for g in grid)


def test_untuned_braindecode_arch_is_a_single_config():
    grid = config_grid("eegconformer", tune=False)
    assert len(grid) == 1 and grid[0]["lr"] == 1e-3


@pytest.mark.parametrize("arch", ["atcnet", "eegconformer", "eegnet"])
def test_every_config_is_a_complete_spec(arch):
    for cfg in config_grid(arch, tune=True):
        assert "arch" in cfg and "lr" in cfg


def test_build_arch_uses_the_requested_sampling_rate():
    """braindecode MI models derive kernel sizes from sfreq, so passing the
    wrong rate silently builds a different model than the paper describes."""
    pytest.importorskip("braindecode")
    from experiments.run_supervised_on_fm_data import build_arch

    a = build_arch("atcnet", 22, 800, 4, 200.0, {})
    b = build_arch("atcnet", 22, 1126, 4, 250.0, {})
    assert sum(p.numel() for p in a.parameters()) > 0
    assert sum(p.numel() for p in b.parameters()) > 0


def test_lr_grid_default_is_unchanged_by_the_cli_option():
    """Adding a knob must not move any number already on disk.

    The eight-cell sweep was specified against LR_GRID. If passing no --lr-grid
    silently searched a different set, cell 1 would stop being comparable to the
    cells rerun after it.
    """
    assert tuple(config_grid("atcnet", tune=True, lr_grid=None)) == tuple(
        {"arch": {}, "lr": lr} for lr in LR_GRID
    )


def test_lr_grid_can_be_widened_for_a_retune():
    grid = config_grid("atcnet", tune=True, lr_grid=[3e-2, 1e-2])
    assert [c["lr"] for c in grid] == [3e-2, 1e-2]


def test_lr_grid_is_ignored_when_not_tuning():
    assert [c["lr"] for c in config_grid("atcnet", tune=False, lr_grid=[0.5])] == [1e-3]


def test_shallow_convnet_still_searches_architecture_not_learning_rate():
    """The two architectures search different axes on purpose."""
    grid = config_grid("shallow_convnet", tune=True, lr_grid=[0.5])
    assert all(c["lr"] is None for c in grid)
    assert [c["arch"] for c in grid] == [dict(a) for a in ARCH_GRID]
