"""Tests for the foundation-model probe runner's training plumbing.

These cover the parts that fail silently: a layer-decay schedule that assigns the
wrong depth still trains, it just destroys pretrained features; a weight-decay
rule that catches norm parameters still converges, just worse.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments.run_fm_probe import (  # noqa: E402
    LaBraMClassifier,
    _layer_id,
    build_param_groups,
    encode_labels,
    freeze_lower,
    probe_logits,
)


class _FakeEncoder(nn.Module):
    """Minimal stand-in with LaBraM's parameter naming, so no checkpoint is needed."""

    def __init__(self, n_blocks: int = 3, dim: int = 8):
        super().__init__()
        self.patch_embed = nn.Linear(dim, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 4, dim))
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim)) for _ in range(n_blocks)
        )
        self.norm = nn.LayerNorm(dim)

    def forward_features(self, x, input_chans=None):
        return x.mean(dim=(1, 2))


# --- layer depth assignment -------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("encoder.patch_embed.weight", 0),
        ("encoder.cls_token", 0),
        ("encoder.pos_embed", 0),
        ("encoder.time_embed", 0),
        ("encoder.blocks.0.attn.qkv.weight", 1),
        ("encoder.blocks.5.mlp.fc1.weight", 6),
        ("encoder.norm.weight", 13),
        ("head.weight", 13),
    ],
)
def test_layer_id_assignment(name, expected):
    assert _layer_id(name, n_blocks=12) == expected


def test_deeper_layers_get_larger_learning_rates():
    """Layer decay must leave embeddings slowest and the head fastest."""
    model = LaBraMClassifier(_FakeEncoder(n_blocks=3), embed_dim=8, n_classes=4)
    groups = build_param_groups(model, lr=1e-3, weight_decay=0.05, layer_decay=0.9)
    by_layer = {}
    for g in groups:
        by_layer.setdefault(g["layer_id"], g["lr"])
    layers = sorted(by_layer)
    lrs = [by_layer[layer] for layer in layers]
    assert lrs == sorted(lrs), "learning rate must increase with depth"
    assert by_layer[max(layers)] > by_layer[min(layers)]


def test_head_learning_rate_equals_base_lr():
    model = LaBraMClassifier(_FakeEncoder(n_blocks=3), embed_dim=8, n_classes=4)
    groups = build_param_groups(model, lr=1e-3, weight_decay=0.05, layer_decay=0.9)
    top = max(g["layer_id"] for g in groups)
    head = [g for g in groups if g["layer_id"] == top]
    assert all(np.isclose(g["lr"], 1e-3) for g in head)


def test_layer_decay_of_one_gives_uniform_lr():
    model = LaBraMClassifier(_FakeEncoder(n_blocks=3), embed_dim=8, n_classes=4)
    groups = build_param_groups(model, lr=1e-3, weight_decay=0.05, layer_decay=1.0)
    assert all(np.isclose(g["lr"], 1e-3) for g in groups)


# --- weight decay -----------------------------------------------------------


def test_norms_and_biases_are_excluded_from_weight_decay():
    model = LaBraMClassifier(_FakeEncoder(n_blocks=2), embed_dim=8, n_classes=4)
    groups = build_param_groups(model, lr=1e-3, weight_decay=0.05, layer_decay=0.9)
    decayed = {id(p) for g in groups if g["weight_decay"] > 0 for p in g["params"]}
    for name, param in model.named_parameters():
        if param.ndim <= 1 or name.endswith(".bias"):
            assert id(param) not in decayed, f"{name} must not be weight-decayed"


def test_every_trainable_parameter_lands_in_exactly_one_group():
    model = LaBraMClassifier(_FakeEncoder(n_blocks=2), embed_dim=8, n_classes=4)
    groups = build_param_groups(model, lr=1e-3, weight_decay=0.05, layer_decay=0.9)
    seen = [id(p) for g in groups for p in g["params"]]
    expected = [id(p) for p in model.parameters() if p.requires_grad]
    assert len(seen) == len(set(seen)), "a parameter appears in more than one group"
    assert set(seen) == set(expected)


# --- partial fine-tuning ----------------------------------------------------


def test_freeze_lower_leaves_upper_blocks_and_head_trainable():
    model = LaBraMClassifier(_FakeEncoder(n_blocks=4), embed_dim=8, n_classes=4)
    freeze_lower(model, 2)  # embeddings + blocks 0 and 1
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert not any(n.startswith("encoder.blocks.0.") for n in trainable)
    assert not any(n.startswith("encoder.blocks.1.") for n in trainable)
    assert any(n.startswith("encoder.blocks.2.") for n in trainable)
    assert any(n.startswith("encoder.blocks.3.") for n in trainable)
    assert any(n.startswith("head") for n in trainable)
    assert "encoder.patch_embed.weight" not in trainable
    assert "encoder.cls_token" not in trainable


def test_freeze_lower_zero_is_a_no_op():
    model = LaBraMClassifier(_FakeEncoder(n_blocks=3), embed_dim=8, n_classes=4)
    before = [n for n, p in model.named_parameters() if p.requires_grad]
    assert freeze_lower(model, 0) == 0
    after = [n for n, p in model.named_parameters() if p.requires_grad]
    assert before == after


def test_frozen_modules_are_exactly_the_fully_frozen_ones():
    """Dropout must not fire inside a block that cannot learn from it.

    This also avoids a hard failure: with nothing upstream requiring grad, MPS
    picks a fused attention kernel that raises NotImplementedError on dropout.
    """
    from experiments.run_fm_probe import frozen_modules

    model = LaBraMClassifier(_FakeEncoder(n_blocks=4), embed_dim=8, n_classes=4)
    freeze_lower(model, 2)
    still = frozen_modules(model)
    for module in still:
        params = list(module.parameters(recurse=True))
        assert params and all(not p.requires_grad for p in params)
    # The head trains, so neither it nor the whole model may be listed.
    assert model not in still
    assert model.head not in still
    # Frozen block 0 must be listed; trainable block 3 must not.
    assert model.encoder.blocks[0] in still
    assert model.encoder.blocks[3] not in still


def test_no_modules_frozen_when_nothing_is_frozen():
    from experiments.run_fm_probe import frozen_modules

    model = LaBraMClassifier(_FakeEncoder(n_blocks=3), embed_dim=8, n_classes=4)
    assert frozen_modules(model) == []


def test_param_groups_skip_frozen_parameters():
    """A frozen tensor handed to AdamW would still be decayed; it must not appear."""
    model = LaBraMClassifier(_FakeEncoder(n_blocks=4), embed_dim=8, n_classes=4)
    freeze_lower(model, 2)
    groups = build_param_groups(model, lr=1e-3, weight_decay=0.05, layer_decay=0.9)
    grouped = {id(p) for g in groups for p in g["params"]}
    frozen = {id(p) for p in model.parameters() if not p.requires_grad}
    assert grouped.isdisjoint(frozen)
    assert grouped == {id(p) for p in model.parameters() if p.requires_grad}


# --- label handling ---------------------------------------------------------


def test_encode_labels_maps_to_contiguous_indices():
    train = np.array([769, 770, 771, 772, 769])
    classes, (y_tr, y_ev) = encode_labels(train, np.array([772, 769]))
    assert list(classes) == [769, 770, 771, 772]
    assert list(y_tr) == [0, 1, 2, 3, 0]
    assert list(y_ev) == [3, 0]


def test_encode_labels_shares_one_mapping_across_splits():
    """Held-out labels must use the training mapping, not their own."""
    classes, (y_tr, y_ev) = encode_labels(np.array([2, 1, 0]), np.array([0, 2]))
    assert list(classes) == [0, 1, 2]
    assert list(y_ev) == [0, 2]


# --- probe logits -----------------------------------------------------------


class _BinaryDummy:
    def decision_function(self, X):
        return np.array([0.5, -1.5])


def test_binary_decision_function_is_expanded_to_two_columns():
    """sklearn returns one margin for binary problems; metrics need (n, 2)."""
    out = probe_logits(np.zeros((2, 3)), _BinaryDummy())
    assert out.shape == (2, 2)
    assert np.allclose(out[:, 1] - out[:, 0], np.array([1.0, -3.0]))
