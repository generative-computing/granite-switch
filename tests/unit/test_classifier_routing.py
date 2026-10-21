# SPDX-License-Identifier: Apache-2.0
"""Classifier-slot routing: the kind LUT and the LoRA/classifier index split."""

import torch

from granite_switch.classifier_routing import (
    build_adapter_kind_lut,
    split_adapter_indices,
)
from granite_switch.config import GraniteSwitchConfig


def _config(adapter_kinds, label_ids, base_reset=False):
    num_adapters = len(adapter_kinds)
    token_ids = list(range(500, 500 + num_adapters))
    if base_reset:
        token_ids = [499, *token_ids]
    return GraniteSwitchConfig(
        vocab_size=600,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=4,
        intermediate_size=128,
        num_adapters=num_adapters,
        adapter_token_ids=token_ids,
        adapter_names=[f"a{i}" for i in range(num_adapters)],
        adapter_substitute_token_ids=[1] * len(token_ids),
        adapter_ranks=[4] * num_adapters,
        max_lora_rank=4,
        adapter_kinds=adapter_kinds,
        classifier_label_token_ids=label_ids,
    )


class TestAdapterKindLut:
    def test_none_without_classifier_slots(self):
        assert build_adapter_kind_lut(_config(["lora", "lora"], None)) is None

    def test_marks_classifier_adapter_indices(self):
        cfg = _config(["lora", "classifier", "classifier"], [None, [100], [101, 102]])
        assert build_adapter_kind_lut(cfg).tolist() == [False, False, True, True]

    def test_base_reset_token_does_not_shift_slots(self):
        """The LUT is indexed by adapter index, not by control-token position."""
        cfg = _config(["lora", "classifier"], [None, [100, 101]], base_reset=True)
        assert build_adapter_kind_lut(cfg).tolist() == [False, False, True]
        assert cfg.classifier_control_token_ids == [501]


class TestSplitAdapterIndices:
    def test_classifier_positions_leave_the_lora_stream(self):
        cfg = _config(["lora", "classifier"], [None, [100, 101]])
        lora, classifier = split_adapter_indices(
            build_adapter_kind_lut(cfg), torch.tensor([0, 1, 1, 0, 2])
        )
        assert lora.tolist() == [0, 1, 1, 0, 0]
        assert classifier.tolist() == [0, 0, 0, 0, 2]

    def test_preserves_batched_shape(self):
        cfg = _config(["classifier", "lora"], [[100, 101], None])
        lora, classifier = split_adapter_indices(
            build_adapter_kind_lut(cfg), torch.tensor([[0, 2, 1], [1, 1, 0]])
        )
        assert lora.tolist() == [[0, 2, 0], [0, 0, 0]]
        assert classifier.tolist() == [[0, 0, 1], [1, 1, 0]]

    def test_no_classifier_slots_pass_through(self):
        indices = torch.tensor([0, 1, 2])
        lora, classifier = split_adapter_indices(None, indices)
        assert lora is indices
        assert not classifier.any()
