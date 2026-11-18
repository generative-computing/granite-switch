# SPDX-License-Identifier: Apache-2.0
"""Head modules without a model (``granite_switch.streams``).

CPU, configs and bare modules only. Covers:

* the routing tables both models build from the stream entries;
* one module per head type and backend, attached to the model;
* the probe type: its settings check, its bank, and loading a slot's weights;
* the probe's HF module (its vLLM module needs CUDA: ``tests/vllm/``);
* strict loading of head tensors on vLLM.
"""

import pytest
import torch
from torch import nn

from granite_switch.streams import (
    HeadType,
    get_head_type,
    register_head_type,
    registry,
    routing,
    stream_specs,
)
from granite_switch.streams.checkpoint import HeadWeightLoader
from granite_switch.streams.types.probe import hf_head, vllm_head
from tests.shared.stream_plugins.buffered_head import BufferedHeadType
from tests.shared.stream_plugins.mlp_head import MLPHeads
from tests.unit.test_stream_entries import HIDDEN, LABELS, _config, _probe


def _mixed():
    """Slots: a LoRA, a probe, a LoRA, a probe, an ``mlp_head`` merge."""
    return _config(
        [
            {"type": "lora"},
            _probe(1, labels=[100, 200, 201], read_layer=0),
            {"type": "lora"},
            _probe(0),
            {
                "type": "mlp_head",
                "read_layer": 1,
                "head_row": 0,
                "output": {"mode": "merge", "token_ids": [7], "weight": 0.5},
                "settings": {"width": 4},
            },
        ]
    )


# --------------------------------------------------------------------------- #
# Routing tables
# --------------------------------------------------------------------------- #
class TestRoutingTables:
    def test_heads_group_by_type(self):
        by_type = routing.head_specs(_mixed())
        assert {t: [s.slot for s in specs] for t, specs in by_type.items()} == {
            "probe": [2, 4],
            "mlp_head": [5],
        }

    def test_a_slot_finds_its_row_in_its_types_bank(self):
        config = _mixed()
        # 1 + head_row; 0 for base and for every slot of another type.
        assert routing.row_lut(config, "probe").tolist() == [0, 0, 2, 0, 1, 0]
        assert routing.row_lut(config, "mlp_head").tolist() == [0, 0, 0, 0, 0, 1]

    def test_only_adapter_slots_take_lora_rows(self):
        config = _mixed()
        assert routing.lora_rows(config) == [1, 3]
        assert routing.lora_row_lut(config).tolist() == [0, 1, 0, 2, 0, 0]

    def test_heads_group_by_read_layer(self):
        config = _mixed()
        assert routing.read_groups(config) == {
            0: {"probe": [2]},
            None: {"probe": [4]},
            1: {"mlp_head": [5]},
        }
        assert routing.read_slots(config, None, "probe") == [4]
        assert routing.read_slots(config, 1, "probe") == []

    def test_output_tables_and_score_width(self):
        config = _mixed()
        tables = routing.head_outputs(config)
        assert tables["select"].tolist() == [False, False, True, False, True, False]
        assert tables["merge"].tolist() == [False] * 5 + [True]
        assert tables["weight"][5].item() == 0.5
        assert routing.max_head_outputs(config) == 3  # the widest head
        assert routing.max_head_outputs(_config([{"type": "lora"}])) == 0

    def test_checkpoint_names(self):
        config = _mixed()
        assert routing.module_attr("probe") == "probe_head"  # pinned by the type
        assert routing.module_attr("mlp_head") == "stream_mlp_head"
        assert routing.head_param_prefixes(config) == (
            "model.probe_head.",
            "model.stream_mlp_head.",
        )
        assert routing.head_control_token_ids(config) == [251, 253, 254]


# --------------------------------------------------------------------------- #
# Head modules
# --------------------------------------------------------------------------- #
class TestHeadModules:
    def test_one_module_per_type_for_each_backend(self):
        config = _mixed()
        hf = routing.build_head_modules(config, "hf")
        vllm = routing.build_head_modules(config, "vllm")
        assert isinstance(hf["probe"], hf_head.SwitchedProbeHead)
        assert isinstance(vllm["probe"], vllm_head.SwitchedProbeHead)
        # 1 + max(head_row) bank rows, padded to the widest probe slot.
        assert hf["probe"].weight.shape == vllm["probe"].weight.shape == (2, 3, HIDDEN)
        # A type without a vLLM module of its own serves its HF module.
        assert isinstance(hf["mlp_head"], MLPHeads)
        assert isinstance(vllm["mlp_head"], MLPHeads)

    def test_a_pre_streams_probe_bank_keeps_a_row_per_slot(self):
        legacy = _config(
            None, adapter_kinds=["probe", "lora"], probe_label_token_ids=[LABELS, None]
        )
        probe = routing.build_head_modules(legacy, "hf")["probe"]
        assert probe.weight.shape == (2, len(LABELS), HIDDEN)

    def test_each_slot_is_checked_by_its_type(self):
        config = _config([_probe(0, settings={"normalize": True})])
        with pytest.raises(ValueError, match="probe slot 1 accepts no settings"):
            routing.build_head_modules(config, "hf")

    def test_a_type_must_build_a_module(self, monkeypatch):
        monkeypatch.setattr(registry, "_REGISTRY", dict(registry._REGISTRY))

        @register_head_type
        class NotAModule(HeadType):
            name = "not_a_module"

            def hf_module(self, specs, config):
                return torch.zeros(1)

        config = _config([{**_probe(0), "type": "not_a_module"}])
        with pytest.raises(TypeError, match="'not_a_module' must build an nn.Module"):
            routing.build_head_modules(config, "hf")

    def test_heads_never_replace_a_model_attribute(self):
        modules = routing.build_head_modules(_mixed(), "hf")
        model = nn.Module()
        routing.attach_head_modules(model, modules)
        assert model.probe_head is modules["probe"]
        assert model.stream_mlp_head is modules["mlp_head"]
        with pytest.raises(ValueError, match="'probe_head', which is already"):
            routing.attach_head_modules(model, modules)


# --------------------------------------------------------------------------- #
# The probe type and its HF module
# --------------------------------------------------------------------------- #
class TestProbe:
    @staticmethod
    def _bank_and_spec():
        """The mixed config's probe bank, and its slot-4 probe (row 0, 2 labels)."""
        config = _mixed()
        bank = routing.build_head_modules(config, "hf")["probe"]
        return bank, stream_specs(config)[3]

    def test_a_slots_weights_land_in_its_row(self):
        bank, spec = self._bank_and_spec()
        weight, bias = torch.randn(2, HIDDEN), torch.randn(2)
        get_head_type("probe").load_weights(
            bank, spec.head_row, {"weight": weight, "bias": bias}, spec
        )
        torch.testing.assert_close(bank.weight[0, :2].detach(), weight)
        torch.testing.assert_close(bank.bias[0, :2].detach(), bias)
        # The padded label and the other slot's row stay zero.
        assert not bank.weight[0, 2:].any() and not bank.weight[1].any()

    @pytest.mark.parametrize(
        "state, message",
        [
            ({"weight": torch.zeros(2, HIDDEN)}, "must hold exactly"),
            (
                {"weight": torch.zeros(3, HIDDEN), "bias": torch.zeros(3)},
                "do not match 2 labels x hidden 64",
            ),
            (
                {"weight": torch.zeros(2, HIDDEN - 1), "bias": torch.zeros(2)},
                "do not match 2 labels x hidden 64",
            ),
        ],
    )
    def test_a_probe_file_must_match_its_entry(self, state, message):
        bank, spec = self._bank_and_spec()
        with pytest.raises(ValueError, match=message):
            get_head_type("probe").load_weights(bank, spec.head_row, state, spec)

    def test_each_marker_reads_its_own_row(self):
        torch.manual_seed(0)
        head = hf_head.SwitchedProbeHead(hidden_size=8, num_rows=2, max_num_labels=3)
        x, rows = torch.randn(4, 8), torch.tensor([0, 2, 1, 2])
        with torch.no_grad():
            head.weight.normal_()
            head.bias.normal_()
            expected = torch.zeros(4, 3)
            for t, row in enumerate(rows.tolist()):
                if row:
                    expected[t] = head.weight[row - 1] @ x[t] + head.bias[row - 1]
            torch.testing.assert_close(head(x, rows), expected)
            # [batch, seq, hidden] keeps its shape.
            torch.testing.assert_close(
                head(x.view(2, 2, 8), rows.view(2, 2)), expected.view(2, 2, 3)
            )
            assert not head(x, torch.zeros(4, dtype=torch.long)).any()


# --------------------------------------------------------------------------- #
# Strict head loading on vLLM
# --------------------------------------------------------------------------- #
class _Model(nn.Module):
    """Stands in for vLLM's model: its head modules, on their attributes."""

    def __init__(self, config):
        super().__init__()
        heads = routing.build_head_modules(config, "vllm")
        self._head_types = list(heads)
        routing.attach_head_modules(self, heads)


class TestHeadWeightLoader:
    @staticmethod
    def _loader():
        """A probe, and a plugin whose persistent buffer is part of its state."""
        config = _config([_probe(0), {**_probe(0), "type": BufferedHeadType.name}])
        return HeadWeightLoader(_Model(config))

    def test_loads_every_head_tensor_buffers_included(self):
        loader = self._loader()
        assert sorted(loader.tensors) == [
            "model.probe_head.bias",
            "model.probe_head.weight",
            "model.stream_test_buffered.adapter_token_ids",
            "model.stream_test_buffered.weight",
        ]
        values = {key: torch.randn(t.shape) for key, t in loader.tensors.items()}
        for key, value in values.items():
            assert loader.load(key, value)
        assert loader.finish() == set(values)
        for key, value in values.items():
            torch.testing.assert_close(loader.tensors[key].detach(), value)

    def test_leaves_other_tensors_to_the_decoder(self):
        weight = "model.layers.0.self_attn.qkv_proj.weight"
        assert not self._loader().load(weight, torch.ones(1))

    def test_refuses_a_missing_unexpected_or_misshaped_tensor(self):
        loader = self._loader()
        weight = "model.stream_test_buffered.weight"
        loader.load(weight, torch.zeros(loader.tensors[weight].shape))
        with pytest.raises(ValueError, match="Missing head.*adapter_token_ids"):
            loader.finish()
        with pytest.raises(ValueError, match="Unexpected head checkpoint tensor"):
            loader.load("model.stream_test_buffered.typo", torch.ones(1))
        with pytest.raises(ValueError, match=r"has shape \(1,\); expected"):
            loader.load(weight, torch.ones(1))
