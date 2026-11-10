# SPDX-License-Identifier: Apache-2.0
"""The stream API: a head stream type plugs in with a config entry, a module
and a weights file, and nothing else (``granite_switch.streams``).

CPU, tiny random models. Covers:

* the registry (built-in probe, plugins, unknown names);
* stream entries: validation, and the legacy entries rebuilt from the
  pre-streams probe fields;
* the two output modes, ``select`` and ``merge``;
* an example plugin type (``tests/shared/stream_plugins/mlp_head.py``) composed
  through the real composer and run end to end, with ``merge``;
* compact LoRA rows: head slots take no row in the LoRA tensors;
* the pre-streams checkpoint layout, unchanged.
"""

import pytest
import torch
from safetensors.torch import save_file

from granite_switch.config import GraniteSwitchConfig
from granite_switch.hf import GraniteSwitchForCausalLM
from granite_switch.streams import (
    HeadType,
    get_head_type,
    head_type_names,
    register_head_type,
    stream_specs,
)
from granite_switch.streams import routing as stream_routing
from granite_switch.streams.outputs import apply_head_scores
from granite_switch.streams.registry import _REGISTRY
from granite_switch.streams.spec import parse_entries
from tests.composer.test_sr_compose_e2e import (
    CONTROL_TOKEN_ID,
    HIDDEN,
    VOCAB_SIZE,
    _create_base_model,
    _create_sr_adapter,
)
from tests.shared.stream_plugins.mlp_head import MLP_HEAD_FILE, MLPHeads

LABELS = [100, 200]


def _probe(row, **extra):
    return {
        "type": "probe",
        "head_row": row,
        "output": {"mode": "select", "token_ids": list(LABELS)},
        **extra,
    }


def _config(streams, *, dual_stream=False, hidden=64, **extra):
    n = len(streams)
    return GraniteSwitchConfig(
        vocab_size=300,
        hidden_size=hidden,
        intermediate_size=128,
        num_hidden_layers=4,  # 2 switch cache layers + 2 decoder layers
        num_attention_heads=4,
        num_key_value_heads=4,
        switch_head_dim=16,
        num_adapters=n,
        adapter_token_ids=[250 + i for i in range(n)],
        adapter_substitute_token_ids=[1] * n,
        adapter_names=[f"s{i + 1}" for i in range(n)],
        max_lora_rank=4,
        adapter_ranks=[4] * n,
        streams=streams,
        dual_stream=dual_stream,
        cross_stream_rank=4 if dual_stream else None,
        **extra,
    )


def _model(config):
    model = GraniteSwitchForCausalLM(config).eval()
    model.model.adapter_token_ids.data = torch.tensor(
        config.adapter_token_ids, dtype=torch.long
    )
    return model


def _last_row(model, ids):
    with torch.no_grad():
        return model(input_ids=torch.tensor([ids]), use_cache=False).logits[0, -1]


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
class TestRegistry:
    def test_probe_is_built_in(self):
        assert "probe" in head_type_names()
        assert get_head_type("probe").weights_file == "probe_head.safetensors"

    def test_unknown_type_names_the_known_ones(self):
        with pytest.raises(ValueError, match="Unknown stream type 'nope'.*'probe'"):
            get_head_type("nope")

    def test_lora_is_not_a_head_type(self):
        class Bad(HeadType):
            name = "lora"

        with pytest.raises(ValueError, match="built-in adapter stream"):
            register_head_type(Bad)

    def test_a_type_needs_a_name(self):
        class Nameless(HeadType):
            pass

        with pytest.raises(ValueError, match="has no name"):
            register_head_type(Nameless)

    def test_plugin_registers_on_import(self):
        assert "mlp_head" in _REGISTRY
        assert get_head_type("mlp_head").weights_file == MLP_HEAD_FILE

    def test_plugins_load_from_the_entry_point_group(self, monkeypatch):
        from granite_switch.streams import registry

        class EntryPointType(HeadType):
            name = "ep_type"

        class FakeEntryPoint:
            name = "ep_type"

            def load(self):
                return EntryPointType

        seen = []

        def fake_entry_points(group):
            seen.append(group)
            return [FakeEntryPoint()]

        monkeypatch.setattr(registry, "entry_points", fake_entry_points)
        monkeypatch.delitem(_REGISTRY, "ep_type", raising=False)
        try:
            assert isinstance(get_head_type("ep_type"), EntryPointType)
            assert seen == ["granite_switch.streams"]
        finally:
            _REGISTRY.pop("ep_type", None)


# --------------------------------------------------------------------------- #
# Stream entries
# --------------------------------------------------------------------------- #
class TestStreamEntries:
    def test_parses_lora_and_head_entries(self):
        specs = parse_entries(
            [
                {"type": "lora"},
                _probe(0, read_layer=1, settings={"k": 2}),
            ],
            2,
        )
        assert [s.is_head for s in specs] == [False, True]
        head = specs[1]
        assert (head.slot, head.read_layer, head.head_row) == (2, 1, 0)
        assert (head.mode, head.token_ids, head.weight) == ("select", (100, 200), 1.0)
        assert head.settings == {"k": 2}

    @pytest.mark.parametrize(
        "entry, message",
        [
            ({"read_layer": 1}, "must be a dict with a type"),
            ({"type": "lora", "head_row": 0}, "LoRA stream; it takes no"),
            ({**_probe(0), "output": {"mode": "add", "token_ids": [1]}}, "output.mode"),
            ({**_probe(0), "output": {"token_ids": []}}, "token_ids must be non-empty"),
            (
                {**_probe(0), "output": {"token_ids": [-1]}},
                "token_ids must be non-empty",
            ),
            ({**_probe(0), "head_row": None}, "head_row must be a non-negative int"),
        ],
    )
    def test_rejects_bad_entries(self, entry, message):
        with pytest.raises(ValueError, match=message):
            parse_entries([entry], 1)

    def test_one_entry_per_slot(self):
        with pytest.raises(ValueError, match="one entry per slot"):
            parse_entries([{"type": "lora"}], 2)

    def test_head_rows_are_distinct_per_type(self):
        with pytest.raises(ValueError, match="share a head_row"):
            stream_specs(_config([_probe(0), _probe(0)]))

    def test_config_round_trips(self, tmp_path):
        streams = [{"type": "lora"}, _probe(0, read_layer=1)]
        config = _config(streams)
        config.save_pretrained(tmp_path)
        loaded = GraniteSwitchConfig.from_pretrained(tmp_path)
        assert loaded.streams == streams
        assert stream_specs(loaded) == stream_specs(config)

    def test_kinds_and_probe_fields_follow_the_streams(self):
        config = _config([{"type": "lora"}, _probe(0, read_layer=1)])
        assert config.adapter_kinds == ["lora", "probe"]
        assert config.probe_label_token_ids == [None, LABELS]
        assert config.probe_read_layers == [None, 1]
        assert config.probe_control_token_ids == [251]

    def test_adapter_kinds_must_agree_with_the_streams(self):
        with pytest.raises(ValueError, match="disagree with the streams"):
            _config([{"type": "lora"}, _probe(0)], adapter_kinds=["lora", "lora"])

    def test_legacy_fields_rebuild_the_same_entries(self):
        """A pre-streams probe checkpoint: entries come from its probe fields."""
        legacy = GraniteSwitchConfig(
            vocab_size=300,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=4,
            num_adapters=2,
            adapter_token_ids=[250, 251],
            adapter_substitute_token_ids=[1, 1],
            adapter_names=["s1", "s2"],
            adapter_kinds=["lora", "probe"],
            probe_label_token_ids=[None, LABELS],
            probe_read_layers=[None, 1],
            max_lora_rank=4,
            adapter_ranks=[4, 4],
        )
        assert legacy.streams is None
        (lora, probe) = stream_specs(legacy)
        assert not lora.is_head
        # The pre-streams bank had one row per slot: the head row is the slot's.
        assert (probe.type, probe.head_row, probe.read_layer) == ("probe", 1, 1)
        assert probe.token_ids == tuple(LABELS)
        # ... and every slot kept its LoRA row.
        assert legacy.num_lora_rows == 2


# --------------------------------------------------------------------------- #
# Output modes
# --------------------------------------------------------------------------- #
class TestOutputModes:
    def _apply(self, mode, weight=1.0):
        logits = torch.tensor([[2.1, 1.4, 3.0, 0.5], [1.0, 1.0, 1.0, 1.0]])
        outputs = {3: (mode, torch.tensor([0, 1]), weight)}
        scores = torch.tensor([[5.0, -1.0, 99.0]])  # third column is padding
        return apply_head_scores(
            logits, torch.tensor([0]), torch.tensor([3]), scores, outputs
        )

    def test_select_keeps_only_the_head_scores(self):
        out = self._apply("select")
        assert out[0, :2].tolist() == [5.0, -1.0]
        assert torch.isneginf(out[0, 2:]).all()
        assert out[1].tolist() == [1.0] * 4  # other rows untouched

    def test_merge_nudges_the_lm_row(self):
        out = self._apply("merge", weight=0.5)
        torch.testing.assert_close(out[0], torch.tensor([4.6, 0.9, 3.0, 0.5]))
        assert out[1].tolist() == [1.0] * 4


# --------------------------------------------------------------------------- #
# The example plugin, end to end through the composer
# --------------------------------------------------------------------------- #
WIDTH = 8


def _write_mlp_head(path, w1, w2):
    path.mkdir(parents=True)
    save_file({"w1": w1, "w2": w2}, str(path / MLP_HEAD_FILE))


def _compose_with_plugin(tmp_path, mode, weight=1.0):
    """A tiny base + one SR adapter + one ``mlp_head`` slot, composed for real."""
    from granite_switch.composer import GraniteSwitchComposer

    base, sr, head = tmp_path / "base", tmp_path / "sr", tmp_path / "nudge"
    base.mkdir()
    _create_base_model(base)
    _create_sr_adapter(sr)
    torch.manual_seed(3)
    w1, w2 = torch.randn(WIDTH, HIDDEN) * 0.3, torch.randn(len(LABELS), WIDTH)
    _write_mlp_head(head, w1, w2)
    model = GraniteSwitchComposer.from_base_and_adapters(
        base_model_name_or_path=str(base),
        adapter_paths=[str(sr), str(head)],
        adapter_token_ids=[CONTROL_TOKEN_ID, CONTROL_TOKEN_ID + 1],
        adapter_substitute_token_ids=[1, 2],
        adapter_names=["sr", "nudge"],
        streams=[
            {"type": "lora"},
            {
                "type": "mlp_head",
                "head_row": 0,
                "output": {"mode": mode, "token_ids": LABELS, "weight": weight},
                "settings": {"width": WIDTH},
            },
        ],
        switch_head_dim=16,
    ).eval()
    return model, w1, w2


def test_plugin_type_composes_and_merges(tmp_path):
    model, w1, w2 = _compose_with_plugin(tmp_path, "merge", weight=0.5)
    module = model.model.stream_mlp_head
    assert isinstance(module, MLPHeads)
    torch.testing.assert_close(module.w1[0], w1)
    torch.testing.assert_close(module.w2[0], w2)
    # The head slot owns no LoRA row: the LoRA tensors hold the SR adapter only.
    assert model.config.num_lora_rows == 1
    assert model.model.layers[0].cross_stream.lora_B.shape[0] == 1

    # The same prompt, with and without the head's marker last. The marker is
    # a head slot, so the LoRA path no-ops there, and its LM row must equal
    # the plain model's row at that position, plus the head's scores.
    prompt = [10, 21, 20]
    marker = CONTROL_TOKEN_ID + 1
    states = []
    hook = model.model.norm.register_forward_hook(lambda m, i, o: states.append(o))
    try:
        merged = _last_row(model, [*prompt, marker])
    finally:
        hook.remove()
    assert model.model._last_lora_indices.tolist() == [[0, 0, 0, 0]]

    h = states[0][0, -1]
    expected_scores = w2 @ torch.nn.functional.gelu(w1 @ h)
    lm_row = model.lm_head(h) / model.config.logits_scaling
    torch.testing.assert_close(merged[LABELS], lm_row[LABELS] + 0.5 * expected_scores)
    others = torch.ones(VOCAB_SIZE, dtype=torch.bool)
    others[LABELS] = False
    torch.testing.assert_close(merged[others], lm_row[others])  # merge keeps them


def test_plugin_type_survives_save_and_reload(tmp_path):
    from granite_switch.hf import load_model

    model, _, _ = _compose_with_plugin(tmp_path, "select")
    ids = [10, 21, 20, CONTROL_TOKEN_ID + 1]
    before = _last_row(model, ids)
    assert torch.isneginf(
        before[[t for t in range(VOCAB_SIZE) if t not in LABELS]]
    ).all()

    model.save_pretrained(tmp_path / "composed")
    loaded = load_model(str(tmp_path / "composed")).eval()
    torch.testing.assert_close(_last_row(loaded, ids), before)


def test_plugin_checks_its_settings(tmp_path):
    with pytest.raises(ValueError, match="settings.width"):
        _model(
            _config(
                [
                    {
                        "type": "mlp_head",
                        "head_row": 0,
                        "output": {"mode": "merge", "token_ids": LABELS},
                        "settings": {"width": 0},
                    }
                ]
            )
        )


def test_plugin_rejects_misshaped_weights(tmp_path):
    from granite_switch.composer.weight_transfer import transfer_head_weights

    config = _config(
        [
            {
                "type": "mlp_head",
                "head_row": 0,
                "output": {"mode": "merge", "token_ids": LABELS},
                "settings": {"width": WIDTH},
            }
        ]
    )
    model = _model(config)
    _write_mlp_head(tmp_path / "h", torch.zeros(WIDTH + 1, 64), torch.zeros(2, WIDTH))
    with pytest.raises(ValueError, match="do not match width"):
        transfer_head_weights([str(tmp_path / "h")], [1], model)


# --------------------------------------------------------------------------- #
# Compact rows
# --------------------------------------------------------------------------- #
def test_head_slots_take_no_lora_rows_and_order_is_free():
    """[probe, lora, probe]: one LoRA row, a two-row probe bank, any order."""
    config = _config([_probe(0), {"type": "lora"}, _probe(1)])
    assert config.num_lora_rows == 1
    assert stream_routing.lora_row_lut(config).tolist() == [0, 0, 1, 0]
    assert stream_routing.row_lut(config, "probe").tolist() == [0, 1, 0, 2]
    model = _model(config)
    assert model.model.probe_head.weight.shape[0] == 2
    lora_b = next(p for n, p in model.named_parameters() if "lora_B" in n)
    assert lora_b.shape[0] == 1

    # The LoRA slot, in the middle, routes to LoRA row 1.
    _last_row(model, [10, 251, 20, 30])
    assert model.model._last_lora_indices.tolist() == [[0, 2, 2, 2]]
    # The last probe slot reads its own bank row.
    with torch.no_grad():
        model.model.probe_head.bias[1, :2] = torch.tensor([-5.0, 5.0])
    assert int(_last_row(model, [10, 20, 252]).argmax()) == LABELS[1]


def test_heads_only_checkpoint_has_no_lora():
    config = _config([_probe(0), _probe(1)])
    assert config.num_lora_rows == 0 and config.lora_target_modules == []
    model = _model(config)
    assert not [n for n, _ in model.named_parameters() if "lora_" in n]
    with torch.no_grad():
        model.model.probe_head.bias[0, :2] = torch.tensor([5.0, -5.0])
    assert int(_last_row(model, [10, 20, 250]).argmax()) == LABELS[0]


def test_pre_streams_layout_is_unchanged():
    """No ``streams``: every slot keeps its LoRA row and its probe-bank row."""
    config = GraniteSwitchConfig(
        vocab_size=300,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_adapters=3,
        adapter_token_ids=[250, 251, 252],
        adapter_substitute_token_ids=[1, 1, 1],
        adapter_names=["s1", "s2", "s3"],
        adapter_kinds=["lora", "probe", "probe"],
        probe_label_token_ids=[None, LABELS, LABELS],
        max_lora_rank=4,
        adapter_ranks=[4, 4, 4],
        switch_head_dim=16,
    )
    model = _model(config)
    assert model.model.probe_head.weight.shape[0] == 3
    lora_b = next(p for n, p in model.named_parameters() if "lora_B" in n)
    assert lora_b.shape[0] == 3
    with torch.no_grad():
        model.model.probe_head.bias[2, :2] = torch.tensor([-5.0, 5.0])
    assert int(_last_row(model, [10, 20, 252]).argmax()) == LABELS[1]
