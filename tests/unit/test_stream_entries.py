# SPDX-License-Identifier: Apache-2.0
"""Stream entries without a model (``granite_switch.streams``).

CPU, configs only. Covers:

* the registry (built-in probe, plugins by entry point, refused names, the
  defaults a head type inherits);
* stream entries: validation and refusals, and the legacy entries rebuilt from
  the pre-streams probe fields;
* the two output modes, ``select`` and ``merge``.
"""

import pytest
import torch

from granite_switch.config import GraniteSwitchConfig
from granite_switch.streams import (
    HeadType,
    get_head_type,
    head_type_names,
    register_head_type,
    stream_specs,
)
from granite_switch.streams import routing as stream_routing
from granite_switch.streams.outputs import apply_head_scores, output_tables
from granite_switch.streams.registry import _REGISTRY
from granite_switch.streams.spec import parse_entries
from tests.shared.stream_plugins.mlp_head import MLP_HEAD_FILE

LABELS = [100, 200]
HIDDEN = 64


def _probe(row, labels=LABELS, **extra):
    """A probe slot's stream entry, selecting over ``labels``."""
    return {
        "type": "probe",
        "head_row": row,
        "output": {"mode": "select", "token_ids": list(labels)},
        **extra,
    }


def _config(streams, **extra):
    """A config for the entry tests; no model is built from it.

    ``streams=None`` is a checkpoint composed before streams: pass its
    ``adapter_kinds`` and probe fields in ``extra``.
    """
    n = len(streams if streams is not None else extra["adapter_kinds"])
    return GraniteSwitchConfig(
        vocab_size=300,
        hidden_size=HIDDEN,
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
        **extra,
    )


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

    @staticmethod
    def _install(monkeypatch, entry_points):
        """Stand in for the installed entry points; returns the lookups made."""
        from granite_switch.streams import registry

        seen = []

        def fake_entry_points(group, name=None):
            seen.append((group, name))
            return [ep for ep in entry_points if name is None or ep.name == name]

        monkeypatch.setattr(registry, "entry_points", fake_entry_points)
        return seen

    @staticmethod
    def _entry_point(name, load):
        class FakeEntryPoint:
            value = f"plugin_pkg:{name}"

            def load(self):
                return load()

        ep = FakeEntryPoint()
        ep.name = name
        return ep

    def test_plugins_load_from_the_entry_point_group(self, monkeypatch):
        class EntryPointType(HeadType):
            name = "ep_type"

        seen = self._install(
            monkeypatch, [self._entry_point("ep_type", lambda: EntryPointType)]
        )
        try:
            assert isinstance(get_head_type("ep_type"), EntryPointType)
            # Only the entry point for the name asked is looked up.
            assert seen == [("granite_switch.streams", "ep_type")]
        finally:
            _REGISTRY.pop("ep_type", None)

    def test_a_broken_plugin_affects_only_its_own_name(self, monkeypatch):
        class EntryPointType(HeadType):
            name = "ep_type"

        def broken():
            raise ImportError("the plugin's package is broken")

        self._install(
            monkeypatch,
            [
                self._entry_point("broken", broken),
                self._entry_point("ep_type", lambda: EntryPointType),
            ],
        )
        try:
            assert isinstance(get_head_type("ep_type"), EntryPointType)
            with pytest.raises(ImportError, match="broken"):
                get_head_type("broken")
        finally:
            _REGISTRY.pop("ep_type", None)

    def test_an_entry_point_is_named_after_its_type(self, monkeypatch):
        class EntryPointType(HeadType):
            name = "long_name"

        self._install(monkeypatch, [self._entry_point("short", lambda: EntryPointType)])
        with pytest.raises(ValueError, match="name the entry point after the type"):
            get_head_type("short")
        assert "long_name" not in _REGISTRY

    def test_an_entry_point_loads_a_head_type(self, monkeypatch):
        self._install(monkeypatch, [self._entry_point("not_a_type", lambda: len)])
        with pytest.raises(TypeError, match="not a HeadType subclass"):
            get_head_type("not_a_type")

    def test_names_list_plugins_without_importing_them(self, monkeypatch):
        def must_not_load():
            raise AssertionError("listing names must not import a plugin")

        self._install(monkeypatch, [self._entry_point("listed", must_not_load)])
        assert {"probe", "listed"} <= set(head_type_names())

    def test_a_taken_name_is_refused(self):
        class Impostor(HeadType):
            name = "probe"

        with pytest.raises(ValueError, match="already registered"):
            register_head_type(Impostor)
        assert get_head_type("probe").module_name == "probe_head"

    def test_registering_the_same_class_again_is_a_no_op(self):
        probe = type(get_head_type("probe"))
        assert register_head_type(probe) is probe

    def test_only_head_types_register(self):
        with pytest.raises(TypeError, match="not a HeadType subclass"):
            register_head_type(object)

    def test_vllm_runs_the_hf_module_by_default(self):
        class HFOnly(HeadType):
            name = "hf_only"

            def hf_module(self, specs, config):
                return torch.nn.Identity()

        assert isinstance(HFOnly().vllm_module([], None), torch.nn.Identity)

    def test_a_type_must_build_its_module_and_load_its_weights(self):
        class Bare(HeadType):
            name = "bare"

        with pytest.raises(NotImplementedError):
            Bare().hf_module([], None)
        with pytest.raises(NotImplementedError):
            Bare().load_weights(torch.nn.Module(), 0, {}, None)


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
            ({**_probe(0), "read_layer": -1}, "read_layer must be a decoder layer"),
            ({**_probe(0), "read_layer": True}, "read_layer must be a decoder layer"),
            ({**_probe(0), "read_layer": "1"}, "read_layer must be a decoder layer"),
            (
                {**_probe(0), "output": {"token_ids": [7, 7]}},
                "token_ids must be distinct",
            ),
            (
                {**_probe(0), "output": {"token_ids": [1], "weight": "2"}},
                "weight must be a finite number",
            ),
            (
                {**_probe(0), "output": {"token_ids": [1], "weight": True}},
                "weight must be a finite number",
            ),
            (
                {**_probe(0), "output": {"token_ids": [1], "weight": float("inf")}},
                "weight must be a finite number",
            ),
            ({**_probe(0), "settings": [1]}, "settings must be a dict or null"),
        ],
    )
    def test_rejects_bad_entries(self, entry, message):
        with pytest.raises(ValueError, match=message):
            parse_entries([entry], 1)

    @pytest.mark.parametrize(
        "entry, field",
        [
            ({**_probe(0), "read_layr": 3}, "read_layr"),
            # A head on an adapter was a field once; this version refuses it.
            ({**_probe(0), "adapter": "ad"}, "adapter"),
            ({**_probe(0), "output": {"token_ids": [1], "scale": 2.0}}, "scale"),
        ],
    )
    def test_refuses_fields_it_does_not_know(self, entry, field):
        # A field from a newer version would change what the slot computes;
        # skipping it would run the slot as something else, with no error.
        with pytest.raises(ValueError, match=f"doesn't know: \\['{field}'\\]"):
            parse_entries([entry], 1)

    @pytest.mark.parametrize(
        "head",
        [
            _probe(0, read_layer=2),
            {
                "type": "mlp_head",
                "read_layer": 5,
                "head_row": 0,
                "output": {"mode": "merge", "token_ids": list(LABELS)},
                "settings": {"width": 8},
            },
        ],
        ids=["probe", "plugin"],
    )
    def test_read_layer_must_be_a_decoder_layer(self, head):
        # Every head type, not only the probe: a layer the model doesn't have
        # is never reached, so the head would silently score nothing.
        config = _config([{"type": "lora"}, head])  # 2 decoder layers
        with pytest.raises(ValueError, match=r"has decoder layers 0\.\.1"):
            stream_specs(config)

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
        assert stream_routing.head_control_token_ids(config) == [251]

    def test_adapter_kinds_must_agree_with_the_streams(self):
        with pytest.raises(ValueError, match="disagree with the streams"):
            _config([{"type": "lora"}, _probe(0)], adapter_kinds=["lora", "lora"])

    def test_legacy_fields_rebuild_the_same_entries(self):
        """A pre-streams probe checkpoint: entries come from its probe fields."""
        legacy = _config(
            None,
            adapter_kinds=["lora", "probe"],
            probe_label_token_ids=[None, LABELS],
            probe_read_layers=[None, 1],
        )
        assert legacy.streams is None
        (lora, probe) = stream_specs(legacy)
        assert not lora.is_head
        # The pre-streams bank had one row per slot: the head row is the slot's.
        assert (probe.type, probe.head_row, probe.read_layer) == ("probe", 1, 1)
        assert probe.token_ids == tuple(LABELS)
        # ... and every slot kept its LoRA row.
        assert legacy.num_lora_rows == 2
        assert stream_routing.lora_rows(legacy) == [1, 2]


@pytest.mark.parametrize("token_id", [300, 999])
def test_output_ids_are_checked_before_model_execution(token_id):
    with pytest.raises(ValueError, match="token_ids must be below vocab_size"):
        stream_specs(_config([_probe(0, labels=[token_id])]))


@pytest.mark.parametrize("entry", ["probe", {}, {"type": None}, {"type": []}])
def test_config_errors_name_the_stream_entry(entry):
    with pytest.raises(ValueError, match=r"streams\[0\]"):
        _config([entry])


def test_streams_must_be_a_list():
    with pytest.raises(ValueError, match="streams must be a list of entries"):
        parse_entries({"type": "lora"}, 1)


# --------------------------------------------------------------------------- #
# Output modes
# --------------------------------------------------------------------------- #
def _specs(*heads):
    """Stream specs: slot 1 a LoRA, then one head per ``(mode, token_ids, weight)``."""
    entries = [{"type": "lora"}] + [
        {
            "type": "probe",
            "head_row": row,
            "output": {"mode": mode, "token_ids": ids, "weight": weight},
        }
        for row, (mode, ids, weight) in enumerate(heads)
    ]
    return parse_entries(entries, len(entries))


class TestOutputModes:
    def _apply(self, mode, weight=1.0):
        logits = torch.tensor([[2.1, 1.4, 3.0, 0.5], [1.0, 1.0, 1.0, 1.0]])
        # Slot 2 has two outputs; slot 3 widens the tables to three, so slot
        # 2's third score column is padding and must not be read.
        tables = output_tables(
            _specs((mode, [0, 1], weight), ("select", [0, 1, 2], 1.0)), 3
        )
        scores = torch.tensor([[5.0, -1.0, 99.0], [0.0, 0.0, 0.0]])
        # Row 1 is not a head row (slot 0): it keeps its logits.
        return apply_head_scores(logits, torch.tensor([2, 0]), scores, tables)

    def test_select_keeps_only_the_head_scores(self):
        out = self._apply("select")
        assert out[0, :2].tolist() == [5.0, -1.0]
        assert torch.isneginf(out[0, 2:]).all()
        assert out[1].tolist() == [1.0] * 4  # other rows untouched

    def test_merge_nudges_the_lm_row(self):
        out = self._apply("merge", weight=0.5)
        torch.testing.assert_close(out[0], torch.tensor([4.6, 0.9, 3.0, 0.5]))
        assert out[1].tolist() == [1.0] * 4

    def test_no_host_reads_and_matches_a_per_row_reference(self):
        """Branch-free: no tensor is read back, so the step can be graph-captured.

        One batch of select and merge heads with different widths, LoRA rows,
        base rows and the padded columns of a narrower head.
        """
        heads = [("select", [7, 9], 1.0), ("merge", [11, 3, 5], 0.5)]
        tables = output_tables(_specs(*heads), 3)
        by_slot = dict(enumerate(heads, start=2))
        g = torch.Generator().manual_seed(0)
        for dtype in (torch.float32, torch.bfloat16):
            logits = torch.randn(7, 40, generator=g).to(dtype)
            scores = torch.randn(7, 3, generator=g)
            slots = torch.tensor([0, 2, 1, 3, 0, 3, 2])
            expected = logits.clone()
            for r, s in enumerate(slots.tolist()):
                if s in by_slot:
                    mode, ids, weight = by_slot[s]
                    own = scores[r, : len(ids)].to(dtype)
                    if mode == "select":
                        expected[r] = float("-inf")
                        expected[r, ids] = own
                    else:
                        expected[r, ids] = expected[r, ids] + weight * own
            reads = []
            names = ("__bool__", "nonzero", "tolist", "item", "__int__")
            originals = {n: getattr(torch.Tensor, n) for n in names}

            def counting(name):
                def f(t, *a, **k):
                    reads.append(name)
                    return originals[name](t, *a, **k)

                return f

            for n in names:
                setattr(torch.Tensor, n, counting(n))
            try:
                out = apply_head_scores(logits, slots, scores, tables)
            finally:
                for n in names:
                    setattr(torch.Tensor, n, originals[n])
            assert reads == [], f"host reads in the head-output step: {reads}"
            torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_traces_as_one_graph(self):
        """``fullgraph=True`` compile: no graph break anywhere in the step."""
        tables = output_tables(_specs(("select", [7, 9], 1.0), ("merge", [3], 2.0)), 3)
        logits, scores = torch.randn(4, 16), torch.randn(4, 2)
        slots = torch.tensor([2, 0, 3, 1])
        compiled = torch.compile(apply_head_scores, fullgraph=True, backend="eager")
        torch.testing.assert_close(
            compiled(logits.clone(), slots, scores, tables),
            apply_head_scores(logits.clone(), slots, scores, tables),
            rtol=0,
            atol=0,
        )
