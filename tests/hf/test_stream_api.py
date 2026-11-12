# SPDX-License-Identifier: Apache-2.0
"""The stream API: a head stream type plugs in with a config entry, a module
and a weights file, and nothing else (``granite_switch.streams``).

CPU, tiny models composed by the real composer. Covers:

* the registry (built-in probe, plugins by entry point, refused names);
* stream entries: validation, and the legacy entries rebuilt from the
  pre-streams probe fields;
* the two output modes, ``select`` and ``merge``;
* an example plugin type (``tests/shared/stream_plugins/mlp_head.py``) run end
  to end, with ``merge``;
* a head module's wider, other-dtype output fitted to its score row;
* a head that names an adapter: its marker runs that adapter, so the head
  reads the adapter's state (Shadow Residual from the marker alone;
  single-stream LoRA with the adapter's own token in the prompt too);
* compact LoRA rows: head slots take no row in the LoRA tensors;
* the pre-streams checkpoint layout, unchanged.
"""

import pytest
import torch
from safetensors.torch import save_file

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
from granite_switch.streams.types.probe import PROBE_HEAD_FILE
from tests.composer.test_sr_compose_e2e import (
    CONTROL_TOKEN_ID,
    HIDDEN,
    VOCAB_SIZE,
    _create_base_model,
    _create_sr_adapter,
    _create_standard_lora_adapter,
)
from tests.shared.stream_plugins.mlp_head import MLP_HEAD_FILE, MLPHeads

LABELS = [100, 200]


def _probe(row, labels=LABELS, **extra):
    """A probe slot's stream entry, selecting over ``labels``."""
    return {
        "type": "probe",
        "head_row": row,
        "output": {"mode": "select", "token_ids": list(labels)},
        **extra,
    }


def _config(streams, **extra):
    """A config for the entry and routing tests; models come from the composer."""
    n = len(streams)
    return GraniteSwitchConfig(
        vocab_size=300,
        hidden_size=64,
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


# --------------------------------------------------------------------------- #
# Stream entries
# --------------------------------------------------------------------------- #
class TestStreamEntries:
    def test_parses_lora_and_head_entries(self):
        specs = parse_entries(
            [
                {"type": "lora"},
                _probe(0, read_layer=1, adapter="s1", settings={"k": 2}),
            ],
            2,
        )
        assert [s.is_head for s in specs] == [False, True]
        head = specs[1]
        assert (head.slot, head.read_layer, head.head_row, head.adapter) == (
            2,
            1,
            0,
            "s1",
        )
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
            ({**_probe(0), "adapter": 3}, "adapter must be"),
        ],
    )
    def test_rejects_bad_entries(self, entry, message):
        with pytest.raises(ValueError, match=message):
            parse_entries([entry], 1)

    @pytest.mark.parametrize(
        "entry, field",
        [
            ({**_probe(0), "read_layr": 3}, "read_layr"),
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


# --------------------------------------------------------------------------- #
# Checkpoints composed for real
# --------------------------------------------------------------------------- #
# Every model below is built by GraniteSwitchComposer from adapter and head
# folders on disk, so these tests run the compose pipeline itself (CLAUDE.md).
WIDTH = 8
IDENTITY_LABELS = list(range(100, 100 + HIDDEN))


def _compose(tmp_path, slots, streams, **kwargs):
    """Compose ``slots``, ``[(name, folder), ...]`` in slot order, on a tiny base.

    Slot ``i`` gets control token ``CONTROL_TOKEN_ID + i``; every control token
    embeds as token 1.
    """
    from granite_switch.composer import GraniteSwitchComposer

    base = tmp_path / "base"
    if not base.exists():
        base.mkdir()
        _create_base_model(base)
    n = len(slots)
    return GraniteSwitchComposer.from_base_and_adapters(
        base_model_name_or_path=str(base),
        adapter_paths=[str(folder) for _, folder in slots],
        adapter_token_ids=[CONTROL_TOKEN_ID + i for i in range(n)],
        adapter_substitute_token_ids=[1] * n,
        adapter_names=[name for name, _ in slots],
        streams=streams,
        switch_head_dim=16,
        **kwargs,
    ).eval()


def _sr(path):
    """A Shadow Residual adapter's folder."""
    _create_sr_adapter(path)
    return path


def _lora(path):
    """A single-stream LoRA adapter's folder."""
    _create_standard_lora_adapter(path)
    return path


def _probe_folder(path, weight, bias=None):
    """A probe slot's folder: its ``[n_labels, hidden]`` weight and its bias."""
    path.mkdir(parents=True)
    if bias is None:
        bias = torch.zeros(weight.shape[0])
    save_file({"weight": weight, "bias": bias}, str(path / PROBE_HEAD_FILE))
    return path


def _verdict_probe(path, label):
    """A probe that always answers ``LABELS[label]``: a zero weight and a bias."""
    bias = torch.full((len(LABELS),), -5.0)
    bias[label] = 5.0
    return _probe_folder(path, torch.zeros(len(LABELS), HIDDEN), bias)


def _mlp_head_folder(path, w1, w2):
    """An ``mlp_head`` slot's folder."""
    path.mkdir(parents=True)
    save_file({"w1": w1, "w2": w2}, str(path / MLP_HEAD_FILE))
    return path


def _mlp_entry(mode="merge", weight=1.0, width=WIDTH):
    return {
        "type": "mlp_head",
        "head_row": 0,
        "output": {"mode": mode, "token_ids": LABELS, "weight": weight},
        "settings": {"width": width},
    }


# --------------------------------------------------------------------------- #
# The example plugin, end to end
# --------------------------------------------------------------------------- #
def _compose_with_plugin(tmp_path, mode, weight=1.0):
    """A tiny base + one SR adapter + one ``mlp_head`` slot."""
    torch.manual_seed(3)
    w1, w2 = torch.randn(WIDTH, HIDDEN) * 0.3, torch.randn(len(LABELS), WIDTH)
    model = _compose(
        tmp_path,
        [
            ("sr", _sr(tmp_path / "sr")),
            ("nudge", _mlp_head_folder(tmp_path / "nudge", w1, w2)),
        ],
        [{"type": "lora"}, _mlp_entry(mode, weight)],
    )
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
    head = _mlp_head_folder(
        tmp_path / "nudge", torch.zeros(WIDTH, HIDDEN), torch.zeros(2, WIDTH)
    )
    with pytest.raises(ValueError, match="settings.width"):
        _compose(tmp_path, [("nudge", head)], [_mlp_entry(width=0)])


def test_plugin_rejects_misshaped_weights(tmp_path):
    head = _mlp_head_folder(
        tmp_path / "nudge", torch.zeros(WIDTH + 1, HIDDEN), torch.zeros(2, WIDTH)
    )
    with pytest.raises(ValueError, match="do not match width"):
        _compose(tmp_path, [("nudge", head)], [_mlp_entry()])


class _WideHeads(torch.nn.Module):
    """Returns the leading ``width`` columns of the state it reads, in float64."""

    def __init__(self, width):
        super().__init__()
        self.width = width

    def forward(self, hidden, rows):
        return hidden[:, : self.width].double()


class _WideType(HeadType):
    """A head whose module returns two columns more than its outputs."""

    name = "test_wide"
    weights_file = "wide.safetensors"

    def hf_module(self, specs, config):
        return _WideHeads(max(s.num_outputs for s in specs) + 2)

    def load_weights(self, module, row, state, spec):
        pass


def test_a_head_may_return_extra_columns_in_any_float_dtype(tmp_path, monkeypatch):
    """Column k lands on ``token_ids[k]``; the extra columns are dropped."""
    monkeypatch.setitem(_REGISTRY, _WideType.name, _WideType)
    head = tmp_path / "wide"
    head.mkdir()
    save_file({"unused": torch.zeros(1)}, str(head / _WideType.weights_file))
    model = _compose(tmp_path, [("wide", head)], [_probe(0, type="test_wide")])

    states = []
    hook = model.model.norm.register_forward_hook(lambda m, i, o: states.append(o))
    try:
        row = _last_row(model, [10, 20, CONTROL_TOKEN_ID])
    finally:
        hook.remove()
    assert row.dtype == torch.float32
    torch.testing.assert_close(row[LABELS], states[0][0, -1, : len(LABELS)])


# --------------------------------------------------------------------------- #
# A head on an adapter
# --------------------------------------------------------------------------- #
def _identity_probe_model(tmp_path, adapter):
    """[adapter "ad", a probe on the base, a probe on "ad"], identity heads.

    An identity head returns the state it read, so its label scores ARE that
    state. ``adapter`` writes the adapter's folder (:func:`_sr` or
    :func:`_lora`).
    """
    probe = _probe_folder(tmp_path / "probe", torch.eye(HIDDEN))
    return _compose(
        tmp_path,
        [("ad", adapter(tmp_path / "ad")), ("probe", probe), ("probe_on_ad", probe)],
        [
            {"type": "lora"},
            _probe(0, labels=IDENTITY_LABELS),
            _probe(1, labels=IDENTITY_LABELS, adapter="ad"),
        ],
    )


def test_head_on_an_adapter_reads_that_adapter(tmp_path):
    """Shadow Residual: the marker alone runs the adapter, and the head reads it."""
    model = _identity_probe_model(tmp_path, _sr)
    states = []

    def run(ids):
        states.clear()
        hook = model.model.layers[-1].register_forward_hook(
            lambda m, i, o: states.append(o[0])
        )
        try:
            row = _last_row(model, ids)[IDENTITY_LABELS]
        finally:
            hook.remove()
        base, adapted = (model.model.norm(h)[0, -1] for h in states[0])
        return row, base, adapted

    prompt = [5, 17, 42, 99]
    on_base, base, _ = run([*prompt, 251])
    assert model.model._last_lora_indices.tolist() == [[0, 0, 0, 0, 0]]
    torch.testing.assert_close(on_base, base, atol=1e-5, rtol=0)

    on_adapter, base_again, adapted = run([*prompt, 252])
    # The marker runs adapter slot 1 (its LoRA row), and only the marker.
    assert model.model._last_lora_indices.tolist() == [[0, 0, 0, 0, 1]]
    torch.testing.assert_close(on_adapter, adapted, atol=1e-5, rtol=0)
    assert not torch.allclose(on_adapter, base_again, atol=1e-3)
    # Every token before the marker is the base model's, both times.
    torch.testing.assert_close(base, base_again)


def _final_state(model, ids):
    with torch.no_grad():
        out = model.model(input_ids=torch.tensor([ids]), use_cache=False)
    return out.last_hidden_state[0, -1]


def test_head_on_a_lora_adapter_reads_that_adapter_with_its_token(tmp_path):
    """Single-stream LoRA: the adapter's own token makes the head's read exact.

    The adapter writes the K/V of every token it runs on, so the marker alone
    can't stand in for it. With the adapter's token first, as the chat
    template places it, the head reads exactly the state the adapter computes
    when it runs alone: the identity head's row IS that state.
    """
    model = _identity_probe_model(tmp_path, _lora)
    body = [5, 17, 42, 99]
    # The reference: the adapter alone, then a plain token that embeds as the
    # marker does (its substitute, id 1).
    reference = _final_state(model, [250, *body, 1])
    paired = _last_row(model, [250, *body, 252])[IDENTITY_LABELS]
    assert model.model._last_lora_indices.tolist() == [[1, 1, 1, 1, 1, 1]]
    torch.testing.assert_close(paired, reference, atol=1e-5, rtol=0)

    # The marker alone runs the adapter on one token only: not the same state.
    alone = _last_row(model, [1, *body, 252])[IDENTITY_LABELS]
    assert model.model._last_lora_indices.tolist() == [[0, 0, 0, 0, 0, 1]]
    assert not torch.allclose(alone, reference, atol=1e-2)


def test_a_head_names_an_adapter_slot(tmp_path):
    probe = _verdict_probe(tmp_path / "probe", 0)
    with pytest.raises(ValueError, match="not an adapter slot"):
        _compose(
            tmp_path,
            [("ad", _sr(tmp_path / "ad")), ("probe", probe), ("on_probe", probe)],
            [{"type": "lora"}, _probe(0), _probe(1, adapter="probe")],
        )


# --------------------------------------------------------------------------- #
# Compact rows
# --------------------------------------------------------------------------- #
def test_head_slots_take_no_lora_rows_and_order_is_free(tmp_path):
    """[probe, lora, probe]: one LoRA row, a two-row probe bank, any order."""
    model = _compose(
        tmp_path,
        [
            ("first", _verdict_probe(tmp_path / "first", 0)),
            ("ad", _lora(tmp_path / "ad")),
            ("last", _verdict_probe(tmp_path / "last", 1)),
        ],
        [_probe(0), {"type": "lora"}, _probe(1)],
    )
    config = model.config
    assert config.num_lora_rows == 1
    assert stream_routing.lora_row_lut(config).tolist() == [0, 0, 1, 0]
    assert stream_routing.row_lut(config, "probe").tolist() == [0, 1, 0, 2]
    assert model.model.probe_head.weight.shape[0] == 2
    lora_b = next(p for n, p in model.named_parameters() if "lora_B" in n)
    assert lora_b.shape[0] == 1

    # The LoRA slot, in the middle, routes to LoRA row 1.
    _last_row(model, [10, 251, 20, 30])
    assert model.model._last_lora_indices.tolist() == [[0, 2, 2, 2]]
    # Each probe slot reads its own bank row.
    assert int(_last_row(model, [10, 20, 250]).argmax()) == LABELS[0]
    assert int(_last_row(model, [10, 20, 252]).argmax()) == LABELS[1]


def test_heads_only_checkpoint_has_no_lora(tmp_path):
    model = _compose(
        tmp_path,
        [
            ("first", _verdict_probe(tmp_path / "first", 0)),
            ("last", _verdict_probe(tmp_path / "last", 1)),
        ],
        [_probe(0), _probe(1)],
    )
    assert model.config.num_lora_rows == 0
    assert model.config.lora_target_modules == []
    assert not [n for n, _ in model.named_parameters() if "lora_" in n]
    assert int(_last_row(model, [10, 20, 250]).argmax()) == LABELS[0]
    assert int(_last_row(model, [10, 20, 251]).argmax()) == LABELS[1]


def test_pre_streams_layout_is_unchanged(tmp_path):
    """No ``streams``: every slot keeps its LoRA row and its probe-bank row."""
    model = _compose(
        tmp_path,
        [
            ("ad", _lora(tmp_path / "ad")),
            ("first", _verdict_probe(tmp_path / "first", 0)),
            ("last", _verdict_probe(tmp_path / "last", 1)),
        ],
        None,
        adapter_kinds=["lora", "probe", "probe"],
        probe_label_token_ids=[None, LABELS, LABELS],
    )
    assert model.config.streams is None
    assert model.model.probe_head.weight.shape[0] == 3
    lora_b = next(p for n, p in model.named_parameters() if "lora_B" in n)
    assert lora_b.shape[0] == 3
    assert int(_last_row(model, [10, 20, 251]).argmax()) == LABELS[0]
    assert int(_last_row(model, [10, 20, 252]).argmax()) == LABELS[1]


# --------------------------------------------------------------------------- #
# Compose checks read LoRA rows, not slots
# --------------------------------------------------------------------------- #
def _compose_probe_before_sr(tmp_path):
    """A probe in slot 1 and an SR adapter in slot 2: the adapter owns LoRA row 0."""
    probe = _probe_folder(tmp_path / "probe", torch.randn(len(LABELS), HIDDEN))
    sr = _sr(tmp_path / "sr")
    model = _compose(
        tmp_path, [("tox", probe), ("sr", sr)], [_probe(0), {"type": "lora"}]
    )
    return model, str(sr), str(tmp_path / "base")


def test_cross_stream_check_finds_an_empty_adapter_after_a_head(tmp_path, capsys):
    """The check walks LoRA rows: an adapter behind a head is still checked."""
    from granite_switch.composer.validator import validate_cross_stream_population

    model, _, _ = _compose_probe_before_sr(tmp_path)
    capsys.readouterr()
    validate_cross_stream_population(model)
    assert "OK: every adapter has cross_stream weights" in capsys.readouterr().out
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "cross_stream" in name and "lora_B" in name:
                p.zero_()
    validate_cross_stream_population(model)
    assert "WARNING" in capsys.readouterr().out


def test_population_table_has_one_column_per_lora_row(tmp_path):
    """Given the LoRA slots only (as the compose report passes them), each
    column reads that adapter's own row."""
    from granite_switch.composer.arch import resolve_arch
    from granite_switch.composer.reporting.population_table import (
        generate_adapter_population_table,
    )

    model, sr, base = _compose_probe_before_sr(tmp_path)
    analysis = model._build_mappings["source_analysis"]
    table = generate_adapter_population_table(
        model,
        [sr],
        adapter_names=["sr"],
        arch=resolve_arch(base, dual_stream=True),
        target_module_sets=analysis["adapter_targets"],
    )
    assert table["adapter_names"] == ["sr"]
    cross = {k: v for k, v in table["entries"].items() if k.startswith("cross_stream")}
    assert cross and all(v == {"sr": "copied"} for v in cross.values()), cross
