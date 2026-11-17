# SPDX-License-Identifier: Apache-2.0
"""The stream API on composed models: a head stream type plugs in with a
config entry, a module and a weights file, and nothing else
(``granite_switch.streams``).

CPU, tiny models composed by the real composer. Covers:

* an example plugin type (``tests/shared/stream_plugins/mlp_head.py``) run end
  to end, with ``merge``;
* a head module's wider, other-dtype output fitted to its score row;
* a head reads the base model, even after a Shadow Residual adapter's token;
* compact LoRA rows: head slots take no row in the LoRA tensors;
* the pre-streams checkpoint layout, unchanged.

What needs no model is tested in ``tests/unit/``: the registry, stream
entries and output modes (``test_stream_entries.py``); the routing tables,
head modules and strict head loading (``test_stream_heads.py``).
"""

import pytest
import torch
from safetensors.torch import save_file

from granite_switch.streams import HeadType
from granite_switch.streams import routing as stream_routing
from granite_switch.streams.registry import _REGISTRY
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
from tests.unit.test_stream_entries import LABELS, _probe


def _last_row(model, ids):
    with torch.no_grad():
        return model(input_ids=torch.tensor([ids]), use_cache=False).logits[0, -1]


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


def test_probe_rejects_unknown_settings(tmp_path):
    with pytest.raises(ValueError, match="probe slot 1 accepts no settings"):
        _compose(
            tmp_path,
            [("probe", _verdict_probe(tmp_path / "head", 0))],
            [_probe(0, settings={"normalize": True})],
        )


@pytest.mark.parametrize("attr", ["norm", "switch", "layers", "forward", "config"])
def test_head_cannot_replace_model_attributes(tmp_path, monkeypatch, attr):
    class CollisionType(_WideType):
        name = "test_collision"
        module_name = attr

    monkeypatch.setitem(_REGISTRY, CollisionType.name, CollisionType)
    folder = tmp_path / "head"
    folder.mkdir()
    with pytest.raises(ValueError, match="already a model attribute"):
        _compose(
            tmp_path, [("head", folder)], [{**_probe(0), "type": CollisionType.name}]
        )


@pytest.mark.parametrize(
    "shape,dtype",
    [((1, 1), torch.float32), ((1, 2), torch.long), ((2, 2), torch.float32)],
)
def test_bad_plugin_output_fails_with_its_name(tmp_path, monkeypatch, shape, dtype):
    model = _compose(
        tmp_path, [("probe", _verdict_probe(tmp_path / "head", 0))], [_probe(0)]
    )
    monkeypatch.setattr(
        model.model.probe_head,
        "forward",
        lambda h, rows: torch.zeros(shape, dtype=dtype),
    )
    with pytest.raises(ValueError, match="Head 'probe'.*at least 2 columns"):
        _last_row(model, [10, CONTROL_TOKEN_ID])


def test_head_buffers_and_plain_weight_copies_survive_checkpoint_loading(tmp_path):
    from granite_switch.hf import load_model
    from tests.shared.stream_plugins.buffered_head import BufferedHeadType

    folder = tmp_path / "head"
    folder.mkdir()
    save_file(
        {"weight": torch.zeros(2, HIDDEN), "bias": torch.tensor([2.0, -3.0])},
        str(folder / "head.safetensors"),
    )
    model = _compose(
        tmp_path, [("head", folder)], [{**_probe(0), "type": BufferedHeadType.name}]
    )
    ids = [10, CONTROL_TOKEN_ID]
    expected = _last_row(model, ids)
    torch.testing.assert_close(expected[LABELS], torch.tensor([2.0, -3.0]))
    model.save_pretrained(tmp_path / "checkpoint")
    loaded = load_model(str(tmp_path / "checkpoint")).eval()
    torch.testing.assert_close(_last_row(loaded, ids), expected)
    # vLLM's strict loading of the same tensors: tests/unit/test_stream_heads.py.


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
# A head reads the base model
# --------------------------------------------------------------------------- #
def test_head_reads_the_base_model_after_an_sr_adapter(tmp_path):
    """Shadow Residual writes no K/V, so an adapter's tokens leave the base model
    as it was: an identity head reads the same state with or without them."""
    model = _compose(
        tmp_path,
        [
            ("ad", _sr(tmp_path / "ad")),
            ("probe", _probe_folder(tmp_path / "probe", torch.eye(HIDDEN))),
        ],
        [{"type": "lora"}, _probe(0, labels=IDENTITY_LABELS)],
    )
    body = [5, 17, 42, 99]
    after_adapter = _last_row(model, [250, *body, 251])[IDENTITY_LABELS]
    assert model.model._last_lora_indices.tolist() == [[1, 1, 1, 1, 1, 0]]
    # Token 1 embeds as the adapter's control token does (its substitute).
    plain = _last_row(model, [1, *body, 251])[IDENTITY_LABELS]
    assert model.model._last_lora_indices.tolist() == [[0, 0, 0, 0, 0, 0]]
    torch.testing.assert_close(after_adapter, plain, atol=1e-5, rtol=0)


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
