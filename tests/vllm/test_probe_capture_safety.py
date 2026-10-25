# SPDX-License-Identifier: Apache-2.0
"""CPU method tests for token-aligned, score-at-layer probe verdicts.

GPU capture/replay and packed-output transport are tested in
``test_probe_chunked_prefill.py``. Request-boundary validation is deferred.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm")

from vllm.sequence import IntermediateTensors

from granite_switch.vllm.decoder.interface import (
    LoRADecoderInterface,
    SRDecoderInterface,
)
from granite_switch.vllm.granite_switch_model import (
    GraniteSwitchForCausalLM,
    GraniteSwitchModel,
)


class _StubProbeHead:
    """Score = slot id * hidden[:, 0], so a score shows which slot and source it read."""

    def __call__(self, x, probe_indices):
        return (probe_indices.to(x.dtype) * x[:, 0]).unsqueeze(-1).expand(-1, 2)


class _Model(torch.nn.Module):
    """Stands in for ``GraniteSwitchModel`` with the attributes its __init__ builds.

    ``read_layers``/``kinds`` are per 1-based slot; None reads the final layer.
    """

    _score_group = GraniteSwitchModel._score_group

    def __init__(self, slot_of_marker, read_layers=None, kinds=None):
        super().__init__()
        num_slots = max(slot_of_marker.values())
        read_layers = read_layers or [None] * num_slots
        is_probe = [k == "probe" for k in kinds or ["probe"] * num_slots]
        self.config = type("Cfg", (), {"max_probe_labels": 2})()
        self.probe_head = _StubProbeHead()
        self.probe_slot_mask_t = torch.tensor([False, *is_probe])
        slots_by_layer = {}
        for slot, (is_clf, layer) in enumerate(zip(is_probe, read_layers), start=1):
            if is_clf:
                slots_by_layer.setdefault(layer, []).append(slot)
        self._probe_final_layer_group = None
        self._probe_layer_groups = {}
        for layer, slot_ids in slots_by_layer.items():
            name = f"_probe_slots_{'final' if layer is None else layer}"
            self.register_buffer(name, torch.tensor(slot_ids))
            if layer is None:
                self._probe_final_layer_group = name
            else:
                self._probe_layer_groups[layer] = name


def _run(model, slot_of_marker, tokens):
    """Call ``_compute_probe_outputs`` on a flat token list; hidden[t] = t + 1."""
    indices = [slot_of_marker.get(t, 0) for t in tokens]
    hidden = torch.arange(1.0, len(tokens) + 1).unsqueeze(1).expand(-1, 4)
    return GraniteSwitchModel._compute_probe_outputs(
        model, hidden, torch.tensor(indices), torch.zeros(len(tokens), 2)
    )


def _score(model, layer, hidden_value, indices, scores):
    hidden = torch.full((len(indices), 4), hidden_value)
    return GraniteSwitchModel._score_group(
        model,
        model._probe_layer_groups[layer],
        hidden,
        torch.tensor(indices),
        scores,
    )


def test_every_marker_token_gets_its_slot_and_score():
    model = _Model({90: 2, 91: 3})
    scores, slot = _run(model, {90: 2, 91: 3}, [5, 6, 90, 7, 8, 9, 91])
    assert slot.tolist() == [0, 0, 2, 0, 0, 0, 3]
    assert scores[2].tolist() == [6.0, 6.0]  # slot 2 * hidden 3
    assert scores[6].tolist() == [21.0, 21.0]  # slot 3 * hidden 7


def test_marker_is_identified_by_id_at_any_position():
    model = _Model({90: 2, 91: 3})
    _, slot = _run(model, {90: 2, 91: 3}, [90, 5, 6, 91])
    assert slot.tolist() == [2, 0, 0, 3]


def test_repeated_marker_of_the_same_slot_is_reported_each_time():
    model = _Model({90: 2, 91: 3})
    _, slot = _run(model, {90: 2, 91: 3}, [90, 5, 90, 6])
    assert slot.tolist() == [2, 0, 2, 0]


def test_no_marker_is_all_zero():
    model = _Model({90: 2, 91: 3})
    scores, slot = _run(model, {90: 2, 91: 3}, [5, 6, 7])
    assert slot.tolist() == [0, 0, 0]
    assert not scores.any()


def test_transferred_marker_indices_need_no_input_ids():
    model = _Model({90: 2, 91: 3})
    scores, slot = GraniteSwitchModel._compute_probe_outputs(
        model, torch.ones(3, 4), torch.tensor([0, 2, 0]), torch.zeros(3, 2)
    )
    assert slot.tolist() == [0, 2, 0]
    assert scores.tolist() == [[0.0, 0.0], [2.0, 2.0], [0.0, 0.0]]


def test_final_layer_scoring_leaves_intermediate_slot_scores_untouched():
    model = _Model({90: 2, 91: 3}, read_layers=[None, 1, None])
    pre_scores = torch.tensor([[0.0, 0.0], [99.0, 99.0], [0.0, 0.0]])
    scores, slot = GraniteSwitchModel._compute_probe_outputs(
        model,
        torch.ones(3, 4),
        torch.tensor([0, 2, 3]),
        pre_scores,
    )
    assert slot.tolist() == [0, 2, 3]
    assert scores[1].tolist() == [99.0, 99.0]  # slot 2 reads layer 1
    assert scores[2].tolist() == [3.0, 3.0]


def test_each_intermediate_layer_scores_only_its_own_slots():
    model = _Model({90: 1, 91: 2}, read_layers=[3, 7])
    scores = _score(model, 3, 2.0, [0, 1, 2], torch.zeros(3, 2))
    assert scores[1].tolist() == [2.0, 2.0]
    assert scores[2].tolist() == [0.0, 0.0]
    scores = _score(model, 7, 5.0, [0, 1, 2], scores)
    assert scores[1].tolist() == [2.0, 2.0]  # layer 3's value kept
    assert scores[2].tolist() == [10.0, 10.0]


def test_slots_sharing_a_layer_form_one_group():
    model = _Model({90: 1, 91: 2})
    assert model._probe_layer_groups == {}
    assert getattr(model, model._probe_final_layer_group).tolist() == [1, 2]
    scores, slot = _run(model, {90: 1, 91: 2}, [90, 5, 91])
    assert slot.tolist() == [1, 0, 2]
    assert scores[0].tolist() == [1.0, 1.0]
    assert scores[2].tolist() == [6.0, 6.0]


def test_lora_slots_form_no_group():
    model = _Model({91: 2}, kinds=["lora", "probe"])
    assert getattr(model, model._probe_final_layer_group).tolist() == [2]
    assert model._probe_layer_groups == {}
    _, slot = _run(model, {91: 2}, [5, 91, 6])
    assert slot.tolist() == [0, 2, 0]


def test_profiling_random_slot_on_a_lora_slot_leaves_logits_untouched():
    """vLLM's profiling sampler run feeds random hidden states; 0.75 rounds to LoRA slot 1."""
    wrapper = GraniteSwitchForCausalLM.__new__(GraniteSwitchForCausalLM)
    torch.nn.Module.__init__(wrapper)
    wrapper.model = _Model({91: 2}, kinds=["lora", "probe"])
    wrapper.config = type("Cfg", (), {"probe_label_token_ids": [None, [100, 200]]})()
    logits = torch.randn(1, 300)
    expected = logits.clone()
    out = GraniteSwitchForCausalLM._apply_probe_verdict(
        wrapper, logits, torch.randn(1, 2), torch.tensor([0.75])
    )
    assert torch.equal(out, expected)


def test_scoring_resolves_replaced_registered_buffers():
    model = _Model({90: 1, 91: 2}, read_layers=[None, 0])
    old_buffers = list(model.buffers())
    # Device moves replace buffers; poison the old storage to catch stale aliases.
    model._apply(lambda tensor: tensor.clone())
    for tensor in old_buffers:
        tensor.zero_()
    scores = _score(model, 0, 3.0, [1, 2], torch.zeros(2, 2))
    scores, slots = GraniteSwitchModel._compute_probe_outputs(
        model,
        torch.full((2, 4), 5.0),
        torch.tensor([1, 2]),
        scores,
    )
    assert scores.tolist() == [[5.0, 5.0], [6.0, 6.0]]
    assert slots.tolist() == [1.0, 2.0]


@pytest.mark.parametrize("sr", [False, True], ids=["lora", "sr"])
@pytest.mark.parametrize("stages", [2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_pipeline_handoff_without_later_token_ids(monkeypatch, sr, stages, dtype):
    """Run the real forward and decoder strategies, with deterministic layer outputs.

    A layer-0 probe crosses the boundary; another probe and final norm are scored
    on the last rank. With three stages, the middle rank only relays scores.
    Distinct values expose resets, wrong layers, stream selection and reliance
    on input_ids. The switch keeps the latest slot, like MultiSwitch, so only
    the markers themselves may carry a probe index. Markers route out of
    the LoRA stream; the trailing LoRA token reads the adapted SR stream. The
    token slice matches vLLM's PP buffer contract.
    """
    from granite_switch.vllm import granite_switch_model as model_module

    class Switch:
        def __call__(self, input_ids, **kwargs):
            control_slot = {89: 1, 90: 2, 91: 3, 92: 4}
            active, indices = 0, []
            for token in input_ids.tolist():
                active = control_slot.get(token, active)
                indices.append(active)
            return torch.tensor(indices), input_ids

        def split_indices(self, indices):
            is_probe = indices >= 2
            zeros = torch.zeros_like(indices)
            return (
                torch.where(is_probe, zeros, indices),
                torch.where(is_probe, indices, zeros),
            )

    class Layer(torch.nn.Module):
        def forward(self, positions, hidden_states, residual=None):
            if sr:
                base, adapted = hidden_states.chunk(2)
                return torch.cat([base + 1, adapted + 10])
            if residual is None:
                residual = torch.zeros_like(hidden_states)
            return hidden_states + 1, residual

    model = _Model(
        {90: 2, 91: 3, 92: 4},
        read_layers=[None, 0, stages - 1, None],
        kinds=["lora", "probe", "probe", "probe"],
    )
    model.config.hidden_size = 4
    model.config.embedding_multiplier = 1.0
    model.config.fused_add_norm = False
    model.switch = Switch()
    model.adapter_token_ids = torch.tensor([89, 90, 91, 92])
    model.probe_control_token_ids_t = torch.tensor([90, 91, 92])
    model.lora_meta = model.lora_ctx = None
    model.decoder_interface = SRDecoderInterface() if sr else LoRADecoderInterface()
    model.layers = torch.nn.ModuleList([Layer() for _ in range(stages)])
    model.norm = torch.nn.Identity()
    model._compute_probe_outputs = GraniteSwitchModel._compute_probe_outputs.__get__(
        model
    )
    model.start_layer, model.end_layer = 0, 1
    monkeypatch.setattr(
        model_module,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=False),
    )
    wire = GraniteSwitchModel.forward(
        model,
        torch.tensor([5, 90, 91, 92, 89]),
        torch.arange(5),
        inputs_embeds=torch.ones(5, 4, dtype=dtype),
    )[1:]
    assert isinstance(wire, IntermediateTensors)
    assert wire["probe_indices"].tolist() == [2, 3, 4, 0]
    assert wire["probe_scores"][:, 0].tolist() == [4, 0, 0, 0]
    buffers = GraniteSwitchModel.make_empty_intermediate_tensors(
        model, 4, dtype, torch.device("cpu")
    )
    assert buffers.tensors.keys() == wire.tensors.keys()
    for key, tensor in wire.items():
        assert tensor.shape == buffers[key].shape
        assert tensor.dtype == buffers[key].dtype
        buffers[key].copy_(tensor)

    for rank in range(1, stages):
        model.start_layer, model.end_layer = rank, rank + 1
        group = SimpleNamespace(is_first_rank=False, is_last_rank=rank == stages - 1)
        monkeypatch.setattr(model_module, "get_pp_group", lambda group=group: group)
        packed = GraniteSwitchModel.forward(
            model, None, torch.arange(4), intermediate_tensors=buffers
        )
        if not group.is_last_rank:
            assert torch.equal(packed["probe_scores"], wire["probe_scores"])
            for key, tensor in packed.items():
                buffers[key].copy_(tensor)
    assert packed[:, -1].tolist() == [2, 3, 4, 0]
    # Markers read the base stream; the LoRA token reads the adapted SR stream.
    source = 1 + stages
    assert packed[:, 4].tolist() == [4, 3 * source, 4 * source, 0]
    assert packed[-1, 0] == 1 + (10 if sr else 1) * stages


def test_sr_pipeline_stack_partitions_post_attention_metadata():
    """A later PP stage must retain 2*M through attention and metadata slicing.

    Concatenating independently symbolic PP inputs produces a late 2*((A+B)//2)
    slice bound. vLLM moves that metadata getitem to its producer's partition,
    creating a cycle. The real SR entry hook must allow splitting and reuse for
    changed token counts, without changing the two streams' values.
    """
    from vllm.compilation.backends import split_graph

    @torch.library.custom_op("granite_switch_test::sr_pp_attention", mutates_args=())
    def attention(query: torch.Tensor) -> torch.Tensor:
        return query.clone()

    @attention.register_fake
    def fake_attention(query):
        return torch.empty_like(query)

    interface = SRDecoderInterface()
    compilations = []

    def backend(graph, inputs):
        compilations.append(graph)
        return split_graph(graph, ["granite_switch_test::sr_pp_attention"])[0].forward

    def stage(base, adapted, indices):
        metadata = torch.cat([torch.zeros_like(indices), indices]).unsqueeze(0)
        hs = interface.enter_decoder_stack(base, adapted).hs
        m = hs.shape[0] // 2
        query = torch.cat([hs[:m], hs[m:]], dim=1)
        attended = attention(query)
        stacked = torch.cat([attended[:, :4], attended[:, 4:]], dim=0)
        remapped = metadata[0, : stacked.shape[0]]
        return stacked + remapped.unsqueeze(-1).to(stacked.dtype)

    compiled = torch.compile(stage, backend=backend, dynamic=True, fullgraph=True)
    for m in (4, 6):
        base = torch.arange(m * 4, dtype=torch.float32).reshape(m, 4)
        adapted = base + 17
        indices = torch.arange(1, m + 1)
        for tensor in (base, adapted, indices):
            torch._dynamo.mark_dynamic(tensor, 0)
        expected = torch.cat([base, adapted + indices.unsqueeze(-1)])
        assert torch.equal(compiled(base, adapted, indices), expected)
    assert len(compilations) == 1
