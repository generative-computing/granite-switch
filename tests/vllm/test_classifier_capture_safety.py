# SPDX-License-Identifier: Apache-2.0
"""CPU method tests for token-aligned, score-at-layer classification.

GPU capture/replay and packed-output transport are tested in
``test_classifier_chunked_prefill.py``. Request-boundary validation is deferred.
"""

import pytest
import torch

pytest.importorskip("vllm")

from granite_switch.vllm.granite_switch_model import (
    GraniteSwitchForCausalLM,
    GraniteSwitchModel,
)


class _StubClassifierHead:
    """Score = slot id * hidden[:, 0], so a score shows which slot and source it read."""

    def __call__(self, x, classifier_indices):
        return (classifier_indices.float() * x[:, 0]).unsqueeze(-1).expand(-1, 2)


class _Model(torch.nn.Module):
    """Stands in for ``GraniteSwitchModel`` with the attributes its __init__ builds.

    ``read_layers``/``kinds`` are per 1-based slot; None reads the final layer.
    """

    _score_group = GraniteSwitchModel._score_group

    def __init__(self, slot_of_marker, read_layers=None, kinds=None):
        super().__init__()
        num_slots = max(slot_of_marker.values())
        read_layers = read_layers or [None] * num_slots
        is_classifier = [k == "classifier" for k in kinds or ["classifier"] * num_slots]
        self.config = type("Cfg", (), {"max_classifier_labels": 2})()
        self.classifier_head = _StubClassifierHead()
        self.classifier_control_token_ids_t = torch.tensor(sorted(slot_of_marker))
        self.classifier_slot_mask_t = torch.tensor([False, *is_classifier])
        slots_by_layer = {}
        for slot, (is_clf, layer) in enumerate(
            zip(is_classifier, read_layers), start=1
        ):
            if is_clf:
                slots_by_layer.setdefault(layer, []).append(slot)
        self._classifier_final_layer_group = None
        self._classifier_layer_groups = {}
        for layer, slot_ids in slots_by_layer.items():
            name = f"_classifier_slots_{'final' if layer is None else layer}"
            self.register_buffer(name, torch.tensor(slot_ids))
            if layer is None:
                self._classifier_final_layer_group = name
            else:
                self._classifier_layer_groups[layer] = name


def _run(model, slot_of_marker, tokens, input_ids_override=...):
    """Call ``_compute_classifier_outputs`` on a flat token list; hidden[t] = t + 1."""
    indices, current = [], 0
    for t in tokens:
        current = slot_of_marker.get(t, current)
        indices.append(current)
    input_ids = (
        torch.tensor(tokens) if input_ids_override is ... else input_ids_override
    )
    hidden = torch.arange(1.0, len(tokens) + 1).unsqueeze(1).expand(-1, 4)
    return GraniteSwitchModel._compute_classifier_outputs(
        model, input_ids, hidden, torch.tensor(indices), torch.zeros(len(tokens), 2)
    )


def _score(model, layer, hidden_value, indices, scores):
    hidden = torch.full((len(indices), 4), hidden_value)
    return GraniteSwitchModel._score_group(
        model,
        model._classifier_layer_groups[layer],
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


def test_no_input_ids_is_all_zero_slot():
    model = _Model({90: 2, 91: 3})
    _, slot = _run(model, {90: 2, 91: 3}, [5, 90, 6], input_ids_override=None)
    assert slot.tolist() == [0, 0, 0]


def test_final_layer_scoring_leaves_intermediate_slot_scores_untouched():
    model = _Model({90: 2, 91: 3}, read_layers=[None, 1, None])
    pre_scores = torch.tensor([[0.0, 0.0], [99.0, 99.0], [0.0, 0.0]])
    scores, slot = GraniteSwitchModel._compute_classifier_outputs(
        model,
        torch.tensor([5, 90, 91]),
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
    assert model._classifier_layer_groups == {}
    assert getattr(model, model._classifier_final_layer_group).tolist() == [1, 2]
    scores, slot = _run(model, {90: 1, 91: 2}, [90, 5, 91])
    assert slot.tolist() == [1, 0, 2]
    assert scores[0].tolist() == [1.0, 1.0]
    assert scores[2].tolist() == [6.0, 6.0]


def test_lora_slots_form_no_group():
    model = _Model({91: 2}, kinds=["lora", "classifier"])
    assert getattr(model, model._classifier_final_layer_group).tolist() == [2]
    assert model._classifier_layer_groups == {}
    _, slot = _run(model, {91: 2}, [5, 91, 6])
    assert slot.tolist() == [0, 2, 0]


def test_profiling_random_slot_on_a_lora_slot_leaves_logits_untouched():
    """vLLM's profiling sampler run feeds random hidden states; 0.75 rounds to LoRA slot 1."""
    wrapper = GraniteSwitchForCausalLM.__new__(GraniteSwitchForCausalLM)
    torch.nn.Module.__init__(wrapper)
    wrapper.model = _Model({91: 2}, kinds=["lora", "classifier"])
    wrapper.config = type(
        "Cfg", (), {"classifier_label_token_ids": [None, [100, 200]]}
    )()
    logits = torch.randn(1, 300)
    expected = logits.clone()
    out = GraniteSwitchForCausalLM._apply_classifier_verdict(
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
    scores, slots = GraniteSwitchModel._compute_classifier_outputs(
        model,
        torch.tensor([90, 91]),
        torch.full((2, 4), 5.0),
        torch.tensor([1, 2]),
        scores,
    )
    assert scores.tolist() == [[5.0, 5.0], [6.0, 6.0]]
    assert slots.tolist() == [1.0, 2.0]
