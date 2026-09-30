# SPDX-License-Identifier: Apache-2.0
"""Classifier-substitute path tests for GraniteSwitchForCausalLM (HF backend).

A classifier slot is an alternative to a LoRA adapter: it shares the adapter
index/control-token machinery, the switch splits classifier positions out of
the LoRA stream, and the classifier head reads the final hidden state to emit
ONE verdict per request. The verdict is read at the control token's own position.

The verdict exits as a GENERATED LABEL WORD, matching the vLLM backend: the LM
head rewrites the classifier request's last logit row so the vocab is -inf
except at ``classifier_label_token_ids``, where the per-label verdict scores are
placed. There is no separate float ``classifier_logits`` output field.

CPU, random weights, no checkpoint — mirrors tests/hf/test_model_forward.py.
"""

import pytest
import torch

from granite_switch.config import GraniteSwitchConfig
from granite_switch.hf import GraniteSwitchForCausalLM


def _set_adapter_token_ids(model, token_ids):
    model.model.adapter_token_ids.data = torch.tensor(token_ids, dtype=torch.long)


def _classifier_config(adapter_kinds, num_labels=2, label_token_ids=(100, 200)):
    # Full-width label token ids (length num_adapters, None on LoRA slots). Every
    # classifier slot gets the same ids here; mixed counts are covered by the
    # compose tests.
    label_token_ids = list(label_token_ids)
    return GraniteSwitchConfig(
        vocab_size=300,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,  # 1 switch + 2 decoder
        num_attention_heads=4,
        num_key_value_heads=4,
        num_adapters=2,
        adapter_token_ids=[250, 251],
        adapter_substitute_token_ids=[1, 1],
        adapter_names=["adapter_1", "adapter_2"],
        adapter_kinds=adapter_kinds,
        classifier_label_token_ids=[
            list(label_token_ids) if k == "classifier" else None for k in adapter_kinds
        ],
        max_lora_rank=4,
        adapter_ranks=[4, 4],
        switch_head_dim=16,
    )


def test_classifier_head_built_only_when_slot_present():
    m_lora = GraniteSwitchForCausalLM(_classifier_config(["lora", "lora"]))
    m_clf = GraniteSwitchForCausalLM(_classifier_config(["lora", "classifier"]))
    assert m_lora.model.classifier_head is None
    assert m_clf.model.classifier_head is not None
    # Bank sized to the full index space so slot = index - 1 addresses it.
    assert m_clf.model.classifier_head.num_classifier_slots == 2
    assert m_clf.model.classifier_head.num_labels == 2


def test_classifier_rewrites_last_logit_row_to_label_word():
    # adapter 2 (token 251) is a classifier. Its verdict rides the LM logits:
    # the last prompt row is -inf everywhere except at the label token ids,
    # where the per-label verdict scores land — so decoding emits a label word.
    config = _classifier_config(["lora", "classifier"], label_token_ids=(100, 200))
    model = GraniteSwitchForCausalLM(config)
    _set_adapter_token_ids(model, config.adapter_token_ids)
    model.eval()

    # Marker (251) at the end; this test checks the label-word rewrite of the
    # emitted row. The read position is pinned in
    # test_classifier_reads_marker_hidden_state.
    input_ids = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 251]])
    with torch.no_grad():
        out = model(input_ids=input_ids)

    last_row = out.logits[0, -1, :]  # [vocab]
    # Single classifier slot -> classifier position 0; its own label ids.
    label_ids = config.classifier_label_token_ids[1]  # slot 2 -> index 1
    n = len(label_ids)
    non_label = [t for t in range(config.vocab_size) if t not in label_ids]

    # Every non-label token is blanked; the label ids carry the verdict.
    assert torch.isinf(last_row[non_label]).all()
    assert (last_row[non_label] < 0).all()
    assert torch.isfinite(last_row[torch.tensor(label_ids)]).all()
    # The label-id scores equal the classifier head's per-label verdict, sliced
    # to this slot's real label count (padded columns are never scattered).
    verdict = model.model._last_classifier_logits[0, :n]  # [n_labels]
    assert torch.allclose(last_row[torch.tensor(label_ids)], verdict.to(last_row.dtype))
    # The argmax (emitted token) is one of the label ids.
    assert int(last_row.argmax()) in label_ids


def test_non_classifier_request_logits_untouched():
    # No classifier token (251) present -> last token is base (index 0), the
    # head's last-token verdict row is zero, and NO row is blanked: a
    # non-classifier request keeps a full finite vocab row.
    config = _classifier_config(["lora", "classifier"])
    model = GraniteSwitchForCausalLM(config)
    _set_adapter_token_ids(model, config.adapter_token_ids)
    model.eval()

    input_ids = torch.tensor([[10, 20, 30, 40]])
    with torch.no_grad():
        out = model(input_ids=input_ids)

    assert (model.model._last_classifier_indices == 0).all()
    # Full finite vocab row — nothing blanked.
    assert torch.isfinite(out.logits[0, -1, :]).all()


def test_mixed_batch_only_classifier_row_rewritten():
    # Batch: row 0 is a classifier request (251 fires), row 1 is plain LM.
    config = _classifier_config(["lora", "classifier"], label_token_ids=(100, 200))
    model = GraniteSwitchForCausalLM(config)
    _set_adapter_token_ids(model, config.adapter_token_ids)
    model.eval()

    input_ids = torch.tensor(
        [
            [10, 20, 251, 40, 50],  # classifier
            [10, 20, 30, 40, 50],  # plain
        ]
    )
    with torch.no_grad():
        out = model(input_ids=input_ids)

    label_ids = torch.tensor(config.classifier_label_token_ids[1])  # slot 2
    non_label = [
        t for t in range(config.vocab_size) if t not in set(label_ids.tolist())
    ]

    # Classifier row rewritten; plain row fully finite.
    assert torch.isinf(out.logits[0, -1, non_label]).all()
    assert torch.isfinite(out.logits[1, -1, :]).all()


def test_classifier_is_detect_only_decode_step_raises():
    # A classifier request is prefill-only: it classifies the prompt and stops.
    # A cached decode step (generation) on a classifier model is misuse and
    # must fail loud rather than produce a verdict over generated tokens.
    config = _classifier_config(["lora", "classifier"])
    model = GraniteSwitchForCausalLM(config)
    _set_adapter_token_ids(model, config.adapter_token_ids)
    model.eval()

    input_ids = torch.tensor([[10, 20, 251, 40, 50]])
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    assert out.logits.shape == (1, 5, config.vocab_size)

    with pytest.raises(RuntimeError, match="detect-only"):
        with torch.no_grad():
            model(
                input_ids=torch.tensor([[60]]),
                past_key_values=out.past_key_values,
                use_cache=True,
            )


def test_classifier_missing_label_token_ids_fails_loud():
    # A classifier slot with no resolved label token ids cannot emit a label word.
    # Rejected when the config is built, so a bad checkpoint fails at load rather
    # than at the first inference.
    with pytest.raises(ValueError, match="classifier_label_token_ids"):
        GraniteSwitchConfig(
            vocab_size=300,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=4,
            num_adapters=2,
            adapter_token_ids=[250, 251],
            adapter_substitute_token_ids=[1, 1],
            adapter_names=["adapter_1", "adapter_2"],
            adapter_kinds=["lora", "classifier"],
            classifier_label_token_ids=None,
            max_lora_rank=4,
            adapter_ranks=[4, 4],
            switch_head_dim=16,
        )


def test_lora_only_path_unaffected():
    # A model with no classifier slot behaves exactly as before: no classifier
    # head, no verdict, full finite LM logits.
    config = _classifier_config(["lora", "lora"])
    model = GraniteSwitchForCausalLM(config)
    _set_adapter_token_ids(model, config.adapter_token_ids)
    model.eval()

    input_ids = torch.tensor([[10, 250, 20, 251, 30, 40]])
    with torch.no_grad():
        out = model(input_ids=input_ids)

    assert model.model.classifier_head is None
    assert model.model._last_classifier_logits is None
    assert torch.isfinite(out.logits).all()
    assert out.logits.shape == (1, 6, config.vocab_size)


def test_lora_generation_on_mixed_model_decodes():
    # A model composed WITH a classifier slot also serves plain LoRA/base
    # generation. Firing the LoRA (adapter 1, token 250) — not the classifier —
    # must be allowed to decode multiple tokens: the detect-only guard keys on an
    # actually-fired classifier, not on the mere presence of a classifier head
    # bank. (Regression: the guard previously raised on ANY cached decode step of
    # a classifier-composed model, crashing all multi-token LoRA generation.)
    config = _classifier_config(["lora", "classifier"])
    model = GraniteSwitchForCausalLM(config)
    _set_adapter_token_ids(model, config.adapter_token_ids)
    model.eval()
    assert model.model.classifier_head is not None  # mixed model has the bank

    # Prefill fires the LoRA (250), never the classifier (251).
    input_ids = torch.tensor([[10, 250, 20, 30, 40]])
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    # No classifier fired -> no verdict, full finite vocab row.
    assert model.model._last_classifier_logits is None
    assert torch.isfinite(out.logits).all()

    # Cached decode step of the same (LoRA) request must NOT raise and must
    # yield finite logits — classifier_indices stays zero, so the detect-only
    # guard does not fire.
    with torch.no_grad():
        step = model(
            input_ids=torch.tensor([[60]]),
            past_key_values=out.past_key_values,
            use_cache=True,
        )
    assert step.logits.shape == (1, 1, config.vocab_size)
    assert torch.isfinite(step.logits).all()
    assert model.model._last_classifier_logits is None


def test_classifier_reads_marker_hidden_state():
    """The verdict is read at the marker.

    Reading the marker's own row is what makes the read point independent of where
    the chat template puts the token. The exact position is pinned by capturing (a)
    the final post-norm hidden states via a hook on ``model.norm`` and (b) the
    hidden state the classifier head actually receives, then asserting the head's
    input equals the post-norm hidden at the marker -- and differs from
    ``marker - 1``, the position this replaced.

    The marker is placed INTERIOR (not last) so that "at the marker" is
    distinguishable from "at the sequence's last row": a marker-last layout would
    make the two coincide and the assertion would not pin the read point.
    """
    config = _classifier_config(["lora", "classifier"], label_token_ids=(100, 200))
    model = GraniteSwitchForCausalLM(config)
    _set_adapter_token_ids(model, config.adapter_token_ids)
    # Populate the classifier bank (slot 1 = adapter index 2) so the head is a real,
    # non-zero map (a zero bank would make every position's verdict identical).
    with torch.no_grad():
        model.model.classifier_head.weight[1].normal_(0.0, 0.3)
    model.eval()

    # Marker (251) INTERIOR, with content after it, so the read point is distinct
    # from both marker - 1 and the sequence's last row.
    input_ids = torch.tensor([[10, 20, 30, 40, 60, 251, 70, 80]])
    marker_pos = 5
    prev_pos = marker_pos - 1  # 4, the old read point
    last_pos = input_ids.shape[1] - 1  # 7, the sequence's final row

    captured = {}

    # Capture the final post-norm hidden states (exactly what the classifier reads).
    def norm_hook(_module, _inp, output):
        captured["normed"] = output.detach().clone()

    h = model.model.norm.register_forward_hook(norm_hook)

    # Spy on the head input so we see which hidden state it was handed.
    real_head = model.model.classifier_head
    orig_forward = real_head.forward

    def spy_forward(x, classifier_indices):
        captured["head_x"] = x.detach().clone()
        captured["head_slot"] = classifier_indices.detach().clone()
        return orig_forward(x, classifier_indices)

    real_head.forward = spy_forward
    try:
        with torch.no_grad():
            model(input_ids=input_ids)
    finally:
        real_head.forward = orig_forward
        h.remove()

    normed = captured["normed"][0]  # [seq_len, hidden]
    head_x = captured["head_x"]  # [batch, hidden] (gathered read-point state)
    assert head_x.shape == (1, config.hidden_size)

    # The head was handed the post-norm hidden AT the marker.
    assert torch.allclose(head_x[0], normed[marker_pos], atol=1e-5), (
        "classifier head did not read the marker's own hidden state"
    )
    # Not its predecessor (the read point this replaced) ...
    assert not torch.allclose(head_x[0], normed[prev_pos], atol=1e-5), (
        "classifier head read marker - 1, not the marker"
    )
    # ... and not the sequence's last row, which the marker is deliberately not.
    assert not torch.allclose(head_x[0], normed[last_pos], atol=1e-5), (
        "classifier head read the sequence's last row, not the marker"
    )
    assert marker_pos not in (prev_pos, last_pos)

    # The slot id is read at the marker, which is also the read point.
    assert int(captured["head_slot"][0]) == 2, (
        "classifier slot id must be read at the marker (adapter index 2)"
    )


def test_two_classifier_slots_in_one_request_rejected():
    """A request may name only one classifier slot.

    The verdict exit rewrites a single logit row, so two slots cannot both report.
    ``classifier_indices`` is nonzero from either marker onward, so silently
    taking one would compute the verdict from whichever position won the race.
    """
    config = _classifier_config(
        ["classifier", "classifier"], label_token_ids=(100, 200)
    )
    model = GraniteSwitchForCausalLM(config)
    _set_adapter_token_ids(model, config.adapter_token_ids)
    model.eval()

    first_tok, last_tok = config.adapter_token_ids
    with pytest.raises(RuntimeError, match="only one classifier slot"):
        with torch.no_grad():
            model(input_ids=torch.tensor([[first_tok, 10, 20, 30, last_tok]]))


def test_repeated_marker_for_one_slot_reads_the_last():
    """The same slot's marker may repeat; the last occurrence is the read point.

    A multi-turn prompt carries markers from earlier turns, so a repeat is
    legitimate and must not be rejected -- only the current turn's marker decides
    where the verdict is read.
    """
    config = _classifier_config(["lora", "classifier"], label_token_ids=(100, 200))
    model = GraniteSwitchForCausalLM(config)
    _set_adapter_token_ids(model, config.adapter_token_ids)
    with torch.no_grad():
        model.model.classifier_head.weight[1].normal_(0.0, 0.3)
    model.eval()

    cls_tok = config.adapter_token_ids[1]
    input_ids = torch.tensor([[cls_tok, 10, 20, 30, cls_tok]])
    marker_pos = 4

    captured = {}
    real_head = model.model.classifier_head
    orig_forward = real_head.forward

    def spy(x, classifier_indices):
        captured["x"] = x.detach().clone()
        return orig_forward(x, classifier_indices)

    real_head.forward = spy
    hook = model.model.norm.register_forward_hook(
        lambda _m, _i, out: captured.__setitem__("normed", out.detach().clone())
    )
    try:
        with torch.no_grad():
            model(input_ids=input_ids)
    finally:
        real_head.forward = orig_forward
        hook.remove()

    normed = captured["normed"][0]
    # Read at the LAST marker's own position, not the first marker's (position 0).
    assert torch.allclose(captured["x"][0], normed[marker_pos], atol=1e-5), (
        "verdict was not read at the last marker"
    )
    assert not torch.allclose(captured["x"][0], normed[0], atol=1e-5), (
        "verdict was read at the first marker, not the last"
    )
