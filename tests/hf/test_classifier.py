# SPDX-License-Identifier: Apache-2.0
"""Classifier-substitute path tests for GraniteSwitchForCausalLM (HF backend).

A classifier slot is an alternative to a LoRA adapter: it shares the adapter
index/control-token machinery, the switch splits classifier positions out of
the LoRA stream, and the classifier head reads the final hidden state to emit
ONE verdict per request. The verdict is read at the control token, which must be
the request's last token.

The verdict exits as a GENERATED LABEL WORD, matching the vLLM backend: the LM
head rewrites the marker's logit row so the vocab is -inf except at
``classifier_label_token_ids``, where the per-label verdict scores are placed. There is no separate float ``classifier_logits`` output field.

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

    # Marker (251) last; this test checks the label-word rewrite of the emitted row.
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
    verdict = model.model._last_classifier_verdict[2][0, :n]  # [n_labels]
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
            [10, 20, 30, 40, 251],  # classifier
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


def test_decode_step_after_verdict_is_plain_lm():
    # After the verdict token, a cached decode step carries no marker, so it runs
    # as a plain LM step: no verdict, full finite vocab row.
    config = _classifier_config(["lora", "classifier"])
    model = GraniteSwitchForCausalLM(config)
    _set_adapter_token_ids(model, config.adapter_token_ids)
    model.eval()

    input_ids = torch.tensor([[10, 20, 30, 251]])
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    assert model.model._last_classifier_verdict is not None

    with torch.no_grad():
        step = model(
            input_ids=torch.tensor([[100]]),
            past_key_values=out.past_key_values,
            use_cache=True,
        )
    assert model.model._last_classifier_verdict is None
    assert torch.isfinite(step.logits).all()


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
    assert model.model._last_classifier_verdict is None
    assert torch.isfinite(out.logits).all()
    assert out.logits.shape == (1, 6, config.vocab_size)


def test_lora_generation_on_mixed_model_decodes():
    # A model composed WITH a classifier slot also serves plain LoRA/base
    # generation. Firing the LoRA (adapter 1, token 250) — not the classifier —
    # must decode multiple tokens with no verdict and full LM logits.
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
    assert model.model._last_classifier_verdict is None
    assert torch.isfinite(out.logits).all()

    # Cached decode step of the same (LoRA) request yields finite logits.
    with torch.no_grad():
        step = model(
            input_ids=torch.tensor([[60]]),
            past_key_values=out.past_key_values,
            use_cache=True,
        )
    assert step.logits.shape == (1, 1, config.vocab_size)
    assert torch.isfinite(step.logits).all()
    assert model.model._last_classifier_verdict is None


@pytest.mark.parametrize(
    "kinds, input_ids",
    [
        # Content after the marker.
        (["lora", "classifier"], [[10, 20, 251, 30, 40]]),
        # Two different classifier slots, the second one last.
        (["classifier", "classifier"], [[250, 10, 20, 30, 251]]),
        # The same slot's marker twice.
        (["lora", "classifier"], [[251, 10, 20, 30, 251]]),
    ],
    ids=["content-after", "two-slots", "repeated"],
)
def test_marker_that_is_not_the_last_token_raises(kinds, input_ids):
    """The verdict is read at the last token, so any other marker would be
    ignored silently; it raises instead."""
    config = _classifier_config(kinds, label_token_ids=(100, 200))
    model = GraniteSwitchForCausalLM(config)
    _set_adapter_token_ids(model, config.adapter_token_ids)
    model.eval()

    with pytest.raises(RuntimeError, match="not their last token"):
        with torch.no_grad():
            model(input_ids=torch.tensor(input_ids))


def test_right_padded_row_rewrites_the_marker_column():
    """With right padding the marker is the row's last real token, not its last
    column; the verdict lands on the marker's own logit row."""
    config = _classifier_config(["lora", "classifier"], label_token_ids=(100, 200))
    model = GraniteSwitchForCausalLM(config)
    _set_adapter_token_ids(model, config.adapter_token_ids)
    model.eval()

    input_ids = torch.tensor([[10, 20, 251, 0, 0], [10, 20, 30, 40, 251]])
    attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)

    label_ids = config.classifier_label_token_ids[1]
    non_label = [t for t in range(config.vocab_size) if t not in label_ids]
    assert torch.isinf(out.logits[0, 2, non_label]).all()
    assert torch.isfinite(out.logits[0, -1, :]).all()  # padding untouched
    assert torch.isinf(out.logits[1, -1, non_label]).all()

    # Keeping only the last position drops row 0's marker row: fail loud.
    with pytest.raises(RuntimeError, match="logits_to_keep"):
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=1)
