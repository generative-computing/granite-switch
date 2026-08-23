# SPDX-License-Identifier: Apache-2.0
"""The policies must route differently, not merely produce different prompts.

``tests/unit/test_conversation_policy.py`` asserts the token-level properties.
This file closes the loop by feeding each policy's prompt to the REAL
``MultiSwitch`` and checking the per-position adapter indices, because a
prompt difference that did not change routing would buy nothing.

CPU-only: the switch runs on a synthetic geometry, so no checkpoint is needed.
"""

import pytest
import torch

from granite_switch import Conversation, KVHistoryPolicy
from granite_switch.hf.switch import create_switch
from tests.hf.test_multi_switch import _MockSwitchConfig
from tests.shared.conversation_stubs import StubConfig, make_stub_tokenizer

A_NAME, B_NAME = "uncertainty", "requirement_check"
ADAPTERS = [
    (A_NAME, "alora", "<certainty>"),
    (B_NAME, "alora", "<|start_of_role|>assistant<|end_of_role|>"),
]
Q1 = "Is this answerable from the context? <certainty>"
ANSWER_1 = "Yes, with high confidence."
Q2 = "Now summarize it."


@pytest.fixture
def pieces():
    """(tokenizer, config, switch) sharing one control-token id assignment."""
    tok = make_stub_tokenizer(ADAPTERS)
    a_id, b_id = tok.token_id(f"<|{A_NAME}|>"), tok.token_id(f"<|{B_NAME}|>")
    cfg = StubConfig([a_id, b_id])
    switch_cfg = _MockSwitchConfig(
        "multi", [a_id, b_id], [1, 2], backend="sdpa", num_adapters=2
    )
    switch_cfg.vocab_size = 20000
    return tok, cfg, create_switch(switch_cfg, layer_idx=0), a_id, b_id


def _route(switch, ids, control_ids):
    out, _ = switch.forward(
        input_ids=torch.tensor([ids]),
        adapter_token_ids=torch.tensor(control_ids),
        cache_position=torch.arange(len(ids)),
    )
    return [int(x) for x in out[0]]


def _prompts(tok, cfg, policy):
    conv = Conversation(tok, policy=policy, config=cfg)
    conv.user(Q1)
    p1 = conv.build_prompt(adapter=A_NAME)
    conv.record_answer(ANSWER_1, adapter=A_NAME)
    conv.user(Q2)
    return conv, p1, conv.build_prompt(adapter=B_NAME)


class TestRoutingDiffersByPolicy:
    def test_preserve_keeps_the_history_on_its_original_adapter(self, pieces):
        tok, cfg, switch, a_id, b_id = pieces
        conv, p1, p2 = _prompts(tok, cfg, KVHistoryPolicy.PRESERVE_MIXED_HISTORY)
        indices = _route(switch, p2, [a_id, b_id])

        a_pos, b_pos = p2.index(a_id), p2.index(b_id)
        # The region turn 1 produced still routes to adapter 1 (index 1).
        assert set(indices[a_pos:b_pos]) == {1}, (
            f"positions {a_pos}..{b_pos - 1} should all route to adapter 1, got "
            f"{sorted(set(indices[a_pos:b_pos]))}"
        )
        # The new turn routes to adapter 2, and the untouched opening to base.
        assert set(indices[b_pos:]) == {2}
        assert set(indices[:a_pos]) == {0}

    def test_re_prefill_reinterprets_the_history_as_base(self, pieces):
        """The twin: without the earlier token the same region must read base."""
        tok, cfg, switch, a_id, b_id = pieces
        _conv, _p1, p2 = _prompts(tok, cfg, KVHistoryPolicy.RE_PREFILL)
        indices = _route(switch, p2, [a_id, b_id])

        assert a_id not in p2
        b_pos = p2.index(b_id)
        assert set(indices[:b_pos]) == {0}, (
            "with turn 1's control token gone, every earlier position must read base"
        )
        assert set(indices[b_pos:]) == {2}

    def test_the_two_policies_do_not_agree(self, pieces):
        """Anti-vacuity: if the vectors matched, neither test above means anything."""
        tok, cfg, switch, a_id, b_id = pieces
        _c1, _p1, preserve = _prompts(tok, cfg, KVHistoryPolicy.PRESERVE_MIXED_HISTORY)
        _c2, _p2, reprefill = _prompts(tok, cfg, KVHistoryPolicy.RE_PREFILL)
        assert _route(switch, preserve, [a_id, b_id]) != _route(
            switch, reprefill, [a_id, b_id]
        )


class TestDecodeStep:
    def test_generated_tokens_stay_on_the_new_adapter(self, pieces):
        """Routing must hold through decode, not just prefill.

        After the prompt is processed the control token is no longer in the
        step's input_ids; it lives in the cache, and the switch reads it back
        from there. A decode step over a grown cache is the only way to see it.
        """
        from transformers.cache_utils import DynamicCache

        tok, cfg, switch, a_id, b_id = pieces
        _conv, _p1, p2 = _prompts(tok, cfg, KVHistoryPolicy.PRESERVE_MIXED_HISTORY)
        cache = DynamicCache()
        atok = torch.tensor([a_id, b_id])

        prefill, _ = switch.forward(
            input_ids=torch.tensor([p2]),
            adapter_token_ids=atok,
            past_key_values=cache,
            cache_position=torch.arange(len(p2)),
        )
        assert int(prefill[0, -1]) == 2

        step, _ = switch.forward(
            input_ids=torch.tensor([[1234]]),  # an ordinary generated token
            adapter_token_ids=atok,
            past_key_values=cache,
            cache_position=torch.tensor([len(p2)]),
        )
        assert int(step[0, 0]) == 2, (
            f"the decode step routed to {int(step[0, 0])}, expected 2 (carry from the "
            "control token now living only in the cache)"
        )
