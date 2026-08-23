# SPDX-License-Identifier: Apache-2.0
"""The KV-history policies must route the same way on the vLLM switch as on HF.

``tests/hf/test_conversation_routing.py`` proves the policies produce different
routing through the HF ``MultiSwitch``. ``Conversation`` itself is
backend-agnostic -- it only touches the tokenizer -- but the switch that consumes
its ids is a different implementation per backend, with real ``vllm.Attention``
kernels and paged KV instead of a DynamicCache. Agreement is therefore something
to verify, not assume: a divergence here would mean the same prompt is routed
differently in production than in the HF tests.

Reuses the long-lived GPU worker from ``test_multi_switch.py`` rather than
standing up a second one, so all CUDA work stays in that subprocess and the
parent pytest process never creates a context.

Requires a CUDA GPU + vLLM; skips otherwise.

STATUS: green on 1x A100 (vLLM 0.19.x) -- 4 passed, first run, no fixes needed.
``test_vllm_routing_matches_hf_routing`` confirmed the two backends produce
identical per-position index vectors for both policies, which was the open
question: Conversation is backend-agnostic, but the switches are separate
implementations with different attention kernels and cache mechanics.
"""

import importlib.util

import pytest

_VLLM_AVAILABLE = importlib.util.find_spec("vllm") is not None

pytestmark = pytest.mark.skipif(
    not _VLLM_AVAILABLE,
    reason="requires vLLM installed (GPU checked by worker)",
)

from granite_switch import Conversation, KVHistoryPolicy
from tests.shared.conversation_stubs import (
    StubConfig,
    make_stub_tokenizer,
)
from tests.vllm.test_multi_switch import _send

A_NAME, B_NAME = "uncertainty", "requirement_check"
ADAPTERS = [
    (A_NAME, "alora", "<certainty>"),
    (B_NAME, "alora", "<|start_of_role|>assistant<|end_of_role|>"),
]
Q1 = "Is this answerable from the context? <certainty>"
ANSWER_1 = "Yes, with high confidence."
Q2 = "Now summarize it."


def _prompts(policy):
    """Build turn 1 and turn 2 prompts under ``policy``.

    The stub keeps every id below the worker's ``vocab_size=2000``; an id at or
    above it would index past the switch's token-exchange LUT.
    """
    tok = make_stub_tokenizer(ADAPTERS)
    a_id, b_id = tok.token_id(f"<|{A_NAME}|>"), tok.token_id(f"<|{B_NAME}|>")
    conv = Conversation(tok, policy=policy, config=StubConfig([a_id, b_id]))
    conv.user(Q1)
    p1 = conv.build_prompt(adapter=A_NAME)
    conv.record_answer(ANSWER_1, adapter=A_NAME)
    conv.user(Q2)
    return conv, p1, conv.build_prompt(adapter=B_NAME), a_id, b_id


def _route(ids, control_ids):
    return _send("multi", {"seq": list(ids), "adapter_token_ids": list(control_ids)})


class TestPolicyRoutingOnVLLM:
    def test_preserve_keeps_the_history_on_its_original_adapter(self):
        _conv, _p1, p2, a_id, b_id = _prompts(KVHistoryPolicy.PRESERVE_MIXED_HISTORY)
        indices = _route(p2, [a_id, b_id])
        a_pos, b_pos = p2.index(a_id), p2.index(b_id)

        assert set(indices[a_pos:b_pos]) == {1}, (
            f"positions {a_pos}..{b_pos - 1} should route to adapter 1 on vLLM too, "
            f"got {sorted(set(indices[a_pos:b_pos]))}"
        )
        assert set(indices[b_pos:]) == {2}
        assert set(indices[:a_pos]) == {0}

    def test_re_prefill_reinterprets_the_history_as_base(self):
        _conv, _p1, p2, a_id, b_id = _prompts(KVHistoryPolicy.RE_PREFILL)
        indices = _route(p2, [a_id, b_id])

        assert a_id not in p2
        b_pos = p2.index(b_id)
        assert set(indices[:b_pos]) == {0}
        assert set(indices[b_pos:]) == {2}

    def test_the_two_policies_do_not_agree(self):
        """Anti-vacuity: identical vectors would make both tests above meaningless."""
        _c1, _p1, preserve, a_id, b_id = _prompts(
            KVHistoryPolicy.PRESERVE_MIXED_HISTORY
        )
        _c2, _p2, reprefill, _a, _b = _prompts(KVHistoryPolicy.RE_PREFILL)
        assert _route(preserve, [a_id, b_id]) != _route(reprefill, [a_id, b_id])


class TestBackendAgreement:
    def test_vllm_routing_matches_hf_routing(self):
        """The same prompt must route identically on both backends.

        This is the assertion the file exists for. The HF file checks the shape
        against intent; this checks the two implementations against each other,
        which is what catches a divergence that both files' intent-checks would
        otherwise accept.
        """
        import torch

        from granite_switch.hf.switch import create_switch
        from tests.hf.test_multi_switch import _MockSwitchConfig

        for policy in (
            KVHistoryPolicy.RE_PREFILL,
            KVHistoryPolicy.PRESERVE_MIXED_HISTORY,
        ):
            _conv, _p1, p2, a_id, b_id = _prompts(policy)
            vllm_indices = _route(p2, [a_id, b_id])

            cfg = _MockSwitchConfig(
                "multi", [a_id, b_id], [1, 2], backend="sdpa", num_adapters=2
            )
            cfg.vocab_size = 2000
            hf_switch = create_switch(cfg, layer_idx=0)
            out, _ = hf_switch.forward(
                input_ids=torch.tensor([p2]),
                adapter_token_ids=torch.tensor([a_id, b_id]),
                cache_position=torch.arange(len(p2)),
            )
            hf_indices = [int(x) for x in out[0]]

            assert vllm_indices == hf_indices, (
                f"backends disagree under {policy.value}: vLLM {vllm_indices} vs "
                f"HF {hf_indices}"
            )
