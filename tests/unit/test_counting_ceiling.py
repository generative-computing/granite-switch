# SPDX-License-Identifier: Apache-2.0
"""MAX_RETAINED_CONTROL_TOKENS must match the arithmetic it claims to describe.

``Conversation`` refuses a request carrying more than
``MAX_RETAINED_CONTROL_TOKENS`` control tokens. The existing policy test patches
that bound to 1 to prove the guard fires -- which says nothing about whether 188
is the right number. This pins the number itself.

The claim. The coded switch recovers a control token's write address from a
``1/(1+n)`` attention signal. The signal is produced in the model dtype, and
``recover_count_from_signal`` upcasts to fp32 only to invert it -- so in bf16 the
information is already gone above a certain n, addresses alias, and every token
past that point retrieves whatever expert was written at the wrong address. The
codebook advertises 2048 addresses; bf16 does not deliver them.

Both directions are asserted. Too low a bound rejects conversations that would
work; too high a bound admits ones that mis-route silently. If someone makes the
counting head fp32, this fails with "the bound is now too strict" -- which is the
intended signal, and means the constant has to follow the improvement.

CPU-only, no model, milliseconds.
"""

import pytest
import torch

from granite_switch.conversation import MAX_RETAINED_CONTROL_TOKENS
from granite_switch.hf.switch.codes import (
    KerdockDGCodeGenerator,
    recover_count_from_signal,
)

# Above the bound, how much headroom before the constant counts as needlessly
# strict. Every unit below the true boundary is a conversation refused for
# nothing, so the gap is worth bounding -- loosely, since the exact boundary is a
# property of IEEE 754 rounding rather than of anything in this repo.
STRICTNESS_SLACK = 16


def _capacity():
    """The codebook's advertised address count (Kerdock m=6 -> 2048)."""
    return KerdockDGCodeGenerator(m=6, verbose=False).capacity


def _first_misrecovery(dtype, capacity):
    """Smallest n whose 1/(1+n) signal does not invert back to n in ``dtype``."""
    for n in range(capacity):
        signal = torch.tensor(1.0 / (1.0 + n), dtype=dtype)
        recovered = int(recover_count_from_signal(signal, capacity=capacity))
        if recovered != min(n, capacity - 1):
            return n
    return None


class TestBoundMatchesTheArithmetic:
    def test_bf16_recovers_every_address_up_to_the_bound(self):
        """The bound must be SAFE: no aliasing at or below it."""
        capacity = _capacity()
        first_bad = _first_misrecovery(torch.bfloat16, capacity)
        assert first_bad is not None, (
            "bf16 recovered every address up to the codebook capacity, so the "
            "ceiling this constant exists for does not appear to be real. Either "
            "the counting dtype changed or the inversion did; re-derive the bound."
        )
        assert first_bad > MAX_RETAINED_CONTROL_TOKENS, (
            f"bf16 first mis-recovers at n={first_bad}, which is at or below "
            f"MAX_RETAINED_CONTROL_TOKENS={MAX_RETAINED_CONTROL_TOKENS}. The guard "
            "would admit requests whose write addresses alias, so routing would be "
            "wrong with no error -- the failure the guard exists to prevent."
        )

    def test_the_bound_is_not_needlessly_strict(self):
        """The bound must not be far BELOW the true boundary.

        Every unit of slack is a conversation refused for nothing. This is the
        assertion that fails if the counting head is ever made fp32 and the
        constant is not raised to follow -- deliberately, because a silent
        never-taken guard is how a stale constant survives.
        """
        capacity = _capacity()
        first_bad = _first_misrecovery(torch.bfloat16, capacity)
        assert first_bad <= MAX_RETAINED_CONTROL_TOKENS + STRICTNESS_SLACK, (
            f"bf16 holds to n={first_bad} but the bound is "
            f"{MAX_RETAINED_CONTROL_TOKENS}, refusing "
            f"{first_bad - MAX_RETAINED_CONTROL_TOKENS} usable addresses. If the "
            "counting signal's dtype improved, raise MAX_RETAINED_CONTROL_TOKENS "
            "to match rather than leaving the guard stricter than the hardware."
        )

    def test_fp32_is_not_the_binding_constraint(self):
        """Locates the ceiling in the dtype, not in the inversion or the codes.

        If this ever fails, the problem is in recover_count_from_signal or the
        codebook rather than in precision, and the bound is the wrong remedy.
        """
        capacity = _capacity()
        assert _first_misrecovery(torch.float32, capacity) is None, (
            "fp32 also mis-recovers, so the ceiling is not merely a bf16 precision "
            "limit; the inversion itself is losing addresses"
        )

    def test_the_advertised_capacity_is_not_the_usable_one(self):
        """Report the gap, so nobody sizes a design against 2048 addresses."""
        capacity = _capacity()
        first_bad = _first_misrecovery(torch.bfloat16, capacity)
        print(
            f"\n  codebook capacity   {capacity}"
            f"\n  usable in bf16      {first_bad}"
            f"\n  guard admits        {MAX_RETAINED_CONTROL_TOKENS}"
        )
        assert first_bad < capacity, (
            f"bf16 delivers the full {capacity}-address capacity, so "
            "MAX_RETAINED_CONTROL_TOKENS is guarding against nothing and could be "
            "raised to the capacity"
        )


class TestGuardUsesTheBound:
    """The constant has to be the thing the guard actually reads."""

    @pytest.mark.parametrize("count", [1, 5, 200])
    def test_budget_check_counts_control_tokens_not_positions(self, count):
        """A long prompt with few control tokens must never trip the guard.

        The ceiling is on control tokens, not sequence length; conflating them
        would refuse ordinary long conversations.
        """
        from granite_switch import Conversation, KVHistoryPolicy
        from tests.shared.conversation_stubs import StubConfig, make_stub_tokenizer

        tok = make_stub_tokenizer([("unc", "alora", "<certainty>")])
        ctl = tok.token_id("<|unc|>")
        conv = Conversation(
            tok,
            policy=KVHistoryPolicy.PRESERVE_MIXED_HISTORY,
            config=StubConfig([ctl]),
        )
        ids = [999] * 5000 + [ctl] * count

        if count > MAX_RETAINED_CONTROL_TOKENS:
            with pytest.raises(RuntimeError, match="control tokens"):
                conv._assert_control_budget(ids)
        else:
            conv._assert_control_budget(ids)  # 5000 text tokens are irrelevant

    def test_reprefill_policy_never_accumulates_control_tokens(self):
        """RE_PREFILL drops history's control tokens, so the ceiling is
        PRESERVE-only: many turns never approach the bound.

        The guard exists for PRESERVE, which keeps every turn's control token.
        RE_PREFILL demotes history to base, so each request carries at most the
        current turn's one control token no matter how long the conversation.
        """
        from granite_switch import Conversation, KVHistoryPolicy
        from tests.shared.conversation_stubs import StubConfig, make_stub_tokenizer

        tok = make_stub_tokenizer([("unc", "alora", "<certainty>")])
        ctl = tok.token_id("<|unc|>")
        conv = Conversation(
            tok, policy=KVHistoryPolicy.RE_PREFILL, config=StubConfig([ctl])
        )
        for i in range(50):
            conv.user(f"Turn {i}: answerable? <certainty>")
            ids = list(conv.build_prompt(adapter="unc"))  # no raise
            assert sum(1 for t in ids if t == ctl) <= 1, (
                f"turn {i} accumulated more than one control token under RE_PREFILL"
            )
            conv.record_answer("Answer.", adapter="unc")
