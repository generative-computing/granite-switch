# SPDX-License-Identifier: Apache-2.0
"""The two KV-history policies must produce genuinely different prompts.

The discriminating property is about token ids, not about the model, so these run
on CPU in milliseconds against the real Granite chat template (a stub supplies
only encode/decode -- see ``tests/shared/conversation_stubs.py``).

Every assertion here has a twin: whenever one policy is asserted to do something,
the other is asserted NOT to. A bug that made both behave identically would
otherwise pass half the file.
"""

import logging

import pytest

from granite_switch import Conversation, KVHistoryPolicy
from granite_switch import conversation as conversation_module
from tests.shared.conversation_stubs import StubConfig, make_stub_tokenizer

A_NAME, B_NAME = "uncertainty", "requirement_check"
# Both aLoRA: an aLoRA control token lands inside the turn being generated, which
# is what makes it appear in the delta. LoRA is covered separately below.
ADAPTERS = [
    (A_NAME, "alora", "<certainty>"),
    (B_NAME, "alora", "<|start_of_role|>assistant<|end_of_role|>"),
]

Q1 = "Is this answerable from the context? <certainty>"
ANSWER_1 = "Yes, with high confidence."
Q2 = "Now summarize it."


@pytest.fixture
def tok():
    return make_stub_tokenizer(ADAPTERS)


@pytest.fixture
def config(tok):
    return StubConfig([tok.token_id(f"<|{A_NAME}|>"), tok.token_id(f"<|{B_NAME}|>")])


def _two_turns(tok, config, policy):
    """Run turn 1 with adapter A and turn 2 with adapter B; return both prompts."""
    conv = Conversation(tok, policy=policy, config=config)
    conv.user(Q1)
    p1 = conv.build_prompt(adapter=A_NAME)
    conv.record_answer(ANSWER_1, adapter=A_NAME)
    conv.user(Q2)
    p2 = conv.build_prompt(adapter=B_NAME)
    return conv, p1, p2


def _run_reprefill(tok, config, n_turns):
    """RE_PREFILL over n_turns with adapter A each turn; the user message carries
    the aLoRA invocation so this turn's control token lands in the new region.

    Returns the conversation, the ids sent each turn, and the base-ids reused at
    build time each turn (None where nothing was reusable yet).
    """
    conv = Conversation(tok, policy=KVHistoryPolicy.RE_PREFILL, config=config)
    sent, base_at_build = [], []
    for i in range(n_turns):
        conv.user(f"Turn {i}: is this answerable? <certainty>")
        base = conv._re_base_ids
        base_at_build.append(None if base is None else list(base))
        ids = list(conv.build_prompt(adapter=A_NAME))
        sent.append(ids)
        conv.record_answer("Answer.", adapter=A_NAME)
    return conv, sent, base_at_build


class TestReprefillIdReuse:
    """RE_PREFILL reuses base-demoted ids across turns (lag by one turn)."""

    def test_output_equals_a_from_scratch_render(self, tok, config):
        """Id reuse must send EXACTLY what rendering the transcript would.

        This is the load-bearing invariant: reuse is a cache optimization, never
        a change to what the model sees. A from-scratch RE_PREFILL render is the
        reference.
        """
        conv, sent, _base = _run_reprefill(tok, config, 3)
        scratch = Conversation(tok, policy=KVHistoryPolicy.RE_PREFILL, config=config)
        for i in range(3):
            scratch.user(f"Turn {i}: is this answerable? <certainty>")
            want = list(
                scratch._encode(
                    scratch._render(scratch._messages, gen=True, adapter=A_NAME)
                )
            )
            assert sent[i] == want, f"turn {i} diverges from a from-scratch render"
            scratch.build_prompt(adapter=A_NAME)
            scratch.record_answer("Answer.", adapter=A_NAME)

    def test_sent_ids_carry_no_history_control_tokens(self, tok, config):
        """RE_PREFILL demotes history to base, so each send has at most this
        turn's one control token -- never an earlier turn's."""
        conv, sent, _base = _run_reprefill(tok, config, 4)
        for i, ids in enumerate(sent):
            n = sum(1 for t in ids if t in conv._control_ids)
            assert n <= 1, f"turn {i} carries {n} control tokens; history leaked one"

    def test_deep_prefix_is_byte_stable_turn_to_turn(self, tok, config):
        """The reused base prefix equals ids actually sent the previous turn.

        That equality is the whole point: the server's cached blocks for the
        previous turn match this turn's prefix, so they are reused rather than
        recomputed.
        """
        conv, sent, base_at_build = _run_reprefill(tok, config, 4)
        for i in (2, 3):
            base = base_at_build[i]
            assert base, f"turn {i} should have had a reusable base prefix"
            assert sent[i][: len(base)] == base, "reused base is this turn's prefix"
            assert sent[i - 1][: len(base)] == base, (
                "the reused base was byte-identically sent last turn, so the cache hits"
            )

    def test_lag_is_one_turn(self, tok, config):
        """Turn 1 has no reusable base (turn 0 was sent with its adapter); reuse
        begins at turn 2."""
        conv, sent, base_at_build = _run_reprefill(tok, config, 3)
        assert base_at_build[0] is None
        assert base_at_build[1] is None
        assert base_at_build[2], "reuse begins once a turn's base form has been sent"


class TestControlTokenSurvival:
    """Whether turn 1's control token is still in turn 2's prompt."""

    def test_preserve_keeps_the_earlier_control_token(self, tok, config):
        _conv, _p1, p2 = _two_turns(tok, config, KVHistoryPolicy.PRESERVE_MIXED_HISTORY)
        assert tok.token_id(f"<|{A_NAME}|>") in p2, (
            "PRESERVE_MIXED_HISTORY must carry turn 1's control token into turn 2; "
            "without it the earlier region routes to base and the policy is a no-op"
        )
        assert tok.token_id(f"<|{B_NAME}|>") in p2, "turn 2's own adapter is missing"

    def test_re_prefill_drops_the_earlier_control_token(self, tok, config):
        """The twin: the default must NOT keep it, or the policies are the same."""
        _conv, _p1, p2 = _two_turns(tok, config, KVHistoryPolicy.RE_PREFILL)
        assert tok.token_id(f"<|{A_NAME}|>") not in p2
        assert tok.token_id(f"<|{B_NAME}|>") in p2

    def test_control_token_order_is_history_then_current(self, tok, config):
        """Latest-wins routing depends on the order, so pin it."""
        _conv, _p1, p2 = _two_turns(tok, config, KVHistoryPolicy.PRESERVE_MIXED_HISTORY)
        assert p2.index(tok.token_id(f"<|{A_NAME}|>")) < p2.index(
            tok.token_id(f"<|{B_NAME}|>")
        )


class TestPrefixProperty:
    """Whether turn 2 extends turn 1's ids, which is what makes reuse possible."""

    def test_preserve_extends_the_previous_prompt(self, tok, config):
        conv, p1, p2 = _two_turns(tok, config, KVHistoryPolicy.PRESERVE_MIXED_HISTORY)
        assert p2[: len(p1)] == p1, (
            "turn 1's ids must be a strict prefix of turn 2's, or the prefix cache "
            "cannot serve the earlier turn and PRESERVE buys nothing"
        )
        assert p2[: len(conv.sent_token_ids)] == conv.sent_token_ids
        assert len(p2) > len(conv.sent_token_ids), "the new turn added nothing"

    def test_re_prefill_does_not_extend_the_previous_prompt(self, tok, config):
        _conv, p1, p2 = _two_turns(tok, config, KVHistoryPolicy.RE_PREFILL)
        assert p2[: len(p1)] != p1

    def test_re_prefill_keeps_no_transcript(self, tok, config):
        conv, _p1, _p2 = _two_turns(tok, config, KVHistoryPolicy.RE_PREFILL)
        assert conv.sent_token_ids == []


class TestDeltaConstruction:
    def test_delta_carries_no_duplicate_system_block(self, tok, config):
        """Rendering the new turn alone would re-emit the system header.

        The subtraction exists precisely to avoid that; if it regressed, turn 2
        would contain a second system block mid-conversation.
        """
        conv, p1, p2 = _two_turns(tok, config, KVHistoryPolicy.PRESERVE_MIXED_HISTORY)
        delta_text = tok.decode(p2[len(conv.sent_token_ids) :])
        assert "system" not in delta_text, (
            f"delta re-emitted a system block: {delta_text!r}"
        )

    def test_delta_begins_on_a_special_token(self, tok, config):
        """Merges must not be able to span the join, or ids shift silently."""
        conv, _p1, p2 = _two_turns(tok, config, KVHistoryPolicy.PRESERVE_MIXED_HISTORY)
        delta_text = tok.decode(p2[len(conv.sent_token_ids) :])
        assert delta_text.startswith("<|")

    def test_join_does_not_shift_ids(self, tok, config):
        """Encoding the whole thing at once must agree with the concatenation."""
        conv, _p1, p2 = _two_turns(tok, config, KVHistoryPolicy.PRESERVE_MIXED_HISTORY)
        assert tok(tok.decode(p2))["input_ids"] == p2

    def test_join_boundary_guard_rejects_a_bare_text_join(self, tok, config):
        """Exercise the guard itself, since no real template reaches it.

        Both Granite templates happen to start a turn with a role marker, so the
        end-to-end tests above pass whether or not the check exists (verified by
        deleting it: 18/18 still green). A template that opened a turn with bare
        text would let a tokenizer merge span the join and renumber the preserved
        prefix, so the invariant is asserted directly.
        """
        conv = Conversation(
            tok, policy=KVHistoryPolicy.PRESERVE_MIXED_HISTORY, config=config
        )
        with pytest.raises(RuntimeError, match="special token"):
            conv._assert_join_boundary("Now summarize it.")
        conv._assert_join_boundary("<|start_of_role|>user<|end_of_role|>ok")  # accepted


ASSISTANT_BOUNDARY = "<|start_of_role|>assistant<|end_of_role|>"

# What PRESERVE needs to APPEND is that THIS turn's control token land in the new
# turn's region -- at or after len(prev). When it does not, PRESERVE does not
# raise: it re-prefills (a full render, history's control tokens dropped for that
# turn). ``should_work`` is now "appends and preserves" vs "falls back to a
# re-prefill". A mixed LoRA+aLoRA checkpoint can hit all these placements.
PLACEMENTS = [
    # A LoRA-placed adapter's control token is at index 0, inside the sent prefix,
    # so turn 2 cannot append -- it re-prefills.
    pytest.param(
        [("ctx", "lora", None), ("req", "alora", ASSISTANT_BOUNDARY)],
        "first turn",
        "second turn",
        "ctx",
        "ctx",
        False,
        id="lora-token-at-index-0",
    ),
    pytest.param(
        [("unc", "alora", "<certainty>"), ("req", "alora", ASSISTANT_BOUNDARY)],
        "first turn <certainty>",
        "second turn with no marker",
        "unc",
        "unc",
        False,
        id="alora-invocation-in-earlier-message",
    ),
    pytest.param(
        [("unc", "alora", "<certainty>"), ("req", "alora", ASSISTANT_BOUNDARY)],
        "first turn <certainty>",
        "second turn <certainty>",
        "unc",
        "unc",
        True,
        id="alora-invocation-in-newest-message",
    ),
    pytest.param(
        [("unc", "alora", "<certainty>"), ("req", "alora", ASSISTANT_BOUNDARY)],
        "first turn <certainty>",
        "second turn",
        "unc",
        "req",
        True,
        id="alora-assistant-boundary",
    ),
]


class TestControlTokenPlacement:
    """PRESERVE appends iff this turn's control token lands in the new turn.

    When it does not, PRESERVE re-prefills for that turn instead of raising.
    """

    @pytest.mark.parametrize(
        "adapters,turn1,turn2,adapter_a,adapter_b,should_work", PLACEMENTS
    )
    def test_placement_decides_append_vs_reprefill(
        self, adapters, turn1, turn2, adapter_a, adapter_b, should_work
    ):
        tok = make_stub_tokenizer(adapters)
        control_ids = [tok.token_id(f"<|{n}|>") for n, _t, _i in adapters]
        conv = Conversation(
            tok,
            policy=KVHistoryPolicy.PRESERVE_MIXED_HISTORY,
            config=StubConfig(control_ids),
        )
        conv.user(turn1)
        first = conv.build_prompt(adapter=adapter_a)
        conv.record_answer("ok", adapter=adapter_a)
        conv.user(turn2)

        if should_work:
            second = conv.build_prompt(adapter=adapter_b)
            assert second[: len(first)] == first, "an appendable turn preserves"
            assert sum(1 for t in second if t in set(control_ids)) == 2
            assert conv.reprefills == 0
            return

        # The control token lands inside the already-sent prefix, so PRESERVE
        # cannot append. It re-prefills instead of raising: a valid full render,
        # counted, that no longer extends the preserved prefix.
        second = conv.build_prompt(adapter=adapter_b)
        assert second, "a re-prefill still produces a valid prompt"
        assert conv.reprefills == 1, "the unappendable turn was re-prefilled"

    def test_control_texts_recoverable_for_appendable_decision(self):
        """`_appendable` needs the tokenizer's control-token spellings.

        ``_control_texts`` swallows exceptions, so a tokenizer without
        ``convert_ids_to_tokens`` would degrade `_control_char_index` to -1 and
        call every turn appendable -- silently wrong. Assert the spellings are
        recovered so the append decision is real.
        """
        tok = make_stub_tokenizer([("unc", "alora", "<certainty>")])
        conv = Conversation(
            tok,
            policy=KVHistoryPolicy.PRESERVE_MIXED_HISTORY,
            config=StubConfig([tok.token_id("<|unc|>")]),
        )
        assert conv._control_texts() == ["<|unc|>"], (
            "control-token spellings could not be recovered, so _appendable would "
            "wrongly treat every turn as appendable"
        )


class TestTranscriptBookkeeping:
    def test_default_policy_is_re_prefill(self, tok, config):
        """Existing callers must be unaffected by this class existing."""
        assert Conversation(tok, config=config).policy is KVHistoryPolicy.RE_PREFILL

    def test_discarded_prompt_does_not_advance_the_transcript(self, tok, config):
        """A judge/guardian call whose answer is thrown away must not pollute history.

        The transcript advances only on record_answer, so an uncommitted turn is
        covered by the NEXT delta rather than being lost or duplicated.
        """
        conv = Conversation(
            tok, policy=KVHistoryPolicy.PRESERVE_MIXED_HISTORY, config=config
        )
        conv.user(Q1)
        p1 = conv.build_prompt(adapter=A_NAME)
        conv.record_answer(ANSWER_1, adapter=A_NAME)
        sent_after_turn1 = conv.sent_token_ids

        conv.user("Screen this for harm.")
        conv.build_prompt(adapter=B_NAME)  # answer deliberately discarded
        assert conv.sent_token_ids == sent_after_turn1, (
            "a discarded call advanced history"
        )

        conv.user(Q2)
        p3 = conv.build_prompt(adapter=B_NAME)
        assert p3[: len(p1)] == p1, "the prefix broke after a discarded call"
        # Both uncommitted user turns are covered exactly once.
        assert tok.decode(p3).count("Screen this for harm.") == 1
        assert tok.decode(p3).count("Now summarize it.") == 1

    def test_record_answer_accepts_ids(self, tok, config):
        """Ids avoid a detokenize/retokenize round trip, so they must be accepted."""
        conv = Conversation(
            tok, policy=KVHistoryPolicy.PRESERVE_MIXED_HISTORY, config=config
        )
        conv.user(Q1)
        conv.build_prompt(adapter=A_NAME)
        answer_ids = tok(ANSWER_1)["input_ids"]
        conv.record_answer(answer_ids, adapter=A_NAME)
        assert conv.messages[-1]["content"] == ANSWER_1
        assert conv.sent_token_ids[-len(answer_ids) - 2 : -2] == answer_ids

    def test_record_answer_without_build_prompt_raises(self, tok, config):
        conv = Conversation(
            tok, policy=KVHistoryPolicy.PRESERVE_MIXED_HISTORY, config=config
        )
        conv.user(Q1)
        with pytest.raises(RuntimeError, match="without a preceding build_prompt"):
            conv.record_answer(ANSWER_1, adapter=A_NAME)

    def test_messages_hold_no_control_tokens(self, tok, config):
        """The text record must stay markup-free -- that is why RE_PREFILL loses it."""
        conv, _p1, _p2 = _two_turns(tok, config, KVHistoryPolicy.PRESERVE_MIXED_HISTORY)
        assert all(f"<|{A_NAME}|>" not in m["content"] for m in conv.messages)


class TestLongerConversations:
    """Two turns is the minimum interesting case, not the general one."""

    def test_three_turns_stay_transitively_prefixed(self, tok, config):
        """Each turn must extend the previous, or reuse breaks at turn 3.

        Two-turn tests cannot see a transitivity bug: the transcript is rebuilt
        every turn, so an off-by-one in _sent_messages or the terminator would
        first show up on the third.
        """
        conv = Conversation(
            tok, policy=KVHistoryPolicy.PRESERVE_MIXED_HISTORY, config=config
        )
        prompts = []
        for i, (question, adapter) in enumerate(
            [(Q1, A_NAME), (Q2, B_NAME), ("And once more.", B_NAME)], start=1
        ):
            conv.user(question)
            prompts.append(conv.build_prompt(adapter=adapter))
            conv.record_answer(f"answer {i}", adapter=adapter)

        assert prompts[1][: len(prompts[0])] == prompts[0]
        assert prompts[2][: len(prompts[1])] == prompts[1], (
            "turn 3 does not extend turn 2, so the cache cannot serve turn 2's "
            "blocks and the policy silently stops paying off as a chat grows"
        )
        control_id = tok.token_id(f"<|{A_NAME}|>")
        assert control_id in prompts[2], "turn 1's adapter was lost by turn 3"

    def test_three_turns_accumulate_one_control_token_each(self, tok, config):
        conv = Conversation(
            tok, policy=KVHistoryPolicy.PRESERVE_MIXED_HISTORY, config=config
        )
        ids = set(config.adapter_token_ids)
        counts = []
        for i, (question, adapter) in enumerate(
            [(Q1, A_NAME), (Q2, B_NAME), ("And once more.", B_NAME)], start=1
        ):
            conv.user(question)
            prompt = conv.build_prompt(adapter=adapter)
            counts.append(sum(1 for t in prompt if t in ids))
            conv.record_answer(f"answer {i}", adapter=adapter)
        assert counts == [1, 2, 3], f"expected one per turn, got {counts}"


class TestTemplateKwargs:
    def test_documents_pass_through_and_keep_the_prefix(self, tok, config):
        """RAG adapters need documents=; PRESERVE must survive them.

        The delta is a subtraction of two renders, and only one of them carries
        the adapter. If documents reached one render and not the other, the
        subtraction would cut in the wrong place.
        """
        docs = [{"doc_id": "0", "text": "a retrieved passage about glide paths"}]
        conv = Conversation(
            tok, policy=KVHistoryPolicy.PRESERVE_MIXED_HISTORY, config=config
        )
        conv.user(Q1)
        first = conv.build_prompt(adapter=A_NAME, documents=docs)
        conv.record_answer(ANSWER_1, adapter=A_NAME)
        conv.user(Q2)
        second = conv.build_prompt(adapter=B_NAME, documents=docs)

        assert second[: len(first)] == first
        assert tok.token_id(f"<|{A_NAME}|>") in second

    def test_system_turn_is_carried(self, tok, config):
        conv = Conversation(
            tok, policy=KVHistoryPolicy.PRESERVE_MIXED_HISTORY, config=config
        )
        conv.system("You are a careful assistant.")
        conv.user(Q1)
        prompt = conv.build_prompt(adapter=A_NAME)
        assert conv.messages[0] == {
            "role": "system",
            "content": "You are a careful assistant.",
        }
        assert "careful assistant" in tok.decode(prompt)


class TestGuards:
    def test_preserve_requires_config(self, tok):
        """Omitting config disables both guards, so it must be refused.

        This is the easiest way to construct the class and it silently turned off
        the checks that exist because their failure modes are silent.
        """
        with pytest.raises(ValueError, match="requires config"):
            Conversation(tok, policy=KVHistoryPolicy.PRESERVE_MIXED_HISTORY)

    def test_re_prefill_does_not_require_config(self, tok):
        """The twin: RE_PREFILL has nothing to guard, so it must stay usable."""
        conv = Conversation(tok, policy=KVHistoryPolicy.RE_PREFILL)
        conv.user(Q1)
        assert conv.build_prompt(adapter=A_NAME)

    def test_re_prefill_works_with_config(self, tok):
        """The twin: the config guard must not over-reach and block the default policy."""
        cfg = StubConfig([tok.token_id(f"<|{A_NAME}|>")])
        Conversation(tok, policy=KVHistoryPolicy.RE_PREFILL, config=cfg)


class TestAutomaticReprefill:
    """Past the ceiling, PRESERVE re-prefills instead of growing the count.

    Twinned like the rest of this file: each assertion that the fallback fires is
    paired with one that it does NOT fire below the ceiling, so a bug that
    re-prefilled every turn would fail here rather than pass silently. The pair also
    pins the comparison as ``>`` rather than ``>=``: a request carrying exactly
    MAX_RETAINED_CONTROL_TOKENS addresses the last write address bf16 inverts
    exactly, so it must be sent as it is, not re-prefilled.
    """

    def _three_turns(self, tok, config, ceiling, monkeypatch):
        """Turns A, B, A. Counts are 1, 2, 3, so ceiling=2 fires on the third only."""
        monkeypatch.setattr(conversation_module, "MAX_RETAINED_CONTROL_TOKENS", ceiling)
        conv = Conversation(
            tok, policy=KVHistoryPolicy.PRESERVE_MIXED_HISTORY, config=config
        )
        conv.user(Q1)
        conv.build_prompt(adapter=A_NAME)
        conv.record_answer(ANSWER_1, adapter=A_NAME)
        conv.user(Q2)
        p2 = conv.build_prompt(adapter=B_NAME)
        conv.record_answer("Summarized.", adapter=B_NAME)
        # A_NAME is an aLoRA whose invocation text is "<certainty>"; under PRESERVE
        # its control token has to land in the NEWEST user message, or _delta refuses
        # before the control budget is ever consulted.
        conv.user("And once more. <certainty>")
        p3 = conv.build_prompt(adapter=A_NAME)
        return conv, p2, p3

    def test_reaching_the_trigger_drops_the_history_control_tokens(
        self, tok, config, monkeypatch
    ):
        conv, p2, p3 = self._three_turns(tok, config, 2, monkeypatch)
        assert tok.token_id(f"<|{A_NAME}|>") in p2, (
            "turn 2 sits AT the ceiling, which is legal, so PRESERVE must still carry "
            "turn 1's token -- re-prefilling here would discard a working prefix"
        )
        assert conv.reprefills == 1, (
            f"turn 3 reaches 3 control tokens and must re-prefill once; "
            f"got reprefills={conv.reprefills}"
        )
        assert tok.token_id(f"<|{B_NAME}|>") not in p3, (
            "the re-prefilled prompt must drop history's control tokens -- that is "
            "the only thing that brings the count back down"
        )
        assert tok.token_id(f"<|{A_NAME}|>") in p3, (
            "the CURRENT turn's control token must survive the re-prefill, or this "
            "turn routes to base and the caller's adapter request is lost"
        )

    def test_at_or_below_the_ceiling_nothing_is_re_prefilled(
        self, tok, config, monkeypatch
    ):
        """The twin: a high ceiling must leave PRESERVE exactly as it was."""
        conv, p2, p3 = self._three_turns(tok, config, 99, monkeypatch)
        assert conv.reprefills == 0
        assert tok.token_id(f"<|{A_NAME}|>") in p2
        assert tok.token_id(f"<|{B_NAME}|>") in p3, (
            "without a re-prefill turn 3 must still carry turn 2's control token"
        )

    def test_the_re_prefilled_prompt_equals_the_re_prefill_render(
        self, tok, config, monkeypatch
    ):
        """Re-prefilling must mean the same ids RE_PREFILL would have sent."""
        monkeypatch.setattr(conversation_module, "MAX_RETAINED_CONTROL_TOKENS", 1)
        _conv_p, _p1, p2 = _two_turns(
            tok, config, KVHistoryPolicy.PRESERVE_MIXED_HISTORY
        )
        _conv_r, _r1, r2 = _two_turns(tok, config, KVHistoryPolicy.RE_PREFILL)
        assert list(p2) == list(r2), (
            "a re-prefilled PRESERVE prompt must be byte-for-byte the RE_PREFILL "
            "render; anything else is a third, untested prompt shape"
        )

    def test_the_re_prefill_is_reported(self, tok, config, monkeypatch, caplog):
        """Silent history loss is the failure mode this whole module guards against."""
        monkeypatch.setattr(conversation_module, "MAX_RETAINED_CONTROL_TOKENS", 1)
        with caplog.at_level(logging.WARNING, logger="granite_switch.conversation"):
            _two_turns(tok, config, KVHistoryPolicy.PRESERVE_MIXED_HISTORY)
        assert any("re-prefilled" in r.getMessage() for r in caplog.records), (
            f"expected a WARNING naming the re-prefill; got "
            f"{[r.getMessage() for r in caplog.records]}"
        )

    def test_preserve_resumes_from_the_new_baseline(self, tok, config, monkeypatch):
        """After the reset, later turns must append to it rather than re-render."""
        conv, _p2, _p3 = self._three_turns(tok, config, 2, monkeypatch)
        conv.record_answer("Once more, then.", adapter=A_NAME)
        conv.user("Last one.")
        sent = list(conv.sent_token_ids)
        p4 = conv.build_prompt(adapter=B_NAME)
        assert conv.reprefills == 1, "turn 4 is back under the trigger"
        assert list(p4[: len(sent)]) == sent, (
            "turn 4 must extend the post-re-prefill ids verbatim; if it does not, "
            "the cache the reset paid for is thrown away every turn after it"
        )

    def test_a_full_render_that_is_still_over_budget_raises(
        self, tok, config, monkeypatch
    ):
        """The one case re-prefilling cannot fix, so it must still be loud.

        Recording an answer as TEXT containing a literal control-token spelling puts
        it in ``_messages``, where the re-render picks it up again -- unlike a control
        token the model emitted as an id, which ``record_answer`` strips via
        ``skip_special_tokens``. So the reset does not reduce this one.
        """
        monkeypatch.setattr(conversation_module, "MAX_RETAINED_CONTROL_TOKENS", 1)
        conv = Conversation(
            tok, policy=KVHistoryPolicy.PRESERVE_MIXED_HISTORY, config=config
        )
        conv.user(Q1)
        conv.build_prompt(adapter=A_NAME)
        conv.record_answer(f"Maybe <|{A_NAME}|> maybe not.", adapter=A_NAME)
        conv.user(Q2)
        with pytest.raises(RuntimeError, match="already a full render"):
            conv.build_prompt(adapter=B_NAME)
