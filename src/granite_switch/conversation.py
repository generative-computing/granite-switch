# SPDX-License-Identifier: Apache-2.0
"""Multi-turn conversations where each turn may use a different adapter.

The problem this solves. A control token is *markup*: the chat template writes it
into the rendered prompt, and it exists only in the token ids that render produced.
A conversation history stored as ``messages`` holds what was *said* -- role and
content text -- and text carries no control tokens. So re-rendering the
conversation on turn 2 silently drops turn 1's control token, the positions it
governed are reinterpreted as base, and the KV blocks computed for them can never
be reused (their token prefix no longer matches).

That is often fine, and it is what every ordinary chat API does. But when a turn's
history should stay attributed to the adapter that produced it -- and, as a direct
consequence, stay reusable from the prefix cache -- the ids have to be kept rather
than re-derived. That is the whole of ``PRESERVE_MIXED_HISTORY``.

    conv = Conversation(tokenizer, policy=KVHistoryPolicy.PRESERVE_MIXED_HISTORY,
                        config=model.config)
    conv.user("Map this report to MITRE techniques.")
    ids = conv.build_prompt(adapter="cti-technique-mapping")
    conv.record_answer(send(ids), adapter="cti-technique-mapping")

    conv.user("Now give me that as JSON.")
    # turn 1's control token is still in here:
    ids = conv.build_prompt(adapter="text-to-json")

The caller never writes a control token, a token id, or a payload. Adapter
placement stays owned by the chat template: the new turn's text is obtained by
rendering the conversation with and without it and subtracting, so this module
never hardcodes a role marker and inherits any future template change for free.

Requirements for PRESERVE_MIXED_HISTORY:

* the prompt must be sent as token ids (``/v1/completions`` with ``prompt=[ids]``,
  or ``model.generate(input_ids=...)``). ``/v1/chat/completions`` re-renders and
  re-tokenizes server-side, which silently turns this back into RE_PREFILL.
* **aLoRA adapters only**, enforced. Use ``RE_PREFILL`` for any conversation whose
  turns need LoRA adapters.

  The reason is placement. A LoRA adapter's control token is emitted at sequence
  position 0 (the template's LoRA prefix insertion, which also suppresses the role
  marker that would follow), and position 0 is inside the already-sent prefix, so
  the delta cannot be derived. An aLoRA adapter activates inside a user message or
  at the assistant boundary -- in the turn being generated -- which is what makes
  it appear in the delta.

  ``build_prompt`` refuses on TURN 1, where a LoRA adapter would otherwise render
  fine: turn 1 takes the full-render path, so there is no prefix to preserve. It
  is refused anyway because a conversation does not change adapter technology
  mid-dialogue -- a LoRA turn 1 is a LoRA turn 2, and turn 2 cannot be served
  under this policy. Succeeding once and then failing on every later turn would
  leave the caller holding a transcript that cannot continue under the policy it
  chose.

  NOT covered by that check: an aLoRA adapter whose invocation text sits in an
  EARLIER user message also rewrites history and fails, and has no index-0
  signature to test for. ``build_prompt`` raises for it on the turn it happens,
  naming the adapter and both character positions.

Not handled here: returning routing to *base* between two adapter regions. That
needs a base-reset control token, which this module never emits, so under this
policy an earlier adapter carries forward until the next control token fires.
Composing with ``--base-reset-token`` puts ``<|base_reset|>`` in the checkpoint
(``num_adapters + 1`` control tokens, base slot first) and the caller may place it
in the prompt, but nothing here does so on the caller's behalf.
"""

import logging
from enum import Enum

__all__ = [
    "MAX_RETAINED_CONTROL_TOKENS",
    "Conversation",
    "KVHistoryPolicy",
    "PromptTokenIds",
]

logger = logging.getLogger(__name__)


# The coded switch recovers a control token's write address from a 1/(1+n)
# attention signal in the model dtype. In bf16 that inverts exactly only below
# this count; past it, addresses alias and routing silently degrades. Preserving
# history is what makes n grow, so the ceiling belongs to this module.
#
# Both the re-prefill and the raise compare with ``>``, so this count is legal and
# only the next one is not: a request carrying C control tokens gives its last one
# the write address C, and C = 188 is the last address bf16 inverts exactly.
#
# 188 is one below the cliff rather than at it, because the first aliased count is
# harmless. In bf16, n=189 recovers as 190 -- but the SAME recovered address is used
# to write the codeword and to read it back (multi.py builds k_memory and q_memory
# from one ``codebook[write_addresses]``), so 189 writes at 190, its followers read
# 190, and routing is correct. The address set is still all-distinct. It is n=190
# that collides: 189 and 190 both recover as 190, two control tokens key the same
# codeword, and the memory head returns the mean of their two expert ids. So the
# functional cliff is 190 control tokens in one request, and 188 leaves exactly one
# token of headroom for a control token the model emits mid-answer -- which joins
# this same request during decode.
#
# That headroom is one token, not two. Two control tokens in a single answer reach
# 190 and collide. Turn N+1 cannot: under PRESERVE ``record_answer`` retains the raw
# answer ids, control tokens included, so the next build_prompt counts 190, trips
# this bound, and re-prefills before the request is ever sent.
#
# Lowering this number is therefore a policy knob as well as a fact, and turns
# tests/unit/test_counting_ceiling.py (which recomputes the fact) red on purpose.
MAX_RETAINED_CONTROL_TOKENS = 188


class KVHistoryPolicy(Enum):
    """How earlier turns are represented when a later turn uses another adapter.

    ``RE_PREFILL``
        Re-render the whole conversation from ``messages`` each turn. Earlier
        control tokens are dropped, so history is interpreted under base and is
        recomputed from the first adapter turn onward. This is what the plain
        ``apply_chat_template`` flow already does, and it is the default.

    ``PRESERVE_MIXED_HISTORY``
        Send the ids already sent, plus the new turn. Earlier control tokens stay
        in the stream, so each region keeps routing to the adapter that produced
        it and its cached KV stays eligible for reuse. aLoRA adapters only: a LoRA
        adapter's control token is emitted at position 0, so it cannot be the
        adapter of turn 2 or later and ``build_prompt`` refuses it on turn 1 --
        see the module docstring.

        A conversation long enough to exceed ``MAX_RETAINED_CONTROL_TOKENS`` control
        tokens in one request is re-prefilled automatically for that turn: history's
        control tokens are dropped, the transcript is re-rendered, and preserving
        resumes from the new baseline. It is logged and counted
        (:attr:`Conversation.reprefills`) because the conversation that comes out is
        no longer the one that went in -- earlier regions route to base from then on.
    """

    RE_PREFILL = "re_prefill"
    PRESERVE_MIXED_HISTORY = "preserve_mixed_history"


class PromptTokenIds(list):
    """Token ids from :meth:`Conversation.build_prompt`, tagged with their policy.

    A ``list[int]`` subclass, so it behaves as one everywhere. It exists to carry
    one fact the bare list cannot: whether these ids MUST be sent as ids. Under
    ``PRESERVE_MIXED_HISTORY`` they must -- re-rendering or re-tokenizing them
    server-side reproduces a different prefix and silently degrades the request to
    ``RE_PREFILL``, with no error and no observable symptom beyond a fallen cache
    hit rate. ``requires_token_ids`` lets a transport layer check rather than
    assume.
    """

    __slots__ = ("policy",)

    def __init__(self, ids, policy):
        super().__init__(ids)
        self.policy = policy

    @property
    def requires_token_ids(self):
        """True: every prompt this class builds must be sent as ids.

        Both policies reuse previously-sent ids as a stable prefix. Re-rendering
        or re-tokenizing them server-side reproduces a different prefix and
        silently degrades cache reuse, with no error and no symptom beyond a
        fallen hit rate. The flag stays so a transport layer can check rather
        than assume; it is now constant.
        """
        return True


class Conversation:
    """A growing conversation that knows which adapter produced which turn.

    Args:
        tokenizer: tokenizer of a composed Granite Switch checkpoint (it carries
            the adapter control tokens and the adapter-aware chat template).
        policy: see :class:`KVHistoryPolicy`. Defaults to ``RE_PREFILL``, which
            reproduces the behaviour of rendering each turn from scratch.
        config: the checkpoint's ``GraniteSwitchConfig``, or anything exposing
            ``adapter_token_ids``. Optional, but without it the control-token-budget
            guard cannot run.
    """

    def __init__(self, tokenizer, policy=KVHistoryPolicy.RE_PREFILL, config=None):
        self.tokenizer = tokenizer
        self.policy = policy
        self.config = config

        self._messages: list[dict] = []
        self._sent_ids: list[int] = []
        # How many entries of _messages are already represented in _sent_ids.
        # Not len(_messages): a prompt whose answer was never recorded (a judge
        # or guardian call the caller discarded) leaves messages uncommitted, and
        # the next delta must cover all of them.
        self._sent_messages = 0
        self._pending: list[int] | None = None
        # How many times build_prompt has re-prefilled to stay inside the counting
        # range. A silent reset would look exactly like working PRESERVE, so it is
        # counted as well as logged.
        self._reprefills = 0
        # The template kwargs the ids in _sent_ids / _pending were rendered with.
        # _delta subtracts two renders, so the "already sent" render must be
        # reproduced with the kwargs that actually produced it -- see _delta.
        self._sent_template_kwargs: dict = {}
        self._pending_template_kwargs: dict = {}
        self._turn_adapters: list[str | None] = []
        # Per recorded answer: the control-token ids the model emitted in it.
        self._generated_control_tokens: list[list[int]] = []

        self._control_ids = set(getattr(config, "adapter_token_ids", None) or [])
        self._control_text_cache: list[str] | None = None

        if policy is KVHistoryPolicy.PRESERVE_MIXED_HISTORY:
            if config is None:
                raise ValueError(
                    "PRESERVE_MIXED_HISTORY requires config=<the checkpoint's "
                    "GraniteSwitchConfig>. This policy's control-token-budget guard "
                    "reads it, so omitting it does not merely lose diagnostics -- it "
                    "disables the guard: a long conversation would slide past the "
                    "counting head's exact range and alias its write addresses, a "
                    "silent failure, which is exactly why the guard exists. RE_PREFILL "
                    "does not need config."
                )

    # ── conversation state ────────────────────────────────────────────────────
    def user(self, content, **fields):
        """Append a user turn. Extra fields are passed through to the template."""
        self._messages.append({"role": "user", "content": content, **fields})

    def system(self, content):
        """Append a system turn. Only meaningful before the first user turn."""
        self._messages.append({"role": "system", "content": content})

    @property
    def messages(self):
        """The conversation as role/content dicts (a copy; text only).

        Under ``PRESERVE_MIXED_HISTORY`` this is a **display** view, not what gets
        sent. Text carries no control tokens, so posting it to a chat endpoint
        would send a different conversation than the one this object has been
        building -- which is why the prompt must travel as ids
        (:meth:`completion_payload`).
        """
        return [dict(m) for m in self._messages]

    @property
    def sent_token_ids(self):
        """Ids the model has already seen, verbatim (empty under RE_PREFILL)."""
        return list(self._sent_ids)

    @property
    def reprefills(self):
        """How many turns were re-prefilled to stay inside the counting range.

        Non-zero means at least one turn dropped its history's control tokens, so
        earlier regions are interpreted under base from that turn on. Nothing else
        distinguishes that conversation from one that never needed it.
        """
        return self._reprefills

    @property
    def generated_control_tokens(self):
        """Control-token ids the MODEL emitted, one list per recorded answer.

        Control tokens are freely generatable -- there is no runtime suppression --
        so a model can name an adapter mid-answer and re-route the rest of its own
        generation. That is deliberate, and this does not prevent it; it makes it
        visible, because nothing else does: the returned text has already dropped
        the token (``skip_special_tokens`` defaults to true, in vLLM and in
        ``decode``), so the only other evidence is a routing trace.

        The two policies then differ, and not by accident:

        * ``PRESERVE_MIXED_HISTORY`` keeps the ids, so a self-named switch persists
          into every later turn and keeps re-routing that region of history.
        * ``RE_PREFILL`` rebuilds from ``messages``, which is text, so it is
          dropped at the turn boundary. Lossy by construction, not a bug.

        A non-empty entry therefore means the two policies no longer describe the
        same conversation -- worth knowing before switching between them.

        One entry per recorded answer, in order:

        * ``[]``     -- checked, the model emitted none;
        * ``[id, ...]`` -- it emitted these;
        * ``None``   -- **not checkable**, because the answer was recorded as text.
          By then the token is already gone (detokenization strips it), so a
          re-encoding cannot recover it. Pass the model's ids to get an answer.
        """
        return [
            None if ids is None else list(ids) for ids in self._generated_control_tokens
        ]

    # ── prompt construction ───────────────────────────────────────────────────
    def build_prompt(self, adapter=None, **template_kwargs):
        """Return the token ids to send for the next assistant turn.

        Does not mutate the transcript: a prompt whose answer is never recorded
        (a guardian screen, a judge call) leaves the conversation untouched.

        Args:
            adapter: adapter name for *this* turn, or ``None`` for base.
            **template_kwargs: forwarded to ``apply_chat_template`` (e.g.
                ``documents=[...]`` for RAG adapters).
        """
        if self.policy is KVHistoryPolicy.RE_PREFILL or not self._sent_ids:
            ids = self._render_full(adapter, **template_kwargs)
        else:
            delta = self._delta(adapter, **template_kwargs)
            ids = self._sent_ids + self._encode(delta)
            # Same comparison _assert_control_budget makes, so anything that reaches
            # it over budget is necessarily a full render -- which is what its error
            # message claims.
            if self._control_count(ids) > MAX_RETAINED_CONTROL_TOKENS:
                ids = self._reprefill(adapter, ids, **template_kwargs)

        self._assert_control_budget(ids)
        self._pending = ids
        self._pending_adapter = adapter
        self._pending_template_kwargs = dict(template_kwargs)
        return PromptTokenIds(ids, self.policy)

    # ── transport helpers ─────────────────────────────────────────────────────
    def completion_payload(self, adapter=None, **extra):
        """A ready ``/v1/completions`` body for the next turn.

        The correct call for both policies, and the only correct one for
        ``PRESERVE_MIXED_HISTORY``: ``prompt`` carries token ids, so the server
        neither re-renders nor re-tokenizes.

        Args:
            adapter: adapter name for this turn, or ``None`` for base.
            **extra: merged into the body (``max_tokens``, ``temperature``, ...).
        """
        return {"prompt": list(self.build_prompt(adapter=adapter)), **extra}

    def record_answer(self, answer, adapter=None):
        """Record the assistant turn produced by the last :meth:`build_prompt`.

        Args:
            answer: the reply, as text or as the token ids the model emitted.
                Ids are preferred when available (``llm.generate`` returns them):
                text has to be re-encoded, and detokenize/retokenize is not
                guaranteed to round-trip for every string.
            adapter: the adapter that produced it; recorded for provenance.
        """
        if self._pending is None:
            raise RuntimeError(
                "record_answer() without a preceding build_prompt(): there is no "
                "record of what the model was actually sent, so the transcript "
                "cannot be extended."
            )

        if isinstance(answer, str):
            text, answer_ids = answer, None
        else:
            answer_ids = list(answer)
            text = self.tokenizer.decode(answer_ids, skip_special_tokens=True)

        self._messages.append({"role": "assistant", "content": text})
        self._turn_adapters.append(adapter)
        self._sent_messages = len(self._messages)

        # Surface a control token the model named itself. Only answerable from
        # ids: the text arrived with the token already stripped, so re-encoding it
        # can never reveal one. None therefore means "not checkable", which is a
        # different fact from [] meaning "checked, none present".
        self._generated_control_tokens.append(
            None
            if answer_ids is None
            else [int(t) for t in answer_ids if t in self._control_ids]
        )

        if self.policy is KVHistoryPolicy.PRESERVE_MIXED_HISTORY:
            # Close the turn on a special token so the next delta can be appended
            # without tokenizer merges shifting ids across the seam.
            tail = answer_ids if answer_ids is not None else self._encode(text)
            self._sent_ids = self._pending + list(tail) + self._encode(self._turn_end())
            self._sent_template_kwargs = dict(self._pending_template_kwargs)

        self._pending = None
        return text

    # ── internals ─────────────────────────────────────────────────────────────
    def _render_full(self, adapter, **template_kwargs):
        """The whole transcript rendered from ``_messages`` -- the RE_PREFILL shape.

        Carries a control token for THIS turn only: ``_render`` passes
        ``adapter_name`` for the current adapter, and ``record_answer`` decodes
        answers with ``skip_special_tokens=True``, so no earlier turn's token
        survives in ``_messages`` to be re-rendered. That is what makes this a
        reset rather than an attempt at one.
        """
        rendered = self._render(
            self._messages, gen=True, adapter=adapter, **template_kwargs
        )
        if self.policy is KVHistoryPolicy.PRESERVE_MIXED_HISTORY:
            self._reject_lora_placement(rendered, adapter)
        return self._encode(rendered)

    def _reprefill(self, adapter, over_budget_ids, **template_kwargs):
        """Rebuild this turn as a full render, because the preserved ids got too long.

        The cost is real and paid here rather than deferred: the rebuilt prefix
        diverges from ``_sent_ids`` at or near position 0, so the whole conversation
        recomputes once, and every earlier region is interpreted under base from this
        turn on. PRESERVE then resumes appending deltas to the new baseline.

        The alternative is worse. Past the counting head's exact range two distinct
        counts recover as one address, so two control tokens key the same codeword and
        the memory head returns the mean of the two expert ids they wrote -- an
        arbitrary adapter after rounding, for every token in both their spans. Not a
        lag: the write and the read use the same recovered address, so a lone aliased
        count still routes correctly, and it is the collision that breaks it. Either
        way there is no error and nothing in the output to reveal it.
        """
        ids = self._render_full(adapter, **template_kwargs)
        self._reprefills += 1
        logger.warning(
            "re-prefilled: the preserved prompt reached %d control tokens, past the "
            "%d the coded switch can address exactly in bf16, so history's control "
            "tokens were dropped and the transcript re-rendered -- %d ids instead of "
            "%d. Earlier turns are interpreted under base from now on, and the prefix "
            "cache for this conversation is recomputed once. Re-prefills so far: %d.",
            self._control_count(over_budget_ids),
            MAX_RETAINED_CONTROL_TOKENS,
            len(ids),
            len(over_budget_ids),
            self._reprefills,
        )
        return ids

    def _control_count(self, ids):
        """Control tokens in ``ids``; 0 when the checkpoint configured none."""
        if not self._control_ids:
            return 0
        return sum(1 for t in ids if t in self._control_ids)

    def _render(self, messages, gen, adapter=None, **template_kwargs):
        kwargs = dict(template_kwargs)
        if adapter:
            kwargs["adapter_name"] = adapter
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=gen, tokenize=False, **kwargs
        )

    def _encode(self, text):
        # add_special_tokens=False throughout: these are fragments that get
        # concatenated, and an auto-prepended BOS would land mid-conversation.
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def _turn_end(self):
        """The text that closes an assistant turn, taken from the template.

        Derived rather than hardcoded so both Granite template families (the
        ``<|start_of_role|>`` role-marker format and ChatML) work unchanged.

        Probed with *this turn's* template kwargs, not the template defaults.
        Granite 4.2 renders the assistant boundary differently depending on
        ``enable_thinking``, so probing with the defaults would refuse a
        conversation that the caller had already configured to work.
        """
        probe = [{"role": "user", "content": ""}, {"role": "assistant", "content": ""}]
        kwargs = self._pending_template_kwargs
        rendered = self._render(probe, gen=False, **kwargs)
        opener = self._render(probe[:1], gen=True, **kwargs)
        if not rendered.startswith(opener):
            raise RuntimeError(
                "cannot derive the assistant turn terminator from this chat "
                "template: rendering a user+assistant pair does not extend the "
                "rendering of the user turn plus a generation prompt. Returning "
                "an empty terminator would join the next turn without one, so the "
                "seam would fall mid-text and tokenizer merges could shift every "
                "preserved id. Use KVHistoryPolicy.RE_PREFILL with this template."
            )
        return rendered[len(opener) :]

    def _delta(self, adapter, **template_kwargs):
        """The not-yet-sent turns only, obtained by subtracting two renders.

        Rendering the new turn alone is not an option: ``apply_chat_template``
        always renders a complete conversation, so it would re-emit the system
        block and inject a second system prompt mid-dialogue. Rendering the
        conversation with and without the new turns and taking the difference
        yields exactly what they added -- including this turn's control token, in
        whatever position the template chooses for it.

        ``prev`` is rendered with ``self._sent_template_kwargs``, NOT with this
        turn's ``template_kwargs``. The subtraction is only meaningful against the
        render the preserved ids actually came from. Using this turn's kwargs for
        both sides makes a newly-added kwarg appear in ``prev`` and in ``full``,
        where it cancels out of the delta -- so a turn that first supplies
        ``documents=[...]`` would send a prefix rendered without them and a delta
        that does not contain them either, activating a RAG adapter against an
        empty context with every guard in this module still passing. Rendering
        ``prev`` from the sent kwargs turns that into a failed prefix relation,
        which is reportable.
        """
        prev = self._render(
            self._messages[: self._sent_messages],
            gen=False,
            **self._sent_template_kwargs,
        )
        full = self._render(
            self._messages, gen=True, adapter=adapter, **template_kwargs
        )

        if not full.startswith(prev):
            self._reject_template_kwargs_drift(template_kwargs)
            self._explain_no_prefix(prev, full, adapter)

        delta = full[len(prev) :]
        self._assert_join_boundary(delta)
        return delta

    def _control_texts(self):
        """The control-token strings for this checkpoint, from the config ids.

        Derived through the tokenizer rather than assuming the ``<|name|>``
        spelling, so a checkpoint that names them differently still diagnoses.
        """
        if self._control_text_cache is None:
            texts = []
            for tid in sorted(self._control_ids):
                try:
                    texts.append(self.tokenizer.convert_ids_to_tokens(tid))
                except Exception:  # pragma: no cover - defensive
                    pass
            self._control_text_cache = [t for t in texts if t]
        return self._control_text_cache

    def _reject_lora_placement(self, rendered, adapter):
        """Refuse a LoRA-technology adapter under PRESERVE_MIXED_HISTORY.

        A LoRA adapter's control token is emitted at sequence position 0 (the
        template's LoRA prefix insertion, which also suppresses the role marker
        that would follow). Index 0 is therefore its signature: an aLoRA adapter
        activates either inside a user message or at the assistant boundary, both
        of which sit after the opening marker.

        Position 0 is inside the already-sent prefix for every turn after the
        first, so the delta can never be derived and the policy cannot hold.
        Turn 1 happens to work -- it takes the full-render path, where there is no
        prefix to preserve -- and is refused anyway, because a conversation does
        not change adapter technology mid-dialogue: a LoRA turn 1 is a LoRA turn 2,
        and that turn fails. Succeeding once and then failing on every later turn
        would leave the caller holding a transcript that cannot continue under the
        policy it chose. LoRA adapters use RE_PREFILL.
        """
        if not adapter:
            return
        for text in self._control_texts():
            if text and rendered.startswith(text):
                raise RuntimeError(
                    f"adapter {adapter!r} is a LoRA-technology adapter: its control "
                    f"token {text!r} is emitted at sequence position 0. That is "
                    "inside the already-sent prefix on every turn after the first, "
                    "so PRESERVE_MIXED_HISTORY cannot hold for it. Refused here on "
                    "turn 1, which would otherwise render fine, because a "
                    "conversation does not change adapter technology mid-dialogue: "
                    "a LoRA turn 1 is a LoRA turn 2, and that turn fails. LoRA "
                    "adapters use KVHistoryPolicy.RE_PREFILL; "
                    "PRESERVE_MIXED_HISTORY is for aLoRA adapters, whose control "
                    "token lands in the turn being generated."
                )

    def _reject_template_kwargs_drift(self, template_kwargs):
        """Raise when this turn's template kwargs differ from the sent ones.

        Checked only once the prefix relation has already failed. A kwarg that
        affects the NEW turn's region alone keeps the prefix intact and is
        perfectly legal (the delta carries it); a kwarg that re-renders the
        history -- ``documents=[...]`` lands in the system block for Granite --
        moves text the preserved ids already fixed, and no delta can repair that,
        because those ids have been sent and are what the server's cache holds.
        """
        if template_kwargs == self._sent_template_kwargs:
            return
        changed = sorted(
            set(template_kwargs) ^ set(self._sent_template_kwargs)
            | {
                k
                for k in set(template_kwargs) & set(self._sent_template_kwargs)
                if template_kwargs[k] != self._sent_template_kwargs[k]
            }
        )
        raise RuntimeError(
            f"template kwargs changed mid-conversation ({', '.join(changed)}), and "
            f"the new values re-render turns this conversation has already sent. "
            f"PRESERVE_MIXED_HISTORY appends to the exact ids already sent, so text "
            f"that moves inside the preserved region cannot be represented -- "
            f"continuing would send a prompt missing it entirely. Pass the same "
            f"template kwargs on every turn (supply documents=[...] from the first "
            f"turn, empty if there are none yet), or use "
            f"KVHistoryPolicy.RE_PREFILL for this conversation."
        )

    def _explain_no_prefix(self, prev, full, adapter):
        """Raise for a missing prefix relation, naming the real cause.

        The usual cause is NOT a defective template -- it is *where this turn's
        control token landed*. The delta is ``full[len(prev):]``, so the trick
        holds exactly when that token sits at or after ``len(prev)``, i.e. inside
        the region the new turn added. Placements that break it:

          * a LoRA-technology adapter, whose token goes to index 0;
          * an aLoRA adapter whose invocation text appears in an EARLIER user
            message, so Pass 1 targets that message and rewrites the history.

        Both leave the template itself append-only -- a no-adapter render still
        yields a clean prefix -- which is why "the template is not append-only"
        was the wrong diagnosis to report.

        Granite 4.2 with ``truncate_history_thinking`` on is a template that
        genuinely is not append-only, and it lands on the generic message below.
        The remedy for it is that flag, not RE_PREFILL; see
        ``docs/SUPPORTED_MODELS.md``.
        """
        # A fresh render carries at most one control token, so the earliest
        # occurrence of any of them is this turn's.
        idx = min(
            (i for i in (full.find(t) for t in self._control_texts()) if i >= 0),
            default=-1,
        )
        if idx == 0:
            # Same condition _reject_lora_placement names on turn 1; reached here
            # only if a conversation was built before that check existed.
            self._reject_lora_placement(full, adapter)
        if 0 <= idx < len(prev):
            raise RuntimeError(
                f"cannot derive a delta for adapter {adapter!r}: its control token "
                f"is placed at character {idx} of the rendered prompt, inside the "
                f"{len(prev)} characters this conversation has already sent. "
                "PRESERVE_MIXED_HISTORY appends the new turn to the ids already "
                "sent, so this turn's control token has to land in the new turn's "
                "region. It does when the adapter activates at the assistant "
                "boundary, or when its invocation text appears in the NEWEST user "
                "message. It does not for a LoRA-technology adapter (token at "
                "index 0), nor for an aLoRA adapter whose invocation text appears "
                "in an earlier user message. Use the aLoRA flavour of this "
                "adapter, put its invocation text in the current turn, or use "
                "KVHistoryPolicy.RE_PREFILL for this conversation."
            )
        raise RuntimeError(
            "cannot derive a delta: this chat template is not append-only for a "
            "growing message list, so slicing the longer render would cut in the "
            "wrong place. Use KVHistoryPolicy.RE_PREFILL with this checkpoint."
        )

    def _assert_join_boundary(self, delta):
        """The seam must sit on a special token, or ids can shift across it.

        ``tok(X + Y)`` is not generally ``tok(X) + tok(Y)``: a merge spanning the
        join would renumber tokens and the carefully preserved prefix would stop
        matching the cache -- silently, as a fallen hit rate rather than an error.
        """
        if not delta.startswith("<|"):
            raise RuntimeError(
                "the appended turn must begin on a special token so tokenizer "
                "merges cannot span the join; the template produced "
                f"{delta[:32]!r}. Use KVHistoryPolicy.RE_PREFILL with this template."
            )

    def _assert_control_budget(self, ids):
        count = self._control_count(ids)
        if count > MAX_RETAINED_CONTROL_TOKENS:
            raise RuntimeError(
                f"{count} control tokens in one request exceeds "
                f"{MAX_RETAINED_CONTROL_TOKENS}, the range over which the coded "
                "switch recovers a write address exactly in bf16; past it addresses "
                "alias and routing degrades silently. This prompt is already a full "
                "render of the transcript, so re-prefilling cannot reduce it -- the "
                "count is coming from the current turn, or from control-token text "
                "recorded into a message (record_answer() with ids strips them; with "
                "text it cannot). Shorten the transcript, or stop recording control "
                "tokens into answers."
            )
