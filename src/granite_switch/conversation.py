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

Requirements:

* the prompt must be sent as token ids (``/v1/completions`` with ``prompt=[ids]``,
  or ``model.generate(input_ids=...)``). ``/v1/chat/completions`` re-renders and
  re-tokenizes server-side, which drops earlier control tokens and degrades cache
  reuse. Both policies reuse the ids already sent as a stable prefix, so both
  require ids; ``completion_payload`` builds the body.

No adapter-technology enforcement. Each turn appends when its control token lands
in the new-turn region -- aLoRA inside the user message or at the assistant
boundary, SR at the generation-prompt boundary -- and falls back to a full text
render when it cannot. A LoRA adapter's control token is at sequence position 0
(the template's LoRA prefix insertion, which also suppresses the role marker that
would follow), inside the already-sent prefix on every turn after the first, so a
LoRA turn always full-renders. This happens on its own; nothing detects "LoRA".
Under ``PRESERVE_MIXED_HISTORY`` that full render is a re-prefill -- it drops the
preserved control tokens for that turn.

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
        it and its cached KV stays eligible for reuse. aLoRA and SR turns append;
        a LoRA turn (control token at position 0) cannot be appended and falls
        back to a full text render for that turn, dropping the preserved control
        tokens -- see the module docstring.

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

        # RE_PREFILL id reuse. _re_base_ids is the base-demoted history prefix
        # already sent (control tokens dropped), carried verbatim from the ids we
        # sent last turn; _re_base_text is the render it came from, for the delta
        # subtraction. It lags one turn: a turn's base form is not sent until the
        # next turn renders it clean. None means "no reusable baseline yet".
        self._re_base_ids: list[int] | None = None
        self._re_base_text: str | None = None
        self._pending_re_base_ids: list[int] | None = None
        self._pending_re_base_text: str | None = None

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
        if self.policy is KVHistoryPolicy.RE_PREFILL:
            ids = self._build_reprefill(adapter, **template_kwargs)
        elif not self._sent_ids:
            ids = self._render_full(adapter, **template_kwargs)
        else:
            prev = self._render(
                self._messages[: self._sent_messages],
                gen=False,
                **self._sent_template_kwargs,
            )
            full = self._render(
                self._messages, gen=True, adapter=adapter, **template_kwargs
            )
            if self._appendable(full, prev):
                delta = self._delta(adapter, **template_kwargs)
                ids = self._sent_ids + self._encode(delta)
                # Same comparison _assert_control_budget makes, so anything that
                # reaches it over budget is necessarily a full render -- which is
                # what its error message claims.
                if self._control_count(ids) > MAX_RETAINED_CONTROL_TOKENS:
                    ids = self._reprefill(
                        adapter,
                        ids,
                        reason=(
                            f"the preserved prompt reached {self._control_count(ids)} "
                            f"control tokens, past the {MAX_RETAINED_CONTROL_TOKENS} "
                            "the coded switch can address exactly in bf16"
                        ),
                        **template_kwargs,
                    )
            elif self._render(
                self._messages, gen=True, adapter=None, **template_kwargs
            ).startswith(prev):
                # A no-adapter render still extends prev, so the TEMPLATE is
                # append-only -- the divergence is only where this turn's control
                # token landed (a LoRA token at index 0, or an aLoRA invocation in
                # an earlier message). No delta can carry it, so re-prefill instead
                # of raising: a full render, history's control tokens dropped.
                ids = self._reprefill(
                    adapter,
                    self._encode(full),
                    reason="this turn's control token lands inside the "
                    "already-sent prefix, so no delta can carry it",
                    **template_kwargs,
                )
            else:
                # Even a no-adapter render fails the prefix relation: the template
                # rewrites history when a turn is added (e.g. Granite 4.2 thinking
                # truncation), or the template kwargs drifted. Raise -- a
                # re-prefill would silently hide a template that cannot be served.
                self._reject_template_kwargs_drift(template_kwargs)
                self._explain_no_prefix()

        self._assert_control_budget(ids)
        self._pending = ids
        self._pending_adapter = adapter
        self._pending_template_kwargs = dict(template_kwargs)
        return PromptTokenIds(ids, self.policy)

    def _build_reprefill(self, adapter, **template_kwargs):
        """RE_PREFILL: reuse base-demoted ids for history, text-render the tail.

        ``base_hist`` is the completed history rendered with no adapter. Answers
        are stored as text with control tokens stripped, so a no-adapter render is
        base by construction -- no substitution needed. ``full`` is the same plus
        the current turn, with this turn's adapter.

        When this turn's control token lands in the delta, reuse the base prefix
        already sent (``_re_base_ids``) and append; the deep prefix stays
        byte-identical to the ids sent last turn, so the cache hits. When it does
        not -- turn 1, a LoRA token at index 0, or a non-append-only template --
        full-render and drop the reusable baseline. The just-demoted turn is
        always freshly rendered here (its base ids never existed before): a
        one-turn lag.
        """
        committed = self._messages[: self._sent_messages]
        full = self._render(
            self._messages, gen=True, adapter=adapter, **template_kwargs
        )

        if not committed:
            # Turn 1: nothing sent yet, nothing to reuse. Full render; turn 2 will
            # render this turn's base form itself.
            self._pending_re_base_ids = None
            self._pending_re_base_text = None
            return self._encode(full)

        base_hist = self._render(committed, gen=False, adapter=None, **template_kwargs)
        if not self._appendable(full, base_hist):
            # LoRA (control token at index 0) or a non-append-only template: full
            # render. RE_PREFILL re-renders each turn anyway, so this is correct;
            # only the id reuse is forgone.
            self._pending_re_base_ids = None
            self._pending_re_base_text = None
            return self._encode(full)

        current_text = full[len(base_hist) :]
        self._assert_join_boundary(current_text)

        if self._re_base_text is not None and base_hist.startswith(self._re_base_text):
            base_delta = base_hist[len(self._re_base_text) :]
            # The just-demoted turn joins the reused prefix here; the seam is a
            # turn boundary (a role marker) and must be a special token, or a
            # tokenizer merge across it would renumber the reused ids.
            if base_delta:
                self._assert_join_boundary(base_delta)
            prefix_ids = self._re_base_ids + self._encode(base_delta)
        else:
            prefix_ids = self._encode(base_hist)

        ids = prefix_ids + self._encode(current_text)
        # Carry the prefix we actually sent (history minus the current turn's
        # region) forward; the current region holds a control token, so it is not
        # reusable until it is re-rendered to base next turn.
        self._pending_re_base_ids = prefix_ids
        self._pending_re_base_text = base_hist
        return ids

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
        elif self.policy is KVHistoryPolicy.RE_PREFILL:
            # Commit the base baseline this turn established: the history prefix
            # (minus the current turn's adaptered region) that the NEXT turn reuses.
            self._re_base_ids = self._pending_re_base_ids
            self._re_base_text = self._pending_re_base_text

        self._pending = None
        return text

    # ── internals ─────────────────────────────────────────────────────────────
    def _render_full(self, adapter, **template_kwargs):
        """The whole transcript rendered from ``_messages`` -- the full-render shape.

        Carries a control token for THIS turn only: ``_render`` passes
        ``adapter_name`` for the current adapter, and ``record_answer`` decodes
        answers with ``skip_special_tokens=True``, so no earlier turn's token
        survives in ``_messages`` to be re-rendered. That is what makes this a
        reset rather than an attempt at one. It is also the fallback for any turn
        that cannot be appended -- a LoRA turn (control token at position 0), or a
        template that is not append-only.
        """
        rendered = self._render(
            self._messages, gen=True, adapter=adapter, **template_kwargs
        )
        return self._encode(rendered)

    def _reprefill(self, adapter, current_ids, reason, **template_kwargs):
        """Rebuild this turn as a full render, dropping history's control tokens.

        Two callers, one mechanism: the preserved ids exceeded the counting head's
        exact range, or this turn's control token cannot be appended (it lands in
        the already-sent prefix). Either way the rebuilt prefix diverges from
        ``_sent_ids`` at or near position 0, so the whole conversation recomputes
        once and every earlier region is interpreted under base from this turn on.
        PRESERVE then resumes appending deltas to the new baseline. ``reason`` is
        logged so the two causes are distinguishable; ``current_ids`` is the ids
        being replaced, for the length in the log.

        For the counting-head case the alternative is worse: past the exact range
        two distinct counts recover as one address, so two control tokens key the
        same codeword and the memory head returns the mean of the two expert ids
        -- an arbitrary adapter after rounding, with no error in the output.
        """
        ids = self._render_full(adapter, **template_kwargs)
        self._reprefills += 1
        logger.warning(
            "re-prefilled (%s): history's control tokens were dropped and the "
            "transcript re-rendered -- %d ids instead of %d. Earlier turns are "
            "interpreted under base from now on, and the prefix cache for this "
            "conversation is recomputed once. Re-prefills so far: %d.",
            reason,
            len(ids),
            len(current_ids),
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
            self._explain_no_prefix()

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

    def _control_char_index(self, rendered):
        """Char offset of this turn's control token in ``rendered``, or -1.

        A fresh render carries at most one control token, so the earliest
        occurrence of any control string is this turn's.
        """
        idxs = [i for i in (rendered.find(t) for t in self._control_texts()) if i >= 0]
        return min(idxs, default=-1)

    def _appendable(self, full_text, prev_text):
        """True when the new turn can be appended as a delta rather than re-rendered.

        The delta is ``full_text[len(prev_text):]``. It carries this turn's
        control token only when that token sits at or after ``len(prev_text)`` --
        i.e. in the region the new turn added. A LoRA control token at index 0
        never satisfies this, so a LoRA turn is not appendable and full-renders.
        A turn with no adapter (no control token) is appendable whenever the
        render is append-only.
        """
        if not full_text.startswith(prev_text):
            return False
        idx = self._control_char_index(full_text)
        return idx == -1 or idx >= len(prev_text)

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

    def _explain_no_prefix(self):
        """Raise for a template that is not append-only for a growing message list.

        Reached only when the prefix relation fails for a turn that WAS
        appendable by control-token placement (``_appendable`` already routed a
        LoRA turn, whose token sits at index 0, to a full render upstream). What
        remains is a template whose earlier turns re-render when a new one is
        added -- Granite 4.2 with ``truncate_history_thinking`` on is the real
        case; the remedy is that flag, not a policy change. See
        ``docs/SUPPORTED_MODELS.md``.
        """
        raise RuntimeError(
            "cannot derive a delta: this chat template is not append-only for a "
            "growing message list, so slicing the longer render would cut in the "
            "wrong place. Use a template that is append-only, or enable "
            "truncate_history_thinking. See docs/SUPPORTED_MODELS.md."
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
