# SPDX-License-Identifier: Apache-2.0
"""Native vLLM structured-decisions support for Granite Switch (Route C).

vLLM 0.32 ships a first-class structured-decisions API (``/v1/systemone``) under
``vllm/entrypoints/generate/structured_decisions/``. It reads a labelled question as the
reply's first token and softmaxes over the option-label logprobs. Two things stop it from
serving the Granite Switch *decision adapter* out of the box:

1. ``select_read_strategy`` looks ``model_config.architecture`` up in ``READ_STRATEGIES`` and
   raises (→ 501) on a miss. ``GraniteSwitchForCausalLM`` is not in that dict.
2. The native read never passes an ``adapter_name`` to ``apply_chat_template``, so even once
   the architecture is allowlisted it would read the *base model's* first-token guess, not the
   trained decision adapter's answer. The adapter activates only when the checkpoint's chat
   template emits a ``<|decision|>`` control token, which happens when the question turn carries
   a ``### DECISION`` invocation marker and ``apply_chat_template(..., adapter_name="decision")``
   is called.

``GraniteSwitchDecisionStrategy`` subclasses ``NextTokenStrategy`` and overrides only the
render-time behaviour: it forces ``adapter_name`` into ``chat_template_kwargs`` (turning the
control token on), builds the two-turn marker layout the adapter was trained on, and uses the
adapter's letter-alias label pool (A..Z then AA, AB, ...; **no digits**) instead of the native
per-type defaults. The read/softmax/response machinery is inherited unchanged.

``register_structured_decisions`` (a ``vllm.general_plugins`` entry point) inserts the strategy
into ``READ_STRATEGIES`` at startup, the same way ``register_granite_switch`` registers the model.

The render helpers (``build_label_pool``, ``escape_special_tokens``, ``TOKEN_BREAK``, the
two-turn layout) are copied verbatim from the endpoint plugin's renderer to keep one source of
truth for the trained layout; keep them in sync with
``granite-switch-systemone/src/granite_switch_decision/renderer.py``.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from typing import Any

try:  # vLLM's logger when available; stdlib fallback keeps the pure helpers importable on CPU.
    from vllm.logger import init_logger

    logger = init_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

# ── Trained layout constants (mirror the endpoint plugin's env var names) ──────────────────

PLAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

_INVOCATION_MARKER_ENV = "DECISION_INVOCATION_MARKER"
_ADAPTER_NAME_ENV = "DECISION_ADAPTER_NAME"
_SYSTEM_PROMPT_ENV = "DECISION_EXPERIMENTAL_SYSTEM_PROMPT"

_DEFAULT_ADAPTER_NAME = "decision"

# The trained system prompt, used for the model card's published score. Off by default so a
# decision's state turn can share the prefix cache with chat; experimental.
SYSTEM_PROMPT = (
    "You are a decision engine. Read the state, then answer the question by replying with "
    "exactly one option label from the list. Do not explain."
)

# The architecture strings register() points at the shared GraniteSwitchForCausalLM class.
# All three can serve decisions, so all three get the strategy.
GRANITE_SWITCH_ARCHS = (
    "GraniteSwitchForCausalLM",
    "SRSwitchForCausalLM",
    "ShadowResidualForCausalLM",
)

# U+2060 WORD JOINER: invisible, and splits a special-token string so the tokenizer reads it as
# text rather than as the special token.
TOKEN_BREAK = "⁠"


def _invocation_marker() -> str:
    marker = os.environ.get(_INVOCATION_MARKER_ENV)
    if not marker:
        raise RuntimeError(
            f"{_INVOCATION_MARKER_ENV} is not set: set it to the text the decision adapter "
            'was trained to activate on (e.g. "### DECISION").'
        )
    return marker


def _adapter_name() -> str:
    return os.environ.get(_ADAPTER_NAME_ENV) or _DEFAULT_ADAPTER_NAME


def _system_prompt_enabled() -> bool:
    value = os.environ.get(_SYSTEM_PROMPT_ENV, "0")
    if value not in ("0", "1"):
        raise RuntimeError(f"{_SYSTEM_PROMPT_ENV}={value!r} must be 0 or 1")
    return value == "1"


def text_of(x: object) -> str:
    """`state`/a description may be a dict/list (jev allows `string | object | array` for
    both); render JSON on one line, pass strings through, "" for None."""
    if isinstance(x, (dict, list)):
        return json.dumps(x, ensure_ascii=False)
    return "" if x is None else str(x)


def escape_special_tokens(text: str, specials: list[str]) -> str:
    """Break every occurrence of the tokenizer's added tokens in caller text, so state,
    instructions and descriptions cannot forge turns or a second adapter activation."""
    for special in specials:
        if special in text:
            text = text.replace(special, special[0] + TOKEN_BREAK + special[1:])
    return text


def _specials(tok) -> list[str]:
    added: Callable[[], dict[str, int]] | None = getattr(tok, "get_added_vocab", None)
    return (
        sorted(list(added()), key=lambda token: len(token), reverse=True)
        if added
        else []
    )


def build_label_pool(tok, need: int) -> list[tuple[str, int]]:
    """`need` single-token labels: A..Z first, then the two-letter combinations AA, AB, ...
    that tokenize to one token, so wide (>26-option) `choice` questions still get a valid label
    pool. This is the alias scheme the adapter was trained on; digits were not in its training
    pool. Returns `(label_string, token_id)` pairs."""
    pool: list[tuple[str, int]] = []
    seen: set[int] = set()
    candidates = list(PLAIN_LETTERS)
    candidates += [a + b for a in PLAIN_LETTERS for b in PLAIN_LETTERS]
    for label in candidates:
        if len(pool) >= need:
            break
        ids = tok(label, add_special_tokens=False)["input_ids"]
        if len(ids) == 1 and ids[0] not in seen:
            pool.append((label, ids[0]))
            seen.add(ids[0])
    if len(pool) < need:
        raise ValueError(
            f"could only build {len(pool)} single-token labels, need {need}"
        )
    return pool[:need]


def render(
    tok, state: object, instructions: object, descriptions: list[object]
) -> tuple[list[dict[str, str]], list[int]]:
    """Messages and label token ids for one question; `label_token_ids[i]` answers
    `descriptions[i]`. The two-turn marker layout the decision adapter was trained on."""
    labels = build_label_pool(tok, len(descriptions))
    marker = _invocation_marker()
    # The chat template activates at the turn's last marker, so caller text must not carry one.
    specials = [*_specials(tok), marker]
    ctx = text_of(state) if state not in ("", None, {}, []) else "(empty)"
    q_lines = [
        "Question:",
        escape_special_tokens(text_of(instructions), specials),
        "",
        "Options:",
    ]
    for (label, _), description in zip(labels, descriptions):
        desc_text = text_of(description) if description not in (None, "") else label
        q_lines.append(f"{label}: {escape_special_tokens(desc_text, specials)}")
    q_lines += ["", "Reply with exactly one option label."]
    second_turn = marker + "\n" + "\n".join(q_lines)
    messages = [
        {"role": "user", "content": "State:\n" + escape_special_tokens(ctx, specials)},
        {"role": "user", "content": second_turn},
    ]
    if _system_prompt_enabled():
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    return messages, [token_id for _, token_id in labels]


def check_activation(tok, adapter_name: str | None = None) -> None:
    """Refuse to start unless a rendered probe question carries exactly one adapter control
    token, placed inside the conversation. With a wrong marker the chat template either adds
    none or falls back to putting one just before the assistant header, so the question would
    silently run on the base model.

    Ported from the endpoint plugin's `plugin._check_activation`.
    """
    adapter_name = adapter_name or _adapter_name()
    control = f"<|{adapter_name}|>"
    control_id = (
        tok.convert_tokens_to_ids(control)
        if hasattr(tok, "convert_tokens_to_ids")
        else None
    )
    if not hasattr(tok, "apply_chat_template") or control_id in (
        None,
        getattr(tok, "unk_token_id", None),
    ):
        logger.warning(
            "cannot check adapter activation: no chat template or no %s token", control
        )
        return
    messages, _ = render(
        tok, "probe state", "Is this a probe?", ["No / false", "Yes / true"]
    )

    def ids(add_generation_prompt: bool) -> list[int]:
        text = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            adapter_name=adapter_name,
        )
        return tok(text, add_special_tokens=False)["input_ids"]

    full, conversation = ids(True), ids(False)
    in_conversation = control_id in conversation[:-1]
    if full.count(control_id) != 1 or not in_conversation:
        raise RuntimeError(
            f"a probe decision prompt does not carry exactly one {control} inside the "
            "question turn: check DECISION_INVOCATION_MARKER and DECISION_ADAPTER_NAME "
            "against the checkpoint's chat template"
        )


def _build_strategy_class():
    """Build the strategy class lazily so importing this module does not require vLLM's
    structured_decisions package (absent on < 0.32). Returns the class."""
    from vllm.entrypoints.generate.structured_decisions.question_types import (
        StructuredDecisionError,
    )
    from vllm.entrypoints.generate.structured_decisions.strategies import (
        DecisionLimits,
        NextTokenStrategy,
        ReadContext,
        prompt_tail,
    )

    class GraniteSwitchDecisionStrategy(NextTokenStrategy):
        """Granite Switch decision adapter via the native ``/v1/systemone`` route.

        Overrides only the render-time behaviour of ``NextTokenStrategy``:

        - ``__init__`` builds the prompt tail but *skips* the native digit/yes-no label probe
          (``type_label_ids``): Granite was not trained with digit labels, so those tokens are
          irrelevant and the probe could spuriously raise. The letter-alias pool is built
          per-question in ``_label_groups`` instead.
        - ``_read_request`` injects ``adapter_name`` into ``chat_template_kwargs`` so the chat
          template emits the ``<|decision|>`` control token (preserving the inherited
          thinking-off guard via ``super()``).
        - ``read`` emits the **two-turn** marker layout (State turn + ``### DECISION`` question
          turn) the adapter was trained on, escaping all caller text with U+2060. The inherited
          tail check / sampling / softmax / response assembly are reused.
        - ``_label_groups`` returns the trained letter-alias token ids for the question's
          options, not the native per-type labels.
        """

        def __init__(self, context: ReadContext):
            # Bypass NextTokenStrategy.__init__'s type_label_ids probe (digit labels), which is
            # irrelevant to Granite's letter-alias scheme and would couple us to the native
            # label sets. We still need the tail for the inherited end-of-prompt check.
            from vllm.entrypoints.generate.structured_decisions.strategies import (
                ReadStrategy,
            )

            ReadStrategy.__init__(self, context)
            self.adapter_name = _adapter_name()
            tok = self._tokenizer()
            self.tail = prompt_tail(tok, self._generation_prompt())
            check_activation(tok, self.adapter_name)

        def limits(self) -> DecisionLimits:
            # The letter-alias pool extends past 26; the ceiling is the two-letter aliases that
            # tokenize to one token. Probe with a generous need and report what we can build.
            tok = self._tokenizer()
            try:
                built = len(build_label_pool(tok, 26 + 26 * 26))
            except ValueError as e:
                # build_label_pool raises with "could only build N single-token labels"
                built = int(str(e).split("could only build ", 1)[1].split(" ", 1)[0])
            return DecisionLimits(max_questions=64, max_options=built)

        def _read_request(
            self,
            chat_template_kwargs: dict[str, Any] | None,
            cache_salt: str | None = None,
        ):
            # Force adapter activation. Never caller-supplied: the adapter is the server's, and
            # a caller-set adapter_name could read a different (or no) adapter.
            kwargs = dict(chat_template_kwargs or {})
            kwargs["adapter_name"] = self.adapter_name
            return super()._read_request(kwargs, cache_salt)

        def _label_groups(self, question) -> list[list[int]]:
            pool = build_label_pool(self._tokenizer(), len(question.options))
            return [[token_id] for _, token_id in pool]

        async def read(
            self,
            questions,
            instructions,
            state,
            *,
            request_id,
            chat_template_kwargs,
            lora_request,
            priority,
            cache_salt=None,
            trace_headers=None,
            media=None,
        ):
            import math

            from vllm.entrypoints.generate.label_reads import next_token_label_reads
            from vllm.entrypoints.generate.structured_decisions.question_types import (
                label_softmax,
            )
            from vllm.entrypoints.generate.structured_decisions.strategies import (
                QuestionRead,
            )

            if media:
                raise StructuredDecisionError(
                    "the Granite Switch decision adapter reads text, not media"
                )

            ctx = self.context
            read_request = self._read_request(chat_template_kwargs, cache_salt)

            slots, engine_inputs, params = [], [], []
            for q in questions:
                descriptions = [o.description for o in q.options]
                messages, label_ids = render(
                    self._tokenizer(), state, q.instructions, descriptions
                )
                if instructions:
                    messages.insert(0, {"role": "system", "content": instructions})
                engine_input, prompt_ids = await self._render(read_request, messages)
                if prompt_ids[len(prompt_ids) - len(self.tail) :] != self.tail:
                    raise StructuredDecisionError(
                        "these chat options end the prompt differently, so the labels' "
                        "tokens are unknown"
                    )
                groups = [[tid] for tid in label_ids]
                slots.append(groups)
                engine_inputs.append(self._read_input(engine_input, prompt_ids))
                params.append(
                    self._sampling_params(
                        [i for ids in groups for i in ids], prompt_ids
                    )
                )

            label_reads = await next_token_label_reads(
                ctx.engine_client,
                engine_inputs,
                params,
                request_id,
                lora_request=lora_request,
                trace_headers=trace_headers,
                priority=priority,
            )

            reads = []
            for groups, label_read in zip(slots, label_reads):
                output = label_read.result.outputs[0]
                logprobs = iter(label_read.logprobs)
                reads.append(
                    QuestionRead(
                        probs=label_softmax(
                            [max(next(logprobs) for _ in g) for g in groups]
                        ),
                        label_mass=sum(math.exp(lp) for lp in label_read.logprobs),
                        argmax_is_label=bool(output.token_ids)
                        and any(output.token_ids[0] in ids for ids in groups),
                        input_tokens=len(label_read.result.prompt_token_ids or ()),
                        output_tokens=len(output.token_ids),
                    )
                )
            return reads

    return GraniteSwitchDecisionStrategy


def register_structured_decisions() -> None:
    """Insert the Granite Switch decision strategy into vLLM's ``READ_STRATEGIES``.

    Called by vLLM's plugin system on startup (the ``vllm.general_plugins`` entry-point group).
    Re-entrant: registering the same class twice is a no-op. A no-op when the running vLLM has
    no structured-decisions API (< 0.32), so the plugin is safe to load on older lines.
    """
    try:
        from vllm.entrypoints.generate.structured_decisions.strategies import (
            READ_STRATEGIES,
        )
    except ImportError:
        logger.debug(
            "vLLM has no structured_decisions API; skipping decision strategy registration"
        )
        return

    strategy = _build_strategy_class()
    for arch in GRANITE_SWITCH_ARCHS:
        READ_STRATEGIES[arch] = strategy
    logger.info(
        "Granite Switch decision strategy registered for %s",
        ", ".join(GRANITE_SWITCH_ARCHS),
    )
