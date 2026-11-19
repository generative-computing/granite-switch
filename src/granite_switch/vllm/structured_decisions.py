# SPDX-License-Identifier: Apache-2.0
"""Granite Switch decision adapters on vLLM's native ``/v1/systemone`` route.

vLLM 0.32 reads a labelled question as the reply's first token and softmaxes over the
option-label logprobs. It knows nothing of Granite Switch adapters: its
``READ_STRATEGIES`` does not list our architectures, and its read never passes an
``adapter_name`` to the chat template, so it would read the base model's guess.

``GraniteSwitchDecisionStrategy`` (a ``NextTokenStrategy``) closes both gaps and keeps
no adapter knowledge itself. Which adapters answer decisions, and how each is asked
(marker, turns, option lines, labels, question types), comes from the checkpoint's
``config.json`` (``decision_adapter`` / ``decision_specs``), which the composer
resolves from each adapter's io.yaml ``decision:`` block (see
``granite_switch.decision_spec`` and docs/DECISION_ADAPTERS.md). The strategy forces
the chosen ``adapter_name`` into the chat template, escapes all caller text, and
reuses vLLM's read / softmax / response machinery.

``register_structured_decisions`` is the ``vllm.general_plugins`` entry point that
inserts the strategy into ``READ_STRATEGIES``.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from granite_switch.config import VLLM_ARCHITECTURES
from granite_switch.decision_spec import (
    DecisionSpecError,
    build_label_pool,
    check_activation,
    control_token_id,
    render_question,
    special_strings,
    text_of,
    validate_decision_fields,
)

try:  # vLLM's logger when available; the stdlib one keeps this importable on CPU.
    from vllm.logger import init_logger

    logger = init_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

#: The vLLM line that moved the strategies out of this path.
_MOVED_IN = (0, 33)
_STRATEGIES = "vllm.entrypoints.generate.structured_decisions.strategies"
#: Caller ``chat_template_kwargs`` keys a read accepts; the template renders the rest
#: unescaped, and ``adapter_name`` is checked against the declared decision adapters.
_CALLER_KWARGS = frozenset({"adapter_name", "enable_thinking"})
_NO_DECISION_ADAPTER = (
    "this checkpoint declares no decision adapter. Compose it with "
    "--decision-adapter NAME, or stamp it with "
    "`python -m granite_switch.composer.decision_config --model DIR "
    "--decision-adapter NAME`"
)


def load_decision_specs(hf_config) -> tuple[str | None, dict[str, dict]]:
    """``(default adapter, {name: spec})`` from a checkpoint config.

    Re-normalized here because ``--hf-overrides`` sets the fields after the config's
    own validation ran.
    """
    return validate_decision_fields(
        getattr(hf_config, "adapter_names", None),
        getattr(hf_config, "decision_adapter", None),
        getattr(hf_config, "decision_specs", None),
        getattr(hf_config, "adapter_kinds", None),
    )


def _build_strategy_class():
    """Build the class lazily: the base classes exist only on vLLM >= 0.32."""
    from vllm.entrypoints.generate.label_reads import next_token_label_reads
    from vllm.entrypoints.generate.structured_decisions.question_types import (
        LABELS,
        StructuredDecisionError,
        label_softmax,
    )
    from vllm.entrypoints.generate.structured_decisions.strategies import (
        DecisionLimits,
        NextTokenStrategy,
        QuestionRead,
        ReadContext,
        ReadStrategy,
        prompt_tail,
    )

    class GraniteSwitchDecisionStrategy(NextTokenStrategy):
        """Reads a Granite Switch decision adapter, as its spec declares."""

        def __init__(self, context: ReadContext):
            # NextTokenStrategy.__init__ also probes vLLM's own digit / yes-no label
            # sets, which these adapters were not trained on; only the tail is needed.
            ReadStrategy.__init__(self, context)
            # A declared decision adapter that is misconfigured, or that the served
            # template does not activate, must stop the server rather than answer from
            # the base model. vLLM builds the strategy under `except ValueError` and
            # only disables the route at DEBUG level, so raise a RuntimeError.
            try:
                self._load(context)
            except DecisionSpecError as e:
                raise RuntimeError(f"Granite Switch decision adapters: {e}") from e
            if self.specs:
                logger.info(
                    "Granite Switch decision adapters %s (default %r)",
                    sorted(self.specs),
                    self.default_adapter,
                )
            else:
                # Every Granite Switch server loads this plugin; most declare none.
                logger.info("/v1/systemone: %s", _NO_DECISION_ADAPTER)

        def _load(self, context: ReadContext) -> None:
            tok = self._tokenizer()
            hf_config = context.engine_client.model_config.hf_config
            self.default_adapter, self.specs = load_decision_specs(hf_config)
            self.specials = special_strings(tok, self.specs)
            self.label_pools, self.tails, self.control_ids = {}, {}, {}
            for name, spec in self.specs.items():
                probe_ids = check_activation(
                    tok,
                    name,
                    spec,
                    chat_template=context.chat_template,
                    template_kwargs=context.default_chat_template_kwargs,
                )
                # From a rendered question with the adapter active, as real reads
                # are: an SR control token replaces the generation prompt's last
                # token, so the tail depends on the adapter.
                self.tails[name] = prompt_tail(tok, probe_ids)
                self.control_ids[name] = control_token_id(tok, name)
                self.label_pools[name] = build_label_pool(tok, spec["labels"])

        def limits(self) -> DecisionLimits:
            # vLLM 0.32's build_question caps a question at min(max_options, its own
            # label set): A..Z for choice. A wider pool cannot be served, so it is not
            # advertised (see docs/DECISION_ADAPTERS.md, "Limits").
            widest = max((len(p) for p in self.label_pools.values()), default=0)
            cap = len(LABELS)
            return DecisionLimits(max_questions=64, max_options=min(widest or cap, cap))

        def render_state(self, state: Any) -> str:
            return "" if state in ("", None, {}, []) else text_of(state)

        def _select_adapter(self, chat_template_kwargs: dict[str, Any] | None) -> str:
            kwargs = chat_template_kwargs or {}
            unknown = sorted(set(kwargs) - _CALLER_KWARGS)
            if unknown:
                raise StructuredDecisionError(
                    f"chat_template_kwargs {unknown} are not supported; a decision "
                    f"read takes only {sorted(_CALLER_KWARGS)}"
                )
            if kwargs.get("enable_thinking"):
                raise StructuredDecisionError(
                    "a read is the reply's first token, so thinking must be off"
                )
            if not self.specs:
                raise StructuredDecisionError(_NO_DECISION_ADAPTER)
            name = kwargs.get("adapter_name") or self.default_adapter
            if name not in self.specs:
                raise StructuredDecisionError(
                    f"adapter_name {name!r} is not a decision adapter of this "
                    f"checkpoint; decision adapters: {sorted(self.specs)}"
                )
            return name

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
            if media and any(media.values()):
                raise StructuredDecisionError(
                    "Granite Switch decision adapters read text; this request "
                    f"carries {sorted(k for k, v in media.items() if v)}"
                )
            name = self._select_adapter(chat_template_kwargs)
            spec, pool, tail = (
                self.specs[name],
                self.label_pools[name],
                self.tails[name],
            )
            labels = [label for label, _ in pool]
            read_request = self._read_request(
                {"adapter_name": name, "enable_thinking": False}, cache_salt
            )

            slots, engine_inputs, params = [], [], []
            for q in questions:
                try:
                    messages, order = render_question(
                        spec,
                        qtype=q.type.name,
                        state=state,
                        instructions=text_of(q.instructions),
                        options=[(o.name, o.description) for o in q.options],
                        labels=labels,
                        specials=self.specials,
                        request_instructions=text_of(instructions) or None,
                    )
                except DecisionSpecError as e:
                    raise StructuredDecisionError(f"question {q.id!r}: {e}") from e
                engine_input, prompt_ids = await self._render(read_request, messages)
                # The startup probe rendered through the tokenizer; this is what vLLM
                # actually renders (content format, chat options), so check it too.
                activations = prompt_ids.count(self.control_ids[name])
                if activations != 1:
                    raise StructuredDecisionError(
                        f"question {q.id!r} rendered with {activations} <|{name}|> "
                        f"control tokens, not one; the read would not be the adapter's"
                    )
                if prompt_ids[len(prompt_ids) - len(tail) :] != tail:
                    raise StructuredDecisionError(
                        "these chat options end the prompt differently, so the "
                        "labels' tokens are unknown"
                    )
                # groups[i] answers caller option i; label j sits at rendered slot j.
                groups: list[list[int]] = [[] for _ in q.options]
                for slot, option in enumerate(order):
                    groups[option] = [pool[slot][1]]
                slots.append(groups)
                engine_inputs.append(self._read_input(engine_input, prompt_ids))
                params.append(
                    self._sampling_params([i for g in groups for i in g], prompt_ids)
                )

            label_reads = await next_token_label_reads(
                self.context.engine_client,
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
                        and any(output.token_ids[0] in g for g in groups),
                        input_tokens=len(label_read.result.prompt_token_ids or ()),
                        output_tokens=len(output.token_ids),
                    )
                )
            return reads

    return GraniteSwitchDecisionStrategy


def _vllm_version() -> tuple[int, ...] | None:
    try:
        import vllm
    except ImportError:
        return None
    parts = []
    for part in str(getattr(vllm, "__version__", "")).split(".")[:2]:
        if not part.isdigit():
            return None
        parts.append(int(part))
    return tuple(parts)


def register_structured_decisions() -> None:
    """Insert the decision strategy into vLLM's ``READ_STRATEGIES``.

    Called by vLLM's plugin system at startup (``vllm.general_plugins``); re-entrant.
    A no-op on vLLM lines before the structured-decisions API (< 0.32). On a line that
    moved it (>= 0.33) it warns: the plugin then serves no decisions.
    """
    try:
        from vllm.entrypoints.generate.structured_decisions.strategies import (
            READ_STRATEGIES,
        )
    except ImportError:
        version = _vllm_version()
        if version is not None and version >= _MOVED_IN:
            logger.warning(
                "vLLM %s has no %s; the Granite Switch decision strategy is NOT "
                "registered and /v1/systemone will not serve decision adapters",
                ".".join(map(str, version)),
                _STRATEGIES,
            )
        else:
            logger.debug("vLLM has no structured-decisions API; nothing to register")
        return

    strategy = _build_strategy_class()
    for arch in VLLM_ARCHITECTURES:
        READ_STRATEGIES[arch] = strategy
    logger.info(
        "Granite Switch decision strategy registered for %s",
        ", ".join(VLLM_ARCHITECTURES),
    )
