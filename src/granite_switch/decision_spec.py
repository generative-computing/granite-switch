# SPDX-License-Identifier: Apache-2.0
"""How a decision adapter is asked a question: the ``decision:`` block of its io.yaml.

An adapter trained as a single-token decisioner declares, in its io.yaml, the prompt
layout it was trained on and how its options are labelled. Granite Switch keeps none
of that in code: the composer validates the block and writes the resolved form into
``config.json`` (``decision_specs``), and the vLLM read strategy renders from it.
Changing the marker or the layout is an io.yaml change and a re-compose (or a
re-stamp, ``granite_switch.composer.decision_config``), not a code change.

Pure Python (no torch / vLLM / yaml), so the composer, the config and the server
share one validator and one renderer.
"""

from __future__ import annotations

import copy
import json
import string
from typing import Any

#: U+2060 WORD JOINER: invisible, and splits a special-token string so the tokenizer
#: reads it as text rather than as the special token.
TOKEN_BREAK = "⁠"

QUESTION_TYPES = ("choice", "noul", "score")
#: vLLM's option names for a noul question, in its order.
NOUL_OPTIONS = ("yes", "no")

#: Where the adapter's control token lands, which decides how activation is checked.
#: ``marker``: before an invocation text the prompt carries in a user turn (aLoRA).
#: ``start``: at the sequence start (LoRA). ``generation``: in the generation prompt
#: (Shadow Residual, or an aLoRA whose invocation is the assistant header).
ACTIVATIONS = ("marker", "start", "generation")

_MESSAGE_FIELDS = frozenset({"state", "instructions", "options", "marker"})
_OPTION_LINE_FIELDS = frozenset({"label", "text"})
_ROLES = ("user", "assistant")
_REQUEST_INSTRUCTIONS = ("reject", "system")

#: Keys a decision block may set, with their defaults (``None`` = required).
_DEFAULTS: dict[str, Any] = {
    "activation": "unset",
    "invocation_marker": "",
    "system_prompt": None,
    "messages": None,
    "option_line": "{label}: {text}",
    "option_separator": "\n",
    "empty_state": "(empty)",
    "labels": "letters",
    "request_instructions": "reject",
    "question_types": None,
}
_TYPE_KEYS = frozenset({"order", "fallback_text"})
_YAML_BOOL_HINT = (
    "YAML reads unquoted yes/no as booleans; quote them, e.g. order: ['no', 'yes']"
)


class DecisionSpecError(ValueError):
    """An invalid decision block, or a question the spec cannot render."""


def _fields(template: str, where: str) -> list[str]:
    """Placeholder names in ``template``; plain ``{name}`` only (no ``.attr``,
    ``[key]``, conversion or format spec), so a template cannot reach into a value."""
    try:
        parsed = list(string.Formatter().parse(template))
    except ValueError as e:
        raise DecisionSpecError(f"{where}: {e}") from e
    names = []
    for _, name, spec, conversion in parsed:
        if name is None:
            continue
        if not name.isidentifier() or spec or conversion:
            raise DecisionSpecError(
                f"{where}: placeholder {{{name}{'!' + conversion if conversion else ''}"
                f"{':' + spec if spec else ''}}} must be a plain {{name}}"
            )
        names.append(name)
    return names


def _check_text(value: Any, key: str, where: str, *, allow_none=False) -> None:
    if value is None and allow_none:
        return
    if not isinstance(value, str):
        raise DecisionSpecError(f"{where}: {key} must be a string, got {value!r}")


def _check_messages(messages: Any, where: str) -> dict[str, int]:
    """Placeholder counts of valid ``messages``; the marker only in a user turn."""
    if not isinstance(messages, list) or not messages:
        raise DecisionSpecError(f"{where}: messages must be a non-empty list")
    counts = dict.fromkeys(_MESSAGE_FIELDS, 0)
    for i, m in enumerate(messages):
        if not isinstance(m, dict) or set(m) != {"role", "content"}:
            raise DecisionSpecError(
                f"{where}: messages[{i}] must be a mapping with exactly role and content"
            )
        if m["role"] not in _ROLES:
            raise DecisionSpecError(
                f"{where}: messages[{i}].role must be one of {_ROLES} (use "
                f"system_prompt for a system turn)"
            )
        _check_text(m["content"], f"messages[{i}].content", where)
        for name in _fields(m["content"], f"{where}: messages[{i}].content"):
            if name not in _MESSAGE_FIELDS:
                raise DecisionSpecError(
                    f"{where}: messages[{i}] uses unknown placeholder {{{name}}}; "
                    f"known: {sorted(_MESSAGE_FIELDS)}"
                )
            if name == "marker" and m["role"] != "user":
                raise DecisionSpecError(f"{where}: {{marker}} must be in a user turn")
            counts[name] += 1
    if messages[-1]["role"] != "user":
        raise DecisionSpecError(
            f"{where}: the last message must be a user turn (the reply is read next)"
        )
    if counts["options"] != 1:
        raise DecisionSpecError(f"{where}: messages must use {{options}} exactly once")
    if counts["marker"] > 1:
        raise DecisionSpecError(f"{where}: {{marker}} may appear at most once")
    return counts


def _check_question_types(types: Any, where: str) -> dict[str, dict]:
    if not isinstance(types, dict) or not types:
        raise DecisionSpecError(
            f"{where}: question_types must map at least one of {QUESTION_TYPES}"
        )
    out = {}
    for qtype, settings in types.items():
        if qtype not in QUESTION_TYPES:
            raise DecisionSpecError(
                f"{where}: unknown question type {qtype!r}; known: {QUESTION_TYPES}"
            )
        settings = {} if settings is None else settings
        if not isinstance(settings, dict) or set(settings) - _TYPE_KEYS:
            raise DecisionSpecError(
                f"{where}: question_types.{qtype} takes only {sorted(_TYPE_KEYS)}"
            )
        order = settings.get("order")
        if order is not None:
            if qtype != "noul":
                raise DecisionSpecError(
                    f"{where}: only noul takes an order (choice and score options "
                    f"are the caller's, in the caller's order)"
                )
            if isinstance(order, list) and any(isinstance(x, bool) for x in order):
                raise DecisionSpecError(
                    f"{where}: question_types.noul.order: {_YAML_BOOL_HINT}"
                )
            if (
                not isinstance(order, list)
                or not all(isinstance(x, str) for x in order)
                or sorted(order) != sorted(NOUL_OPTIONS)
            ):
                raise DecisionSpecError(
                    f"{where}: question_types.noul.order must be a list permuting "
                    f"{list(NOUL_OPTIONS)}"
                )
        fallback = settings.get("fallback_text") or {}
        if not isinstance(fallback, dict):
            raise DecisionSpecError(
                f"{where}: question_types.{qtype}.fallback_text must be a mapping"
            )
        if any(isinstance(k, bool) for k in fallback):
            raise DecisionSpecError(
                f"{where}: question_types.{qtype}.fallback_text: {_YAML_BOOL_HINT}"
            )
        if not all(
            isinstance(k, str) and isinstance(v, str) for k, v in fallback.items()
        ):
            raise DecisionSpecError(
                f"{where}: question_types.{qtype}.fallback_text must map option "
                f"names to text"
            )
        if qtype == "noul" and set(fallback) - set(NOUL_OPTIONS):
            raise DecisionSpecError(
                f"{where}: question_types.noul.fallback_text keys must be among "
                f"{list(NOUL_OPTIONS)}"
            )
        out[qtype] = {"order": order, "fallback_text": fallback}
    return out


def normalize_decision_block(
    block: Any,
    *,
    adapter_name: str,
    activation: str | None = None,
    anchor_text: str | None = None,
) -> dict[str, Any]:
    """Validate a ``decision:`` block and return it with every key filled in.

    ``activation`` / ``anchor_text`` are what the composer knows from the adapter
    itself (its technology and ``adapter_config.json``). They win over the block:
    a block that disagrees is refused, because the chat template activates the
    adapter on the anchor, and a mismatch would silently read the base model.
    ``None`` (the config and the server re-validating a resolved spec) keeps the
    block's own values. Idempotent on its own output.
    """
    where = f"decision adapter {adapter_name!r}"
    if not isinstance(block, dict):
        raise DecisionSpecError(f"{where}: the decision block must be a mapping")
    unknown = sorted(set(block) - set(_DEFAULTS))
    if unknown:
        raise DecisionSpecError(
            f"{where}: unknown decision keys {unknown}; known: {sorted(_DEFAULTS)}"
        )
    spec = {k: copy.deepcopy(block.get(k, v)) for k, v in _DEFAULTS.items()}
    for key in ("messages", "question_types"):
        if spec[key] is None:
            raise DecisionSpecError(f"{where}: the decision block needs {key!r}")

    declared = spec["activation"]
    if declared == "unset":
        declared = None
    if declared is not None and declared not in ACTIVATIONS:
        raise DecisionSpecError(f"{where}: activation must be one of {ACTIVATIONS}")
    if activation is not None and declared is not None and declared != activation:
        raise DecisionSpecError(
            f"{where}: activation {declared!r} differs from the adapter's "
            f"{activation!r} placement"
        )
    activation = activation or declared
    if activation is None:
        raise DecisionSpecError(
            f"{where}: activation is unknown; the composer derives it, a stamp needs "
            f"the adapter's technology"
        )
    spec["activation"] = activation

    _check_text(spec["invocation_marker"], "invocation_marker", where, allow_none=True)
    marker = spec["invocation_marker"] or ""
    if anchor_text is not None and activation == "marker":
        if marker and marker != anchor_text:
            raise DecisionSpecError(
                f"{where}: invocation_marker {marker!r} differs from the adapter's "
                f"aLoRA invocation text {anchor_text!r}; the chat template activates "
                f"the adapter on the latter"
            )
        marker = anchor_text
    spec["invocation_marker"] = marker

    for key in ("option_line", "option_separator", "empty_state"):
        _check_text(spec[key], key, where)
    _check_text(spec["system_prompt"], "system_prompt", where, allow_none=True)
    line_fields = _fields(spec["option_line"], f"{where}: option_line")
    if sorted(line_fields) != sorted(_OPTION_LINE_FIELDS):
        raise DecisionSpecError(
            f"{where}: option_line must use {{label}} and {{text}} once each, "
            f"got {line_fields}"
        )
    if spec["request_instructions"] not in _REQUEST_INSTRUCTIONS:
        raise DecisionSpecError(
            f"{where}: request_instructions must be one of {_REQUEST_INSTRUCTIONS}"
        )

    counts = _check_messages(spec["messages"], where)
    if activation == "marker" and (counts["marker"] != 1 or not marker):
        raise DecisionSpecError(
            f"{where}: a marker-activated adapter needs an invocation_marker and "
            f"messages that place {{marker}} once"
        )
    if counts["marker"] and not marker:
        raise DecisionSpecError(
            f"{where}: messages use {{marker}} but no invocation_marker is known"
        )

    labels = spec["labels"]
    if labels != "letters":
        if (
            not isinstance(labels, list)
            or len(labels) < 2
            or not all(isinstance(x, str) and x for x in labels)
            or len(set(labels)) != len(labels)
        ):
            raise DecisionSpecError(
                f"{where}: labels must be 'letters' or a list of at least two "
                f"distinct non-empty strings"
            )

    spec["question_types"] = _check_question_types(spec["question_types"], where)
    return spec


def validate_decision_fields(
    adapter_names: list[str] | None,
    default: str | None,
    specs: dict[str, Any] | None,
    adapter_kinds: list[str] | None = None,
) -> tuple[str | None, dict[str, dict]]:
    """Check a checkpoint's ``decision_adapter`` / ``decision_specs`` and normalize
    them. Shared by the config (compose, load) and the server (``--hf-overrides``
    bypasses the config's own validation)."""
    names = list(adapter_names or [])
    specs = specs or {}
    if not specs:
        if default is not None:
            raise DecisionSpecError(
                f"decision_adapter {default!r} is set but decision_specs is empty; "
                f"compose with --decision-adapter so its io.yaml spec is recorded"
            )
        return None, {}
    kinds = dict(zip(names, adapter_kinds or []))
    for name in specs:
        if name not in names:
            raise DecisionSpecError(
                f"decision_specs names {name!r}, which is not an adapter of this "
                f"checkpoint {names}"
            )
        if kinds.get(name, "lora") != "lora":
            raise DecisionSpecError(
                f"decision adapter {name!r} is a {kinds[name]} slot; only LoRA, aLoRA "
                f"and Shadow Residual adapters answer decisions"
            )
    normalized = {
        name: normalize_decision_block(spec, adapter_name=name)
        for name, spec in specs.items()
    }
    if default not in normalized:
        raise DecisionSpecError(
            f"decision_adapter {default!r} must name one of {sorted(normalized)}"
        )
    return default, normalized


def text_of(x: object) -> str:
    """A state, instruction or description as text: dicts and lists as one-line JSON."""
    if isinstance(x, (dict, list)):
        return json.dumps(x, ensure_ascii=False)
    return "" if x is None else str(x)


def escape_special_tokens(text: str, specials: list[str]) -> str:
    """Break every occurrence of ``specials`` in caller text, so state, instructions
    and descriptions cannot forge turns, control tokens or a second activation.
    Repeated until none is left: a self-overlapping marker (``----`` for ``---``)
    survives one pass."""
    for special in specials:
        while special and special in text:
            text = text.replace(special, special[0] + TOKEN_BREAK + special[1:])
    return text


def special_strings(tokenizer, specs: dict[str, dict]) -> list[str]:
    """What caller text must not contain: every added token, and every decision
    marker. Longest first, so a token that contains another is broken whole."""
    added = getattr(tokenizer, "get_added_vocab", None)
    found = set(added()) if added else set()
    found |= {s["invocation_marker"] for s in specs.values() if s["invocation_marker"]}
    return sorted(found, key=len, reverse=True)


def build_label_pool(tokenizer, labels: str | list[str]) -> list[tuple[str, int]]:
    """The spec's labels as ``(label, token_id)``, in order; each must be one token.

    ``letters`` is A..Z, the most vLLM 0.32 asks for (a choice is capped at 26
    options, a score at 10 levels).
    """
    labels = list(string.ascii_uppercase) if labels == "letters" else labels
    pool: list[tuple[str, int]] = []
    for label in labels:
        ids = tokenizer(label, add_special_tokens=False)["input_ids"]
        if len(ids) != 1 or ids[0] in {tid for _, tid in pool}:
            raise DecisionSpecError(
                f"label {label!r} is not one distinct token under this tokenizer "
                f"(ids {ids})"
            )
        pool.append((label, ids[0]))
    return pool


def render_question(
    spec: dict[str, Any],
    *,
    qtype: str,
    state: str,
    instructions: str,
    options: list[tuple[str, str | None]],
    labels: list[str],
    specials: list[str],
    request_instructions: str | None = None,
) -> tuple[list[dict[str, str]], list[int]]:
    """Messages for one question, and the rendered order of its options.

    ``options`` are ``(name, description)`` in the caller's order; ``order[j]`` is
    the caller index of the option rendered at position ``j``, which ``labels[j]``
    answers. All caller text (state, instructions, option text, request
    instructions) is escaped; the spec's own text is trusted.
    """
    settings = spec["question_types"].get(qtype)
    if settings is None:
        raise DecisionSpecError(
            f"this decision adapter does not answer {qtype!r} questions; it answers "
            f"{sorted(spec['question_types'])}"
        )
    if len(options) > len(labels):
        raise DecisionSpecError(
            f"{len(options)} options, but this decision adapter has only "
            f"{len(labels)} labels"
        )
    names = [name for name, _ in options]
    order = list(range(len(options)))
    if settings["order"] is not None:
        if sorted(names) != sorted(settings["order"]):
            raise DecisionSpecError(
                f"{qtype} options {names} do not match the spec's order "
                f"{settings['order']}"
            )
        order = [names.index(n) for n in settings["order"]]

    def esc(text: str) -> str:
        return escape_special_tokens(text, specials)

    fallback = settings["fallback_text"]
    lines = []
    for label, i in zip(labels, order):
        name, description = options[i]
        text = description if description else fallback.get(name, name)
        lines.append(spec["option_line"].format(label=label, text=esc(text)))
    values = {
        "state": esc(state) if state else spec["empty_state"],
        "instructions": esc(instructions),
        "options": spec["option_separator"].join(lines),
        "marker": spec["invocation_marker"],
    }
    messages = [
        {"role": m["role"], "content": m["content"].format(**values)}
        for m in spec["messages"]
    ]

    system = [spec["system_prompt"]] if spec["system_prompt"] else []
    if request_instructions:
        if spec["request_instructions"] == "reject":
            raise DecisionSpecError(
                "request-level instructions are not supported by this decision "
                "adapter; put the context in the state"
            )
        system.append(esc(request_instructions))
    if system:
        messages.insert(0, {"role": "system", "content": "\n\n".join(system)})
    return messages, order


def control_token_id(tokenizer, adapter_name: str) -> int:
    control = f"<|{adapter_name}|>"
    control_id = tokenizer.convert_tokens_to_ids(control)
    if control_id in (None, getattr(tokenizer, "unk_token_id", None)):
        raise DecisionSpecError(f"the tokenizer has no {control} control token")
    return control_id


def check_activation(
    tokenizer,
    adapter_name: str,
    spec: dict[str, Any],
    *,
    chat_template: str | None = None,
    template_kwargs: dict[str, Any] | None = None,
) -> list[int]:
    """Refuse a spec whose rendered question does not activate its adapter.

    Renders a probe question through ``spec`` and the chat template with
    ``adapter_name`` set, and requires exactly one ``<|adapter_name|>`` control token.
    A marker-activated spec must also place it inside the conversation: with a wrong
    marker the template falls back to putting the token just before the assistant
    header, and the question would read the base model's guess. Returns the probe's
    prompt ids (with the generation prompt), whose tail every real read shares.
    """
    control_id = control_token_id(tokenizer, adapter_name)
    qtype = next(iter(spec["question_types"]))
    names = list(NOUL_OPTIONS) if qtype == "noul" else ["first", "second"]
    labels = [label for label, _ in build_label_pool(tokenizer, spec["labels"])]
    messages, _ = render_question(
        spec,
        qtype=qtype,
        state="probe state",
        instructions="Is this a probe?",
        options=[(n, None) for n in names],
        labels=labels,
        specials=special_strings(tokenizer, {adapter_name: spec}),
    )
    kwargs = {
        **(template_kwargs or {}),
        "adapter_name": adapter_name,
        "enable_thinking": False,
    }
    if chat_template is not None:
        kwargs["chat_template"] = chat_template

    def ids(add_generation_prompt: bool) -> list[int]:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    full = ids(True)
    if full.count(control_id) != 1:
        raise DecisionSpecError(
            f"a probe question for {adapter_name!r} carries {full.count(control_id)} "
            f"<|{adapter_name}|> tokens, not one: check the chat template's adapter_map"
        )
    # [:-1]: a fallback placement can land as the very last token of a render
    # without the generation prompt, which is still outside the question turn.
    if spec["activation"] == "marker" and control_id not in ids(False)[:-1]:
        raise DecisionSpecError(
            f"a probe question for {adapter_name!r} does not carry <|{adapter_name}|> "
            f"inside the question turn: its marker {spec['invocation_marker']!r} does "
            f"not match the chat template's invocation text"
        )
    return full
