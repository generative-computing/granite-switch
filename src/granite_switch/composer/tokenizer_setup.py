# SPDX-License-Identifier: Apache-2.0
"""Tokenizer configuration for adapter control tokens and chat templates.

Extracted from ``compose_granite_switch.py`` to provide testable units for
token management and chat template modification.
"""

import os
import re
from dataclasses import dataclass

from .adapter_loader import load_adapter_config

# Name of the optional return-to-base control token (``<|base_reset|>``). It is
# NOT an adapter: it occupies ``adapter_token_ids[0]`` and writes expert id 0, so
# a request can route back to the base model mid-sequence. See
# ``add_control_tokens(base_reset=True)``.
BASE_RESET_ADAPTER_NAME = "base_reset"

# ---------------------------------------------------------------------------
# Chat-template format abstraction
# ---------------------------------------------------------------------------
#
# Granite ships two chat-template families that both need adapter control-token
# injection but differ in their role markers and Jinja structure:
#
#   * "granite_format" (Granite 3.x / 4.0 / 4.1): role markers
#     ``<|start_of_role|>ROLE<|end_of_role|>`` ... ``<|end_of_text|>``. Per-message
#     content is built into a ``content = namespace(val=...)`` object. The
#     user/system dispatch branch is ``{%- if (message.role == 'user') or ...``.
#
#   * "chatml" (Granite 4.2): ChatML-style ``<|im_start|>ROLE\n`` ... ``<|im_end|>``
#     markers with a ``<think>`` reasoning block on the assistant/generation
#     turn. Per-message content is a plain string ``content``. The user/system
#     dispatch branch is ``{%- elif message.role == "user" or ...``. The main
#     loop iterates ``loop_messages`` (system message stripped) rather than
#     ``messages``, and the ``ns`` namespace is redefined after the system split
#     — the adapter vars must merge into that *last* redefinition.
#
# ``detect_template_format`` inspects the template string and returns the
# matching ``TemplateFormat``; ``configure_chat_template`` drives all injection
# from these parameters instead of hardcoding the granite_format shapes.


@dataclass(frozen=True)
class TemplateFormat:
    """Format-specific parameters that drive adapter control-token injection.

    Attributes:
        name: ``"granite_format"`` or ``"chatml"``.
        role_open_marker: The literal that opens a role turn.
        content_accessor: The Jinja variable Pass 2 reads/writes to inject the
            ALoRA control token into the target user message
            (``"content.val"`` for granite_format, ``"content"`` for ChatML).
        loop_var_source: The iterable the main message loop walks over
            (``"messages"`` for granite_format, ``"loop_messages"`` for ChatML). Used to
            anchor Pass 1 insertion just before the loop.
        skip_once_re_standalone: Regex matching a standalone role-open literal at
            the start of a ``{{- '<marker>' + expr + ... }}`` emission (Case A).
        skip_once_re_merged: Regex matching a merged role literal
            ``{{- '<marker>ROLE...' (+ expr)? }}`` (Case B).
        user_role_anchor_re: Regex matching the user/system dispatch branch,
            used to anchor Pass 2 insertion inside that branch.
        ns_merge_last: When True, merge adapter vars into the *last*
            ``{%- set ns = namespace(...) %}`` (ChatML redefines ns after the
            system split); when False, the first (granite_format).
    """

    name: str
    role_open_marker: str
    content_accessor: str
    loop_var_source: str
    skip_once_re_standalone: str
    skip_once_re_merged: str
    user_role_anchor_re: str
    ns_merge_last: bool


_GRANITE_FORMAT = TemplateFormat(
    name="granite_format",
    role_open_marker="<|start_of_role|>",
    content_accessor="content.val",
    loop_var_source="messages",
    # Case A: {{- '<|start_of_role|>' + expr + ... }} (single-quoted only).
    skip_once_re_standalone=r"\{\{-\s*'<\|start_of_role\|>'\s*\+\s*",
    # Case B: {{- '<|start_of_role|>ROLE<|end_of_role|>' (+ expr)? }}.
    skip_once_re_merged=r"\{\{-\s*'<\|start_of_role\|>([^']*)'((?:\s*\+\s*[^}]+?)?)\s*\}\}",
    user_role_anchor_re=r"(\{%- if \(message\.role == 'user'\) or)",
    ns_merge_last=False,
)

_CHATML_FORMAT = TemplateFormat(
    name="chatml",
    role_open_marker="<|im_start|>",
    content_accessor="content",
    loop_var_source="loop_messages",
    # Case A: standalone concat with the marker in single quotes:
    #   {{- '<|im_start|>' + message.role + '\n' }}
    skip_once_re_standalone=r"\{\{-\s*'<\|im_start\|>'\s*\+\s*",
    # Case B: merged literal in EITHER single or double quotes, with optional
    # trailing concatenation:
    #   "<|im_start|>system\n"   /   '<|im_start|>assistant\n'   /   '<|im_start|>user\n'
    # The quote char is captured (group 1) and required to match at the close.
    skip_once_re_merged=(
        r"\{\{-\s*(['\"])<\|im_start\|>((?:(?!\1).)*)\1((?:\s*~\s*[^}]+?|\s*\+\s*[^}]+?)?)\s*\}\}"
    ),
    user_role_anchor_re=r"(\{%- elif message\.role == \"user\" or)",
    ns_merge_last=True,
)


def detect_template_format(chat_template: str | None) -> TemplateFormat | None:
    """Detect which Granite chat-template family *chat_template* belongs to.

    Detection is by role-marker presence:

    * ``<|im_start|>`` present  → ChatML (Granite 4.2).
    * ``<|start_of_role|>`` present → Granite role-marker format (3.x / 4.0 / 4.1).
    * neither → ``None`` (caller preserves the template verbatim and warns).

    ChatML is checked first: a template that somehow contained both markers is
    treated as ChatML (the newer format).

    Args:
        chat_template: The tokenizer's chat template string, or ``None``.

    Returns:
        The matching :class:`TemplateFormat`, or ``None`` when the template is
        missing or uses an unrecognized marker set.
    """
    if not chat_template:
        return None
    if "<|im_start|>" in chat_template:
        return _CHATML_FORMAT
    if "<|start_of_role|>" in chat_template:
        return _GRANITE_FORMAT
    return None


#: Placement mode where the control token is inserted immediately *before* an
#: invocation sequence (legacy ALoRA, ``alora_invocation_tokens``).
ANCHOR_MODE_ALORA = "alora"

#: Placement mode where the control token *replaces* a single anchor token at
#: the end of the generation prompt (Shadow Residual, ``last_context_token``).
ANCHOR_MODE_SR = "sr"


def _load_alora_invocation_token_ids(adapter_path: str) -> list[int]:
    """Load alora_invocation_tokens from adapter_config.json.

    Raises:
        FileNotFoundError: If adapter_config.json is not found at adapter_path.
        ValueError: If alora_invocation_tokens is missing or empty.
    """
    config_path = os.path.join(adapter_path, "adapter_config.json")
    adapter_config = load_adapter_config(adapter_path)

    token_ids = adapter_config.get("alora_invocation_tokens")
    if not token_ids:
        raise ValueError(
            f"alora_invocation_tokens is missing or empty in {config_path}"
        )
    return token_ids


def resolve_activation_anchor(adapter_path: str, tokenizer) -> tuple[str, str]:
    """Resolve where an adapter's control token goes, from its adapter_config.json.

    Two checkpoint generations are supported, distinguished by which keys the
    trainer wrote:

    * ``alora_invocation_tokens`` (a token-id *sequence*) — the control token is
      inserted immediately **before** that sequence.  Mode
      :data:`ANCHOR_MODE_ALORA`.
    * ``last_context_token`` + ``last_context_token_id`` (a **single** token) —
      written by shadow-residual ``feature/stock-peft-sr`` onward, where the
      adapter is always-active during training and this marker records the
      prompt/completion boundary.  The control token **replaces** that token at
      the end of the generation prompt.  Mode :data:`ANCHOR_MODE_SR`.

    The legacy key is probed first, so an adapter carrying both keys keeps its
    established placement.

    Args:
        adapter_path: Directory holding ``adapter_config.json``.
        tokenizer: Tokenizer used to decode / re-encode the anchor.

    Returns:
        ``(anchor_text, mode)``, where ``anchor_text`` is the literal the chat
        template renders at the activation point.

    Raises:
        FileNotFoundError: If ``adapter_config.json`` is not found.
        ValueError: If neither key set is present, if ``last_context_token`` is
            not a single token under *tokenizer*, or if it does not encode to
            the recorded ``last_context_token_id``.
    """
    try:
        invocation_text = _decode_alora_invocation_text(adapter_path, tokenizer)
    except ValueError:
        pass  # No alora_invocation_tokens — try the Shadow Residual keys below.
    else:
        return invocation_text, ANCHOR_MODE_ALORA

    config_path = os.path.join(adapter_path, "adapter_config.json")
    anchor_text = load_adapter_config(adapter_path).get("last_context_token")
    if not anchor_text:
        raise ValueError(
            f"{config_path} carries no activation anchor: expected either "
            f"'alora_invocation_tokens' (shadow-residual before "
            f"feature/stock-peft-sr, and the standard aLoRA adapter library) or "
            f"'last_context_token' + 'last_context_token_id' (shadow-residual "
            f"feature/stock-peft-sr onward)."
        )
    _resolve_sr_anchor_token_id(adapter_path, tokenizer, anchor_text)
    return anchor_text, ANCHOR_MODE_SR


def _resolve_sr_anchor_token_id(adapter_path: str, tokenizer, anchor_text: str) -> int:
    """Encode *anchor_text* and check it against the recorded id.

    The trainer already asserts a single id (``_resolve_single_token`` in
    ``shadow_residual/training/train.py``); it is re-checked here against *this*
    tokenizer because the substitute-token mechanism swaps exactly one
    embedding, and a train-vs-compose tokenizer mismatch would otherwise
    activate the adapter at the wrong position and silently lose accuracy.
    """
    config_path = os.path.join(adapter_path, "adapter_config.json")
    encoded = tokenizer.encode(anchor_text, add_special_tokens=False)
    if len(encoded) != 1:
        raise ValueError(
            f"last_context_token {anchor_text!r} from {config_path} encodes to "
            f"{len(encoded)} tokens ({encoded}) with this tokenizer, but the "
            f"control-token swap replaces exactly one embedding. The adapter was "
            f"trained against a different tokenizer than the base model being "
            f"composed."
        )
    recorded_id = load_adapter_config(adapter_path).get("last_context_token_id")
    if recorded_id is not None and recorded_id != encoded[0]:
        raise ValueError(
            f"last_context_token {anchor_text!r} encodes to id {encoded[0]} with "
            f"this tokenizer, but {config_path} records "
            f"last_context_token_id={recorded_id}. The adapter was trained "
            f"against a different tokenizer than the base model being composed; "
            f"composing anyway would activate the adapter at the wrong position "
            f"and silently degrade accuracy."
        )
    return encoded[0]


def load_activation_anchor(adapter_path: str, tokenizer) -> tuple[str, int, str]:
    """Like :func:`resolve_activation_anchor`, but also return the anchor's id.

    Returns:
        ``(anchor_text, anchor_token_id, mode)``.  ``anchor_token_id`` is the id
        whose embedding the runtime token-exchange must place at the control
        token's position so the post-swap sequence is indistinguishable from a
        no-adapter render.
    """
    anchor_text, mode = resolve_activation_anchor(adapter_path, tokenizer)
    if mode == ANCHOR_MODE_SR:
        return (
            anchor_text,
            _resolve_sr_anchor_token_id(adapter_path, tokenizer, anchor_text),
            mode,
        )
    return anchor_text, get_alora_first_invocation_token_id(adapter_path), mode


def _decode_alora_invocation_text(adapter_path: str, tokenizer) -> str:
    """Decode alora_invocation_tokens from adapter_config.json to a string.

    The activation control token must be inserted immediately before the first
    token of the invocation sequence. Decoding the full sequence gives the text
    span to search for in the rendered message content.
    """
    token_ids = _load_alora_invocation_token_ids(adapter_path)
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def get_alora_first_invocation_token_id(adapter_path: str) -> int:
    """Return the first token ID of an ALoRA adapter's invocation sequence.

    Used by token-exchange mode to substitute this embedding for the adapter's
    control token before the decoder runs.
    """
    return _load_alora_invocation_token_ids(adapter_path)[0]


#: A ``{{- ... }}`` emission in a Jinja template.
_EMISSION_RE = re.compile(r"\{\{-?\s*(.*?)\s*-?\}\}", re.DOTALL)

#: An emission expression that is a single quoted string literal and nothing
#: else. Group 1 is the quote character, group 2 the literal body.
_SINGLE_LITERAL_RE = re.compile(r"^(['\"])((?:(?!\1).)*)\1$", re.DOTALL)


def _split_at_generation_prompt(template: str) -> tuple[str, str]:
    """Split *template* into everything before, and everything from, the
    ``{%- if add_generation_prompt %}`` block.

    Raises:
        ValueError: If the template has no such block.
    """
    match = re.search(r"\{%-\s*if add_generation_prompt\s*%\}", template)
    if match is None:
        raise ValueError(
            "Chat template has no 'add_generation_prompt' block, so a Shadow "
            "Residual control token cannot be placed at the end of the "
            "generation prompt."
        )
    return template[: match.start()], template[match.start() :]


#: Escape sequences recognized inside a Jinja string literal.
_JINJA_ESCAPES = {
    "\\": "\\",
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "'": "'",
    '"': '"',
}


def _unescape_literal(body: str) -> str:
    """Resolve a Jinja string-literal *body* to the text it renders.

    Token-level reasoning has to happen on the rendered value: in the template
    source Granite 4.2's generation prompt is
    ``<|im_start|>assistant\\n<think>`` with a two-character ``\\n``, and asking
    a tokenizer about that is meaningless.
    """
    out, i = [], 0
    while i < len(body):
        if body[i] == "\\" and i + 1 < len(body) and body[i + 1] in _JINJA_ESCAPES:
            out.append(_JINJA_ESCAPES[body[i + 1]])
            i += 2
        else:
            out.append(body[i])
            i += 1
    return "".join(out)


def _generation_prompt_literals(region: str) -> list[str]:
    """Bodies of every lone-string-literal emission in the generation prompt.

    One entry per emission, so a template that renders different generation
    prompts under different flags (Granite 4.2's ``enable_thinking``) yields one
    entry per branch.  This is what makes branch coverage checkable: a control
    token placed in only one branch leaves the others inert.
    """
    bodies = []
    for emission in _EMISSION_RE.finditer(region):
        literal = _SINGLE_LITERAL_RE.match(emission.group(1))
        if literal is not None:
            bodies.append(literal.group(2))
    return bodies


def _resolve_sr_substitution_site(region: str, anchor: str, tokenizer) -> str:
    """Pick the token the Shadow Residual control token will replace.

    The adapter declares its ``last_context_token`` (*anchor*), and when every
    generation-prompt branch emits it that is the site — Granite 4.1, where the
    single branch ends ``'<|start_of_role|>assistant<|end_of_role|>'``.

    Granite 4.2 has two branches that do **not** share the anchor::

        enable_thinking      '<|im_start|>assistant\\n<think>\\n'
        not enable_thinking  '<|im_start|>assistant\\n<think></think>'

    The declared anchor ``</think>`` exists only in the second, so substituting
    it leaves the *default* render with no control token at all and ships an
    adapter that never activates.  A single control token cannot carry two
    substitute embeddings either — the runtime swap keys on the control token's
    id — so the site must be one token common to every branch.  Here that is
    ``<think>``, the last token of the branches' common prefix.

    Moving the site earlier is sound: SR trains always-active (its
    ``last_context_token`` is a label-boundary marker, not a gate), and both
    sites sit inside the generation prompt with nothing generated in between.

    Returns:
        The literal to substitute, guaranteed to occur in every branch and to
        encode to exactly one token id.

    Raises:
        ValueError: If the generation prompt emits no string literal at all, if
            its single branch does not emit *anchor*, or if its branches share
            no single-token site.
    """
    literals = _generation_prompt_literals(region)
    if not literals:
        raise ValueError(
            "The generation prompt emits no string literal, so the Shadow "
            "Residual control token has nowhere to go."
        )
    if all(anchor in body for body in literals):
        return anchor
    if len(literals) == 1:
        raise ValueError(
            f"The generation prompt never emits {anchor!r}, so the Shadow "
            f"Residual control token has nowhere to go. The adapter's "
            f"last_context_token does not match this base model's chat template."
        )

    prefix = os.path.commonprefix([_unescape_literal(body) for body in literals])
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    for cut in range(len(prefix)):
        candidate = prefix[cut:]
        # The site is located and re-emitted in the *escaped* Jinja source, so a
        # candidate containing anything the source escapes would not survive the
        # rewrite. This also keeps the site off the wrong side of a ``\n``.
        if any(char in _JINJA_ESCAPES.values() for char in candidate):
            continue
        candidate_ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(candidate_ids) != 1:
            continue
        lead = prefix[: len(prefix) - len(candidate)]
        lead_ids = tokenizer.encode(lead, add_special_tokens=False) if lead else []
        # The control token replaces one *token*, so the site has to start on a
        # token boundary. Without this a bare '>' would qualify — it encodes to
        # one id on its own while being the tail of a longer token in context.
        if prefix_ids == lead_ids + candidate_ids:
            return candidate

    raise ValueError(
        f"The generation prompt has {len(literals)} render paths that do not "
        f"all emit the adapter's last_context_token {anchor!r} ({literals!r}), "
        f"and their common prefix {prefix!r} ends in no single token that could "
        f"stand in for it. A Shadow Residual control token must replace one "
        f"token present in every path, otherwise the paths it misses render an "
        f"adapter that never activates."
    )


def _replace_anchor_in_generation_prompt(
    template: str, anchor: str, tokenizer
) -> tuple[str, str, int]:
    """Make every generation-prompt branch emit the control token in place of
    one token.

    The control token must take that token's *place*, not sit next to it: the
    runtime embedding swap puts the replaced token's embedding back at the
    control token's position, so the post-swap id sequence is identical to a
    no-adapter render and only the switch sees a difference.

    The rewrite is confined to the ``{%- if add_generation_prompt %}`` block so
    that the same literal appearing in earlier turns (every role header on 4.1;
    the history-truncation ``split('</think>')`` calls on 4.2) is untouched.

    Args:
        template: The chat template being patched.
        anchor: The adapter's declared ``last_context_token``.
        tokenizer: Used to check candidate sites are a single token.

    Returns:
        ``(patched_template, site, num_replacements)`` where *site* is the
        literal actually replaced — the anchor itself unless the branches forced
        an earlier common token (see :func:`_resolve_sr_substitution_site`).

    Raises:
        ValueError: If there is no ``add_generation_prompt`` block, if the site
            appears in an emission that is not a lone string literal, if any
            branch does not contain the site exactly once, or if the number of
            substitutions does not equal the number of branches.
    """
    head, region = _split_at_generation_prompt(template)
    site = _resolve_sr_substitution_site(region, anchor, tokenizer)

    literals = _generation_prompt_literals(region)
    for body in literals:
        # One substitution per branch, no more: two control tokens in one render
        # would activate twice and break the id-for-id equality with a
        # no-adapter render.
        if body.count(site) != 1:
            raise ValueError(
                f"The generation prompt branch {body!r} contains the Shadow "
                f"Residual substitution site {site!r} {body.count(site)} times, "
                f"but the control token must replace exactly one token."
            )

    count = 0

    def _rewrite(emission: "re.Match") -> str:
        nonlocal count
        expr = emission.group(1)
        if site not in expr:
            return emission.group(0)
        literal = _SINGLE_LITERAL_RE.match(expr)
        if literal is None:
            raise ValueError(
                f"The generation prompt emits {site!r} from an expression that "
                f"is not a lone string literal ({expr!r}). The Shadow Residual "
                f"control token cannot be substituted for it safely."
            )
        quote, body = literal.group(1), literal.group(2)
        # Emit the control token where the site was, falling back to the site
        # itself for every other adapter type (and for no-adapter renders).
        substitution = (
            "{%- if ns.adapter_token and ns.adapter_type == '"
            + ANCHOR_MODE_SR
            + "' %}{{- ns.adapter_token }}"
            "{%- else %}{{- " + quote + site + quote + " }}{%- endif %}"
        )
        parts = body.split(site)
        out = []
        for i, part in enumerate(parts):
            if part:
                out.append("{{- " + quote + part + quote + " }}")
            if i < len(parts) - 1:
                out.append(substitution)
                count += 1
        return "".join(out)

    region = _EMISSION_RE.sub(_rewrite, region)
    if count != len(literals):
        # Reached when a branch emits the site from something other than a lone
        # literal, or when the region holds a literal the site is absent from.
        # Either way some render path would ship an inert adapter, so fail here
        # rather than at eval time.
        raise ValueError(
            f"Substituted the Shadow Residual control token for {site!r} in "
            f"{count} of the generation prompt's {len(literals)} literal "
            f"emission(s) ({literals!r}). Every render path must place the "
            f"control token, otherwise the ones that miss it render an adapter "
            f"that never activates."
        )
    return head + region, site, count


def add_control_tokens(
    tokenizer,
    discovered_adapters: list[tuple[str | None, str, str, str | None]],
    base_reset: bool = False,
) -> tuple[list[int], list[str]]:
    """Add control tokens to the tokenizer for each adapter.

    Each adapter gets one control token: ``<|adapter_name|>`` which activates that adapter.

    Args:
        tokenizer: HuggingFace tokenizer.
        discovered_adapters: List of ``(adapter_path, adapter_name, technology, source)`` tuples.
        base_reset: When True, prepend the return-to-base control token
            ``<|base_reset|>``, yielding ``num_adapters + 1`` ids. MultiSwitch
            reads ``adapter_token_ids[0]`` as the base-reset slot (expert id 0)
            precisely when the list is one longer than ``num_adapters``, so the
            token MUST be first — appended at the end it would fire adapter 1.

    Returns:
        ``(adapter_token_ids, special_tokens)``

        adapter_token_ids has length ``num_adapters``, or ``num_adapters + 1``
        when ``base_reset`` is set (base-reset slot first).

    Raises:
        ValueError: if ``base_reset`` is set and an adapter is already named
            ``base_reset`` — both would resolve to one token id.
    """
    print(f"\nAdding control tokens for {len(discovered_adapters)} adapter(s)...")

    special_tokens = []
    if base_reset:
        clashing = [
            a[1] for a in discovered_adapters if a[1] == BASE_RESET_ADAPTER_NAME
        ]
        if clashing:
            raise ValueError(
                f"--base-reset-token needs the name {BASE_RESET_ADAPTER_NAME!r}, but an "
                f"adapter already uses it. add_special_tokens de-duplicates, so the "
                f"base-reset slot and that adapter would share one token id: the config's "
                f"uniqueness check would reject the checkpoint, and the token-exchange LUT "
                f"would keep only one of the two substitutes. Rename the adapter (e.g. via "
                f"its io.yaml) or compose without --base-reset-token."
            )
        special_tokens.append(f"<|{BASE_RESET_ADAPTER_NAME}|>")
    for adapter_info in discovered_adapters:
        adapter_name = adapter_info[1]
        special_tokens.append(f"<|{adapter_name}|>")

    print(f"  Tokens to add: {special_tokens}")
    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": special_tokens}
    )
    new_vocab_size = len(tokenizer)
    print(f"Added {num_added} special tokens")
    print(f"  New vocabulary size: {new_vocab_size}")

    # Get token IDs. Iterating ``special_tokens`` (not ``discovered_adapters``)
    # keeps id order identical to token order, so the base-reset slot stays at
    # index 0 by construction rather than by a second, parallel branch.
    print("\nToken ID mapping:")
    adapter_token_ids = []
    for token_name in special_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token_name)
        adapter_token_ids.append(token_id)
        print(f"  {token_name}: {token_id}")

    return adapter_token_ids, special_tokens


# Reserved slots in the Granite vocabulary that never appear as a training
# target. CONFIRMED with the Granite model authors: these ids are reserved and
# the model is trained not to emit them, so borrowing one of their output rows to
# suppress a newly added token is the intended use, not a trick.
#
# Independently measured on granite-4.1-3b and granite-4.2-3b before that
# confirmation, and the numbers are kept because they are a cheap way to re-check
# the property on a new base model: every <|unused_N|> row has collapsed to a
# single point — in 4.2's lm_head the 72 rows sit within 0.001 of each other, with
# a norm at the ~0.1st percentile of ordinary tokens. That is the signature of an
# id which only ever received downward pressure from the softmax denominator and
# never a target gradient. A future base whose unused rows look like ordinary
# tokens would not have this property, and this policy would need revisiting.
#
# The inventory is NOT stable across releases (4.1 has 69 of these, 4.2 has 72),
# so nothing may depend on a particular count or id range — hence the lookup in
# find_reserved_never_emitted_token_id rather than a hardcoded id.
_RESERVED_UNUSED_TOKEN_RE = re.compile(r"^<\|unused_\d+\|>$")


def find_reserved_never_emitted_token_id(tokenizer) -> int | None:
    """Id of a reserved ``<|unused_N|>`` token, or ``None`` if the vocab has none.

    Used to give a newly added placeholder token an output row that the base
    model was trained not to emit, instead of the arbitrary row
    ``resize_token_embeddings`` would leave behind.

    Any of the reserved slots serves equally well — they are numerically
    interchangeable — so the highest id is returned for determinism.
    """
    ids = [
        token_id
        for token, token_id in tokenizer.get_vocab().items()
        if _RESERVED_UNUSED_TOKEN_RE.match(token)
    ]
    return max(ids) if ids else None


def add_audio_token(
    tokenizer,
    marker: str = "<|audio|>",
    keep_special_tokens: list[str] | None = None,
) -> int:
    """Add the audio placeholder marker token to the tokenizer.

    Used for the audio cascade: this single special token is placed in the
    prompt and the vLLM ASR processor replaces it with the transcript tokens at
    request time (see granite_switch.vllm.audio). Registering it as one special
    token keeps the processor's prompt-replacement match clean.

    ``keep_special_tokens`` must list every token an earlier
    ``add_special_tokens({"additional_special_tokens": ...})`` call registered —
    in practice the adapter control tokens from :func:`add_control_tokens`.
    That call *replaces* the additional-special-tokens list instead of appending
    to it, and transformers exposes no way to read the current list back, so any
    token not re-passed here silently drops out of ``all_special_tokens`` and
    out of the saved ``tokenizer_config.json``. Re-passing an already-added
    token is free: it keeps its id and does not grow the vocabulary.

    Must be called before the model's embedding resize so the new row is sized
    in. Returns the marker's token id.
    """
    print(f"\nAdding audio marker token: {marker}")
    # Marker last so it takes the next free id and the kept tokens keep theirs.
    kept = [t for t in (keep_special_tokens or []) if t != marker]
    tokenizer.add_special_tokens({"additional_special_tokens": [*kept, marker]})
    token_id = tokenizer.convert_tokens_to_ids(marker)
    print(f"  {marker}: {token_id}")
    if kept:
        print(f"  (preserved {len(kept)} existing special token(s))")
    return token_id


# The text-only branch of the granite_format content-part loop, and the ChatML
# per-message content assignment. Each is the anchor its family's audio
# injection attaches to.
_GRANITE_TEXT_PART_BRANCH = (
    "                    {%- set content.val = content.val + entry.text %}\n"
    "                {%- endif %}"
)
_CHATML_CONTENT_SET = "{%- set content = message.content | string %}"


def _inject_audio_granite_format(template: str, marker: str) -> str:
    """Add an audio ``elif`` to the granite_format content-part loop.

    The loop already walks the parts for ``entry.type == 'text'`` and silently
    drops everything else, so audio only needs one more branch.
    """
    if _GRANITE_TEXT_PART_BRANCH not in template:
        raise ValueError(
            "Could not find the Granite content-part loop to inject audio "
            "handling; the base chat template may have changed."
        )
    new = (
        "                    {%- set content.val = content.val + entry.text %}\n"
        "                {%- elif 'audio' in entry.type %}\n"
        "                    {%- set content.val = content.val + '" + marker + "' %}\n"
        "                {%- endif %}"
    )
    return template.replace(_GRANITE_TEXT_PART_BRANCH, new, 1)


def _inject_audio_chatml(template: str, marker: str) -> str:
    """Rebuild ChatML's stringified content from its parts, emitting the marker.

    ChatML has no content-part loop at all: its user/system branch does
    ``{%- set content = message.content | string %}``, so a multimodal parts
    *list* renders as that list's Python repr — base64 audio payload included —
    and the marker never appears.

    The flattening block is appended immediately *after* that assignment rather
    than replacing it, which keeps the statement intact as
    :func:`configure_chat_template`'s ALoRA Pass 2 anchor. Since Pass 2 is
    inserted directly after the same anchor, calling this after
    ``configure_chat_template`` lands the flattening ahead of Pass 2 — required,
    because Pass 2 calls ``rsplit`` on ``content`` and so needs a string.

    Part joining mirrors the granite_format loop: a text part is newline-
    separated from whatever precedes it, and the marker is appended directly.
    Part *order* is preserved, so the transcript the ASR processor splices in
    lands where the clip sat relative to the text.
    """
    if _CHATML_CONTENT_SET not in template:
        raise ValueError(
            "Could not find the ChatML per-message content assignment "
            f"({_CHATML_CONTENT_SET!r}) to inject audio handling; the base chat "
            "template may have changed."
        )
    block = (
        "\n"
        "        {#- Audio: ChatML stringifies message.content, so rebuild it from\n"
        "         the parts when it is a list — otherwise a multimodal turn renders\n"
        "         as a Python repr and the audio marker never appears. -#}\n"
        "        {%- if message.content is not string and message.content is iterable %}\n"
        "            {%- set _audio = namespace(val='') %}\n"
        "            {%- for entry in message.content %}\n"
        "                {%- if entry.type is defined and 'audio' in entry.type %}\n"
        "                    {%- set _audio.val = _audio.val + '" + marker + "' %}\n"
        "                {%- elif entry.type is defined and entry.type == 'text' %}\n"
        "                    {%- if _audio.val != '' %}\n"
        "                        {%- set _audio.val = _audio.val + '\\n' %}\n"
        "                    {%- endif %}\n"
        "                    {%- set _audio.val = _audio.val + entry.text %}\n"
        "                {%- endif %}\n"
        "            {%- endfor %}\n"
        "            {%- set content = _audio.val %}\n"
        "        {%- endif %}"
    )
    return template.replace(_CHATML_CONTENT_SET, _CHATML_CONTENT_SET + block, 1)


def configure_audio_chat_template(tokenizer, marker: str = "<|audio|>") -> None:
    """Make the chat template emit the audio marker for audio content parts.

    vLLM passes multimodal chat content to the template as a *list of parts*.
    Neither Granite template family emits anything for an audio part on its own,
    so without this the ``<|audio|>`` marker never reaches the rendered prompt
    and the ASR processor's prompt replacement fails
    (``Failed to apply prompt replacement for mm_items['audio'][0]``).

    Both families are handled, dispatched on :func:`detect_template_format`:

    * **granite_format** (3.x / 4.0 / 4.1) — one ``elif`` added to the existing
      content-part loop. See :func:`_inject_audio_granite_format`.
    * **chatml** (Granite 4.2) — no content-part loop exists, so a flattening
      block rebuilds ``content`` from the parts. See :func:`_inject_audio_chatml`.

    Matching on ``'audio' in entry.type`` covers ``audio`` / ``input_audio`` /
    ``audio_url`` in both.

    Call *after* :func:`configure_chat_template`, gated on audio being enabled;
    the ChatML path depends on that order.

    Scope is the per-message loop. An audio part in the leading system message is
    dropped by both families — matching the pre-existing granite_format
    behaviour, and not a shape any caller sends.

    Raises:
        ValueError: if the template family is unrecognized or its injection
            anchor is missing. Audio is explicitly requested by the time this
            runs, so failing to wire it is loud rather than silent.
    """
    template = tokenizer.chat_template
    if template is None:
        print("Warning: no chat template; skipping audio chat-template handling")
        return

    fmt = detect_template_format(template)
    if fmt is None:
        raise ValueError(
            "Could not detect the chat-template family (found neither "
            "<|im_start|> nor <|start_of_role|>), so audio handling cannot be "
            "injected even though audio was requested."
        )

    if fmt.name == "chatml":
        tokenizer.chat_template = _inject_audio_chatml(template, marker)
    else:
        tokenizer.chat_template = _inject_audio_granite_format(template, marker)
    print(
        f"  Audio chat-template handling added for {fmt.name} "
        f"(emits {marker} for audio parts)"
    )


def build_substitute_token_ids(
    discovered_adapters: list[tuple[str | None, str, str, str | None]],
    lora_substitute_id: int,
    base_reset: bool = False,
) -> list[int]:
    """Build the token-exchange substitute ids, index-aligned with the control tokens.

    The substitute must mirror the token that appears right after the control
    token in the rendered prompt, so swapping the embedding keeps the residual
    stream in-distribution:

      * ALoRA: the first token of the adapter's ``alora_invocation_tokens``.
      * LoRA / built-in: ``lora_substitute_id`` — whatever the chat template
        emits at the start of a no-adapter turn (probed at compose time).
      * base-reset: ``lora_substitute_id`` as well. The token is placed at a turn
        boundary, where the role-open marker is exactly what the decoder expects
        at that position.

    Args:
        discovered_adapters: ``(adapter_path, adapter_name, technology, source)`` tuples.
        lora_substitute_id: Probed sequence-start token id.
        base_reset: When True, prepend the base-reset slot's substitute so the
            list stays aligned with ``add_control_tokens(base_reset=True)``.

    Returns:
        One substitute id per control token, in the same order.
    """
    substitute_ids = [lora_substitute_id] if base_reset else []
    for adapter_path, _name, technology, _source in discovered_adapters:
        if technology == "alora":
            substitute_ids.append(get_alora_first_invocation_token_id(adapter_path))
        else:
            substitute_ids.append(lora_substitute_id)
    return substitute_ids


def configure_chat_template(
    tokenizer,
    discovered_adapters: list[tuple[str | None, str, str, str | None]],
):
    """Inject adapter control token mappings into a Granite chat template.

    Modifies the tokenizer's chat template so that callers can pass
    ``adapter_name="..."`` to ``apply_chat_template()`` and have the
    correct control token inserted automatically:

    * **LoRA** adapters: token at the **beginning** of the sequence.
    * **ALoRA** adapters: token immediately before ``alora_invocation_tokens``
      in the last user message (e.g. before ``<requirements>`` for the
      requirement-checker), or right before the generation prompt for adapters
      whose invocation sequence is the assistant role token sequence and
      therefore does not appear in any user message.

    ALoRA placement uses a two-pass Jinja2 approach embedded in the template:

    * **Pass 1** (before the message loop): scans messages for the last user
      message containing the decoded invocation text; stores its index in
      ``ns.alora_target_idx`` (stays ``-1`` when not found).
    * **Pass 2** (inside the message loop): when the current message is the
      target, splits ``content.val`` on the invocation text and rejoins with
      the control token inserted before the final occurrence.
    * **Fallback** (before ``add_generation_prompt``): fires when
      ``ns.alora_target_idx == -1``, covering adapters whose invocation
      sequence is the assistant role tokens.

    The injection targets Granite-specific template patterns
    (``namespace()``, ``add_generation_prompt``, etc.).  The caller is
    responsible for gating invocation to Granite models only.

    Args:
        tokenizer: HuggingFace tokenizer with a chat_template to modify.
        discovered_adapters: List of ``(adapter_path, adapter_name, technology, source)`` tuples.

    Returns:
        ``(template_format_name, sr_substitute_token_ids)``.  The second element
        maps each Shadow Residual adapter's name to the id of the token its
        control token displaced, which the runtime token exchange must restore.
        It is resolved here because the answer depends on the base model's
        template shape, not only on the adapter's config.
    """
    print("\nConfiguring chat template with adapter support...")

    if tokenizer.chat_template is None:
        print(
            "Warning: Base model does not have a chat template, "
            "skipping adapter configuration"
        )
        return None, {}

    base_chat_template = tokenizer.chat_template

    # Build adapter mapping. Adapters that declare an activation point (``alora``
    # or ``sr``) carry an anchor in their adapter_config.json; resolving it also
    # confirms which convention the checkpoint uses, and that decides placement:
    #
    #   ANCHOR_MODE_ALORA -> control token inserted *before* the invocation
    #                        sequence (or before the generation prompt).
    #   ANCHOR_MODE_SR    -> control token *replaces* the single anchor token at
    #                        the end of the generation prompt.
    #
    # ``lora`` means sequence-start placement. Shadow Residual must never take
    # that path: its adapter stream does not write the KV cache (it reuses the
    # base stream's K/V), so the adapter hidden states at prompt positions are
    # never read back, and activating before the anchor changes no output while
    # still costing compute. Callers therefore classify SR from the checkpoint
    # weights (``is_shadow_residual_adapter``) and pass ``sr`` here, rather than
    # trusting a directory label that no SR training run produces.
    adapter_mapping: dict[str, dict[str, str]] = {}
    sr_anchors: set[str] = set()
    for adapter_info in discovered_adapters:
        adapter_path = adapter_info[0]
        adapter_name = adapter_info[1]
        technology = adapter_info[2]
        entry: dict[str, str] = {
            "token": f"<|{adapter_name}|>",
            "type": technology,
        }
        if technology in (ANCHOR_MODE_ALORA, ANCHOR_MODE_SR) and adapter_path:
            anchor_text, mode = resolve_activation_anchor(adapter_path, tokenizer)
            if technology == ANCHOR_MODE_SR and mode != ANCHOR_MODE_SR:
                raise ValueError(
                    f"Adapter '{adapter_name}' at {adapter_path} was classified as "
                    f"Shadow Residual from its weights, but its adapter_config.json "
                    f"carries 'alora_invocation_tokens' instead of "
                    f"'last_context_token'. SR activates by replacing a single "
                    f"anchor token, so the checkpoint and its metadata disagree."
                )
            entry["type"] = mode
            if mode == ANCHOR_MODE_SR:
                sr_anchors.add(anchor_text)
            else:
                entry["invocation_text"] = anchor_text
        adapter_mapping[adapter_name] = entry

    if len(sr_anchors) > 1:
        raise ValueError(
            f"Shadow Residual adapters in one checkpoint must share a "
            f"last_context_token (it is baked into the chat template), but got "
            f"{sorted(sr_anchors)}. These adapters were trained against "
            f"different base models."
        )

    mapping_entries = []
    for adapter_name, info in adapter_mapping.items():
        if "invocation_text" in info:
            mapping_entries.append(
                f"    '{adapter_name}': {{'token': '{info['token']}', "
                f"'type': '{info['type']}', "
                f"'invocation_text': '{info['invocation_text']}'}}"
            )
        else:
            mapping_entries.append(
                f"    '{adapter_name}': {{'token': '{info['token']}', 'type': '{info['type']}'}}"
            )
    adapter_map_def = (
        "{%- set adapter_map = {\n" + ",\n".join(mapping_entries) + "\n} %}\n"
    )

    adapter_lookup = """{#- Look up adapter token, type, and invocation text from adapter_name -#}
{%- set adapter_token = '' %}
{%- set adapter_type = '' %}
{%- set adapter_invocation_text = '' %}
{%- if adapter_name is defined and adapter_name in adapter_map %}
{%- set adapter_token = adapter_map[adapter_name]['token'] %}
{%- set adapter_type = adapter_map[adapter_name]['type'] %}
{%- if adapter_map[adapter_name]['type'] == 'alora' %}
{%- set adapter_invocation_text = adapter_map[adapter_name]['invocation_text'] %}
{%- endif %}
{%- endif %}

"""

    # Detect the template family so all injection is format-driven rather than
    # hardcoded to the granite_format role markers. Unknown templates are left verbatim.
    fmt = detect_template_format(base_chat_template)
    if fmt is None:
        raise ValueError(
            "Chat template uses an unrecognized role-marker format "
            "(neither <|im_start|> nor <|start_of_role|> found). Adapter control "
            "tokens cannot be auto-inserted, so the composed checkpoint would ship "
            "with adapters that can never be activated via adapter_name=. Failing "
            "at compose time rather than producing a silently-broken checkpoint."
        )
    marker = fmt.role_open_marker
    content_var = fmt.content_accessor

    # LoRA prefix: emit the control token at the sequence start AND arm
    # skip_next_role_marker so the template's very next role-open marker
    # emission is suppressed. This avoids a duplicate-embedding OOD at runtime:
    # the runtime swap replaces the control token's embedding with the
    # role-open marker's embedding, and without this drop the sequence would
    # carry two identical embeddings back-to-back.
    lora_prefix_insertion = """{#- For lora adapters: insert activation token at the very beginning -#}
{%- if adapter_token and adapter_type == 'lora' %}
{{- adapter_token }}
{%- set ns.skip_next_role_marker = true %}
{%- endif %}

"""

    # Pass 1: scan messages before the main loop to find the target user message.
    # We iterate with a different loop variable (_msg) to avoid shadowing the
    # main-loop variable. Using the last occurrence (not first) so multi-turn
    # conversations always activate on the final user turn, which is the one
    # being answered. We iterate the SAME source list the main loop walks
    # (``fmt.loop_var_source``) so ``loop.index0`` here aligns with the main
    # loop's ``loop.index0`` used in Pass 2 (ChatML strips the system message
    # into ``loop_messages``; using ``messages`` would misalign the index).
    alora_pass1 = (
        """{#- ALoRA Pass 1: find the last user message containing the invocation text.
     ns.alora_target_idx stays -1 when the invocation sequence is the assistant role
     token sequence (not present in any user message); the fallback insertion below
     handles that case. -#}
{%- if ns.adapter_type == 'alora' and ns.adapter_invocation_text %}
    {%- for _msg in """
        + fmt.loop_var_source
        + """ %}
        {%- if _msg.role == 'user' %}
            {%- if _msg.content is string and ns.adapter_invocation_text in _msg.content %}
                {%- set ns.alora_target_idx = loop.index0 %}
            {%- elif _msg.content is not string and _msg.content is iterable %}
                {%- set _msg_idx = loop.index0 %}
                {%- for _entry in _msg.content %}
                    {%- if _entry.type == 'text' and ns.adapter_invocation_text in _entry.text %}
                        {%- set ns.alora_target_idx = _msg_idx %}
                    {%- endif %}
                {%- endfor %}
            {%- endif %}
        {%- endif %}
    {%- endfor %}
{%- endif %}
"""
    )

    # Pass 2: runs inside the main message loop after the content variable is
    # assembled. rsplit(..., 1) splits on the last occurrence so the token
    # lands in the right place when the invocation text appears more than once
    # in the message.
    #
    # Token drop (mirrors the role-marker skip-once flag used for LoRA /
    # assistant-boundary ALoRA): we also omit the FIRST CHARACTER of the
    # invocation text. The runtime embedding swap replaces the control-token
    # embedding with the first-invocation-token's embedding; writing the full
    # invocation text after the control token would then produce two copies
    # of that first-invocation-token back to back — an OOD pattern at the
    # swap site.
    #
    # For every granite_format ALoRA invocation text in the standard Granite adapter
    # library (<requirements>, <certainty>, <guardian>, <context>, etc.) the
    # first character is a single '<' that the tokenizer emits as its own token,
    # and the tail of the string retokenizes identically to the tail of the full
    # string. So dropping the first character on the string side is equivalent
    # to dropping exactly the first token on the tokenized side. For ChatML the
    # trained 4.2 adapters use the assistant-boundary invocation
    # (<|im_start|>assistant\\n) and therefore take the fallback path below, not
    # Pass 2; a user-message ChatML ALoRA whose invocation text does not begin
    # with a standalone-tokenizing character would need the first-token-drop
    # invariant re-checked (see the property test in test_chat_template.py).
    #
    # ``content_var`` is ``content.val`` for the granite_format namespace-object content
    # or ``content`` for ChatML's plain-string content.
    alora_pass2 = (
        """    {#- ALoRA Pass 2: inject activation token AND drop the first char of
         the invocation text so the runtime-swapped embedding doesn't duplicate. -#}
    {%- if loop.index0 == ns.alora_target_idx %}
        {%- set _parts = """
        + content_var
        + """.rsplit(ns.adapter_invocation_text, 1) %}
        {%- if _parts | length > 1 %}
            {%- set """
        + content_var
        + """ = _parts[0] + ns.adapter_token + ns.adapter_invocation_text[1:] + _parts[1] %}
        {%- endif %}
    {%- endif %}
"""
    )

    # Fallback for adapters whose invocation sequence is the assistant role tokens:
    # Pass 1 never sets alora_target_idx >= 0 for those, so we emit here instead.
    # Also arm skip_next_role_marker so the generation-prompt role marker
    # that would immediately follow is suppressed — mirrors the LoRA rationale:
    # the runtime swap replaces the control token's embedding with the first
    # invocation token's embedding (the role-open marker), so without this drop
    # the sequence would carry two identical embeddings back-to-back. For ChatML
    # the generation prompt is ``<|im_start|>assistant\\n<think>...`` so the
    # suppressed marker is ``<|im_start|>`` and the trailing ``assistant\\n<think>``
    # is preserved.
    alora_insertion = """{#- ALoRA fallback: insert activation token right before generation prompt.
     Only fires when Pass 1 found no user message with the invocation text
     (alora_target_idx == -1), meaning the adapter activates at the assistant
     role token boundary rather than inside a user message. -#}
{%- if ns.adapter_token and ns.adapter_type == 'alora' and ns.alora_target_idx == -1 %}
{{- ns.adapter_token }}
{%- set ns.skip_next_role_marker = true %}
{%- endif %}
"""

    # Build the modified template
    modified_chat_template = adapter_map_def + adapter_lookup + base_chat_template

    # Merge adapter variables into the ns namespace so they survive loop
    # iterations. alora_target_idx initializes to -1; Pass 1 updates it at
    # render time. ChatML redefines ``ns`` after the system-message split, so we
    # merge into the LAST ``set ns = namespace(...)`` (fmt.ns_merge_last);
    # granite_format has a single ns definition, so we take the first. Done BEFORE the
    # LoRA-prefix insertion so the prefix can be anchored right after this ns
    # definition (the adapter vars must be in scope where the prefix runs).
    ns_pattern = r"(\{%- set ns = namespace\([^)]*)\)"
    ns_matches = list(re.finditer(ns_pattern, modified_chat_template))
    ns_end_after_merge = None
    if ns_matches:
        match = ns_matches[-1] if fmt.ns_merge_last else ns_matches[0]
        ns_def = match.group(1)
        if not ns_def.strip().endswith("("):
            ns_def += ","
        ns_def += (
            "\n                       adapter_token=adapter_token,"
            "\n                       adapter_type=adapter_type,"
            "\n                       adapter_invocation_text=adapter_invocation_text,"
            "\n                       alora_target_idx=-1,"
            "\n                       skip_next_role_marker=false"
            "\n                       )"
        )
        modified_chat_template = (
            modified_chat_template[: match.start()]
            + ns_def
            + modified_chat_template[match.end() :]
        )
        # Position just AFTER the closing ``%}`` of the (rewritten) ns tag —
        # the earliest point at which the adapter vars are in scope and outside
        # the Jinja statement block. The regex captured up to the ``)`` only, so
        # we advance past the ``%}`` that follows. (The unqualified
        # adapter_token/adapter_type in the LoRA prefix are qualified to ns.*
        # after the prefix is inserted below.)
        ns_close_search_from = match.start() + len(ns_def)
        close_idx = modified_chat_template.find("%}", ns_close_search_from)
        ns_end_after_merge = (
            close_idx + len("%}") if close_idx != -1 else ns_close_search_from
        )

    # Insert the LoRA prefix. It must run (a) after the ns definition carrying
    # the adapter vars and (b) before the first role-marker emission, so the
    # control token lands at sequence position 0.
    #
    # Legacy: the single ns is in the preamble and the first emission is the
    # optional system header; anchoring on the first message-start pattern
    # (which sits after that ns) works.
    #
    # ChatML: ``ns`` is redefined AFTER the system-message split but BEFORE the
    # system header is emitted, so we anchor right at the end of that ns
    # definition — after it the adapter vars are in scope and no role marker has
    # been emitted yet.
    if fmt.name == "chatml" and ns_end_after_merge is not None:
        lora_insert_at = ns_end_after_merge
        modified_chat_template = (
            modified_chat_template[:lora_insert_at]
            + "\n"
            + lora_prefix_insertion
            + modified_chat_template[lora_insert_at:]
        )
    else:
        message_start_patterns = [
            r"(\{%- if messages\[0\])",
            r"(\{%- if system_message)",
            r"(\{%- for message in " + fmt.loop_var_source + r")",
            r"(\{%- for message in)",
        ]
        lora_insert_at = None
        for pattern in message_start_patterns:
            m = re.search(pattern, modified_chat_template)
            if m:
                lora_insert_at = m.start()
                break
        if lora_insert_at is not None:
            modified_chat_template = (
                modified_chat_template[:lora_insert_at]
                + lora_prefix_insertion
                + modified_chat_template[lora_insert_at:]
            )
        else:
            modified_chat_template = (
                adapter_map_def
                + adapter_lookup
                + lora_prefix_insertion
                + modified_chat_template[len(adapter_map_def + adapter_lookup) :]
            )

    # The LoRA prefix uses unqualified adapter_token/adapter_type; qualify them
    # to ns.* now that the prefix has been inserted (the ns namespace carries
    # these vars so they survive loop iterations).
    modified_chat_template = modified_chat_template.replace(
        "{%- if adapter_token and adapter_type ==",
        "{%- if ns.adapter_token and ns.adapter_type ==",
    )
    modified_chat_template = modified_chat_template.replace(
        "{{- adapter_token }}", "{{- ns.adapter_token }}"
    )

    # Inject Pass 1 immediately before the main message loop
    for_loop_pattern = r"(\{%- for message in " + fmt.loop_var_source + r" %\})"
    match = re.search(for_loop_pattern, modified_chat_template)
    if match:
        insertion_point = match.start()
        modified_chat_template = (
            modified_chat_template[:insertion_point]
            + alora_pass1
            + modified_chat_template[insertion_point:]
        )

    # Inject Pass 2 inside the loop, after the content variable is built, before
    # the role-dispatch branch. For ChatML the content variable is assigned
    # inside the user/system branch (``{%- set content = message.content | string %}``)
    # AFTER the branch opens, so injecting Pass 2 right before the branch would
    # run before content exists. Instead, for ChatML we inject just after that
    # per-branch content assignment; for granite_format we inject before the branch
    # (content.val is already assembled at the top of the loop body).
    if fmt.name == "chatml":
        chatml_content_set = "{%- set content = message.content | string %}"
        idx = modified_chat_template.find(chatml_content_set)
        if idx != -1:
            insertion_point = idx + len(chatml_content_set)
            modified_chat_template = (
                modified_chat_template[:insertion_point]
                + "\n"
                + alora_pass2
                + modified_chat_template[insertion_point:]
            )
    else:
        match = re.search(fmt.user_role_anchor_re, modified_chat_template)
        if match:
            insertion_point = match.start()
            modified_chat_template = (
                modified_chat_template[:insertion_point]
                + alora_pass2
                + modified_chat_template[insertion_point:]
            )

    # Insert alora fallback before generation prompt
    gen_prompt_pattern = r"(\{%- if add_generation_prompt %\})"
    match = re.search(gen_prompt_pattern, modified_chat_template)
    if match:
        insertion_point = match.start()
        modified_chat_template = (
            modified_chat_template[:insertion_point]
            + alora_insertion
            + modified_chat_template[insertion_point:]
        )
    else:
        modified_chat_template += "\n" + alora_insertion

    # Shadow Residual placement: rewrite the generation prompt so its final
    # token (the adapter's ``last_context_token``) is emitted as the control
    # token instead. Runs before the skip-once wrappers below so it sees the
    # base template's original emission literals.
    sr_replacements = 0
    sr_site = ""
    sr_substitute_token_ids: dict[str, int] = {}
    if sr_anchors:
        (
            modified_chat_template,
            sr_site,
            sr_replacements,
        ) = _replace_anchor_in_generation_prompt(
            modified_chat_template, next(iter(sr_anchors)), tokenizer
        )
        # The runtime swap must restore the token the control token displaced,
        # which is the resolved site — not necessarily the declared anchor.
        site_id = tokenizer.encode(sr_site, add_special_tokens=False)[0]
        sr_substitute_token_ids = {
            name: site_id
            for name, info in adapter_mapping.items()
            if info["type"] == ANCHOR_MODE_SR
        }

    # Skip-once wrapper for every role-open marker emission in the template.
    # ns.skip_next_role_marker is set to true immediately after a LoRA or
    # assistant-boundary ALoRA control token is emitted; the very next role
    # marker consumes the flag and is suppressed. Prevents a duplicate
    # embedding at position 1 (see lora_prefix_insertion / alora_insertion
    # comments).
    #
    # Every role-open marker in the base template appears inside a string
    # literal, either merged with the following role text
    # ('<|start_of_role|>user<|end_of_role|>' / "<|im_start|>system\\n") or
    # standalone ('<|start_of_role|>' + message.role + ... / '<|im_start|>' +
    # message.role + '\\n'). We split at the marker boundary and route only that
    # fragment through the skip-once Jinja block.
    skip_once_block = (
        "{%- if ns.skip_next_role_marker %}"
        "{%- set ns.skip_next_role_marker = false %}"
        "{%- else %}"
        "{{- '" + marker + "' }}"
        "{%- endif %}"
    )
    # Case A: role marker as a standalone literal at the start of a
    # concatenation ({{- '<marker>' + expr + ... }}). Replace the literal
    # emission with the skip block; the rest of the expression stays. Must run
    # before Case B so the standalone marker is consumed and the leftover
    # (``message.role + ...``) is not re-matched.
    modified_chat_template = re.sub(
        fmt.skip_once_re_standalone,
        skip_once_block + "\n        {{- ",
        modified_chat_template,
    )

    # Case B: merged role literal ('<marker>ROLE...'), with or without trailing
    # concatenation. Split the literal so only the marker prefix goes through
    # the skip block and the rest (role text + anything) emits normally. The
    # ChatML variant captures the quote char (group 1), the literal remainder
    # after the marker (group 2), and any trailing ``+``/``~`` concatenation
    # (group 3). The granite_format variant has no quote-char group: remainder is group
    # 1 and tail is group 2.
    def _split_merged(match: "re.Match") -> str:
        if fmt.name == "chatml":
            quote = match.group(1)
            remainder = match.group(2)
            tail = match.group(3)
        else:
            quote = "'"
            remainder = match.group(1)
            tail = match.group(2)
        return (
            skip_once_block
            + "\n        {{- "
            + quote
            + remainder
            + quote
            + tail
            + " }}"
        )

    modified_chat_template = re.sub(
        fmt.skip_once_re_merged,
        _split_merged,
        modified_chat_template,
    )

    tokenizer.chat_template = modified_chat_template
    print(f"Chat template configured with {len(adapter_mapping)} adapter mappings:")
    for adapter_name, info in adapter_mapping.items():
        if info["type"] == ANCHOR_MODE_SR:
            placement = f"replacing '{sr_site}' in the generation prompt"
            anchor = next(iter(sr_anchors))
            if sr_site != anchor:
                placement += (
                    f" (its last_context_token '{anchor}' is absent from some "
                    f"render paths; '{sr_site}' is common to all of them)"
                )
        elif "invocation_text" in info:
            placement = f"before '{info['invocation_text']}' in last user message"
        elif info["type"] == ANCHOR_MODE_ALORA:
            placement = "before generation prompt (fallback)"
        else:
            placement = "at the beginning of the sequence"
        print(f"  - {adapter_name}: {info['token']} ({info['type']}) → {placement}")
    print(f"  Template format: {fmt.name} (role marker {fmt.role_open_marker!r})")
    print("Adapter token insertion logic added:")
    print("  - LoRA tokens: inserted at BEGINNING of sequence")
    print(
        "  - ALoRA tokens (user-message invocation): before invocation text in last user message"
    )
    print("  - ALoRA tokens (role-token invocation): before generation prompt")
    if sr_replacements:
        print(
            f"  - SR tokens: substituted for {sr_site!r} in all "
            f"{sr_replacements} generation-prompt render path(s)"
        )
    return fmt.name, sr_substitute_token_ids
