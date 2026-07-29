# SPDX-License-Identifier: Apache-2.0
"""Tokenizer configuration for adapter control tokens and chat templates.

Extracted from ``compose_granite_switch.py`` to provide testable units for
token management and chat template modification.
"""

import json
import os
import re
from dataclasses import dataclass

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


def _load_alora_invocation_token_ids(adapter_path: str) -> list[int]:
    """Load alora_invocation_tokens from adapter_config.json.

    Raises:
        FileNotFoundError: If adapter_config.json is not found at adapter_path.
        ValueError: If alora_invocation_tokens is missing or empty.
    """
    config_path = os.path.join(adapter_path, "adapter_config.json")
    with open(config_path) as f:
        adapter_config = json.load(f)

    token_ids = adapter_config.get("alora_invocation_tokens")
    if not token_ids:
        raise ValueError(
            f"alora_invocation_tokens is missing or empty in {config_path}"
        )
    return token_ids


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


def add_control_tokens(
    tokenizer,
    discovered_adapters: list[tuple[str | None, str, str, str | None]],
) -> tuple[list[int], list[str]]:
    """Add control tokens to the tokenizer for each adapter.

    Each adapter gets one control token: ``<|adapter_name|>`` which activates that adapter.

    Args:
        tokenizer: HuggingFace tokenizer.
        discovered_adapters: List of ``(adapter_path, adapter_name, technology, source)`` tuples.

    Returns:
        ``(adapter_token_ids, special_tokens)``

        adapter_token_ids has length ``num_adapters``.
    """
    print(f"\nAdding control tokens for {len(discovered_adapters)} adapter(s)...")

    special_tokens = []
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

    # Get token IDs
    print("\nToken ID mapping:")
    adapter_token_ids = []
    for adapter_info in discovered_adapters:
        adapter_name = adapter_info[1]
        token_name = f"<|{adapter_name}|>"
        token_id = tokenizer.convert_tokens_to_ids(token_name)
        adapter_token_ids.append(token_id)
        print(f"  {token_name}: {token_id}")

    return adapter_token_ids, special_tokens


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
    """
    print("\nConfiguring chat template with adapter support...")

    if tokenizer.chat_template is None:
        print(
            "Warning: Base model does not have a chat template, "
            "skipping adapter configuration"
        )
        return

    base_chat_template = tokenizer.chat_template

    # Build adapter mapping. For ALoRA adapters, decode alora_invocation_tokens
    # so the template can locate the right insertion point at render time.
    adapter_mapping: dict[str, dict[str, str]] = {}
    for adapter_info in discovered_adapters:
        adapter_path = adapter_info[0]
        adapter_name = adapter_info[1]
        technology = adapter_info[2]
        entry: dict[str, str] = {
            "token": f"<|{adapter_name}|>",
            "type": technology,
        }
        if technology == "alora" and adapter_path is not None:
            entry["invocation_text"] = _decode_alora_invocation_text(
                adapter_path, tokenizer
            )
        adapter_mapping[adapter_name] = entry

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
        if "invocation_text" in info:
            placement = f"before '{info['invocation_text']}' in last user message"
        else:
            placement = "before generation prompt (fallback)"
        print(f"  - {adapter_name}: {info['token']} ({info['type']}) → {placement}")
    print(f"  Template format: {fmt.name} (role marker {fmt.role_open_marker!r})")
    print("Adapter token insertion logic added:")
    print("  - LoRA tokens: inserted at BEGINNING of sequence")
    print(
        "  - ALoRA tokens (user-message invocation): before invocation text in last user message"
    )
    print("  - ALoRA tokens (role-token invocation): before generation prompt")
    return fmt.name
