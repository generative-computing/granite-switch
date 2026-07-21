# SPDX-License-Identifier: Apache-2.0
"""Tokenizer configuration for adapter control tokens and chat templates.

Extracted from ``compose_granite_switch.py`` to provide testable units for
token management and chat template modification.
"""

import json
import os
import re


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


def add_audio_token(tokenizer, marker: str = "<|audio|>") -> int:
    """Add the audio placeholder marker token to the tokenizer.

    Used for the audio cascade: this single special token is placed in the
    prompt and the vLLM ASR processor replaces it with the transcript tokens at
    request time (see granite_switch.vllm.audio). Registering it as one special
    token keeps the processor's prompt-replacement match clean.

    Must be called before the model's embedding resize so the new row is sized
    in. Returns the marker's token id.
    """
    print(f"\nAdding audio marker token: {marker}")
    tokenizer.add_special_tokens({"additional_special_tokens": [marker]})
    token_id = tokenizer.convert_tokens_to_ids(marker)
    print(f"  {marker}: {token_id}")
    return token_id


def configure_audio_chat_template(tokenizer, marker: str = "<|audio|>") -> None:
    """Make the chat template emit the audio marker for audio content parts.

    The Granite content-part loop only handles ``entry.type == 'text'`` and
    silently drops other parts. vLLM passes multimodal chat content to the
    template as a *list of parts*, so without this the ``<|audio|>`` marker never
    reaches the rendered prompt and the ASR processor's prompt replacement fails
    (``Failed to apply prompt replacement for mm_items['audio'][0]``).

    We inject an ``elif`` that appends the marker for any part whose ``type``
    contains ``'audio'`` (covers ``audio`` / ``input_audio`` / ``audio_url``).
    Call after :func:`configure_chat_template`, gated on audio being enabled.
    """
    template = tokenizer.chat_template
    if template is None:
        print("Warning: no chat template; skipping audio chat-template handling")
        return

    # The text-only branch of the Granite content-part loop:
    old = (
        "                    {%- set content.val = content.val + entry.text %}\n"
        "                {%- endif %}"
    )
    if old not in template:
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
    tokenizer.chat_template = template.replace(old, new, 1)
    print(f"  Audio chat-template handling added (emits {marker} for audio parts)")


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

    # LoRA prefix: emit the control token at the sequence start AND arm
    # skip_next_start_of_role so the template's very next <|start_of_role|>
    # emission is suppressed. This avoids a duplicate-embedding OOD at runtime:
    # the runtime swap replaces the control token's embedding with
    # <|start_of_role|>'s embedding, and without this drop the sequence
    # would carry two identical embeddings back-to-back.
    lora_prefix_insertion = """{#- For lora adapters: insert activation token at the very beginning -#}
{%- if adapter_token and adapter_type == 'lora' %}
{{- adapter_token }}
{%- set ns.skip_next_start_of_role = true %}
{%- endif %}

"""

    # Pass 1: scan messages before the main loop to find the target user message.
    # We iterate with a different loop variable (_msg) to avoid shadowing `message`.
    # Using the last occurrence (not first) so multi-turn conversations always
    # activate on the final user turn, which is the one being answered.
    alora_pass1 = """{#- ALoRA Pass 1: find the last user message containing the invocation text.
     ns.alora_target_idx stays -1 when the invocation sequence is the assistant role
     token sequence (not present in any user message); the fallback insertion below
     handles that case. -#}
{%- if ns.adapter_type == 'alora' and ns.adapter_invocation_text %}
    {%- for _msg in messages %}
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

    # Pass 2: runs inside the main message loop after content.val is assembled.
    # rsplit(..., 1) splits on the last occurrence so the token lands in the
    # right place when the invocation text appears more than once in the message.
    #
    # Token drop (mirrors the <|start_of_role|> skip-once flag used for LoRA /
    # assistant-boundary ALoRA): we also omit the FIRST CHARACTER of the
    # invocation text. The runtime embedding swap replaces the control-token
    # embedding with the first-invocation-token's embedding; writing the full
    # invocation text after the control token would then produce two copies
    # of that first-invocation-token back to back — an OOD pattern at the
    # swap site.
    #
    # For every ALoRA invocation text in the standard Granite adapter library
    # (<requirements>, <certainty>, <guardian>, <context>, etc.) the first
    # character is a single '<' that the tokenizer emits as its own token,
    # and the tail of the string retokenizes identically to the tail of the
    # full string. So dropping the first character on the string side is
    # equivalent to dropping exactly the first token on the tokenized side —
    # no re-merging, no change to what follows.
    alora_pass2 = """    {#- ALoRA Pass 2: inject activation token AND drop the first char of
         the invocation text so the runtime-swapped embedding doesn't duplicate. -#}
    {%- if loop.index0 == ns.alora_target_idx %}
        {%- set _parts = content.val.rsplit(ns.adapter_invocation_text, 1) %}
        {%- if _parts | length > 1 %}
            {%- set content.val = _parts[0] + ns.adapter_token + ns.adapter_invocation_text[1:] + _parts[1] %}
        {%- endif %}
    {%- endif %}
"""

    # Fallback for adapters whose invocation sequence is the assistant role tokens:
    # Pass 1 never sets alora_target_idx >= 0 for those, so we emit here instead.
    # Also arm skip_next_start_of_role so the generation-prompt <|start_of_role|>
    # that would immediately follow is suppressed — mirrors the LoRA rationale:
    # the runtime swap replaces the control token's embedding with the first
    # invocation token's embedding (<|start_of_role|>), so without this drop the
    # sequence would carry two identical embeddings back-to-back.
    alora_insertion = """{#- ALoRA fallback: insert activation token right before generation prompt.
     Only fires when Pass 1 found no user message with the invocation text
     (alora_target_idx == -1), meaning the adapter activates at the assistant
     role token boundary rather than inside a user message. -#}
{%- if ns.adapter_token and ns.adapter_type == 'alora' and ns.alora_target_idx == -1 %}
{{- ns.adapter_token }}
{%- set ns.skip_next_start_of_role = true %}
{%- endif %}
"""

    # Build the modified template
    modified_chat_template = adapter_map_def + adapter_lookup

    # Find insertion point for lora prefix (after ns is defined, before system message)
    message_start_patterns = [
        r"(\{%- if messages\[0\])",
        r"(\{%- if system_message)",
        r"(\{%- for message in)",
    ]

    insertion_point = None
    for pattern in message_start_patterns:
        match = re.search(pattern, base_chat_template)
        if match:
            insertion_point = match.start()
            break

    if insertion_point is not None:
        modified_chat_template += (
            base_chat_template[:insertion_point]
            + lora_prefix_insertion
            + base_chat_template[insertion_point:]
        )
    else:
        modified_chat_template += lora_prefix_insertion + base_chat_template

    # Merge adapter variables into the ns namespace so they survive loop iterations.
    # alora_target_idx initializes to -1; Pass 1 updates it at render time.
    ns_pattern = r"(\{%- set ns = namespace\([^)]+)\)"
    match = re.search(ns_pattern, modified_chat_template)
    if match:
        ns_def = match.group(1)
        if not ns_def.strip().endswith("("):
            ns_def += ","
        ns_def += (
            "\n                       adapter_token=adapter_token,"
            "\n                       adapter_type=adapter_type,"
            "\n                       adapter_invocation_text=adapter_invocation_text,"
            "\n                       alora_target_idx=-1,"
            "\n                       skip_next_start_of_role=false"
            "\n                       )"
        )
        modified_chat_template = (
            modified_chat_template[: match.start()]
            + ns_def
            + modified_chat_template[match.end() :]
        )
        modified_chat_template = modified_chat_template.replace(
            "{%- if adapter_token and adapter_type ==",
            "{%- if ns.adapter_token and ns.adapter_type ==",
        )
        modified_chat_template = modified_chat_template.replace(
            "{{- adapter_token }}", "{{- ns.adapter_token }}"
        )

    # Inject Pass 1 immediately before the main message loop
    for_loop_pattern = r"(\{%- for message in messages %\})"
    match = re.search(for_loop_pattern, modified_chat_template)
    if match:
        insertion_point = match.start()
        modified_chat_template = (
            modified_chat_template[:insertion_point]
            + alora_pass1
            + modified_chat_template[insertion_point:]
        )

    # Inject Pass 2 inside the loop, after content.val is built, before role dispatch
    user_role_pattern = r"(\{%- if \(message\.role == 'user'\) or)"
    match = re.search(user_role_pattern, modified_chat_template)
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

    # Skip-once wrapper for every <|start_of_role|> emission in the template.
    # ns.skip_next_start_of_role is set to true immediately after a LoRA or
    # assistant-boundary ALoRA control token is emitted; the very next role
    # marker consumes the flag and is suppressed. Prevents a duplicate
    # embedding at position 1 (see lora_prefix_insertion / alora_insertion
    # comments).
    #
    # Every <|start_of_role|> in the base template appears inside a string
    # literal, either merged with the following role text ('<|start_of_role|>user<|end_of_role|>')
    # or standalone ('<|start_of_role|>' + message.role + ...). We split at
    # the '<|start_of_role|>' boundary and route only that fragment through
    # the skip-once Jinja block.
    skip_once_block = (
        "{%- if ns.skip_next_start_of_role %}"
        "{%- set ns.skip_next_start_of_role = false %}"
        "{%- else %}"
        "{{- '<|start_of_role|>' }}"
        "{%- endif %}"
    )
    # Case A: '<|start_of_role|>' as a standalone literal, possibly at the
    # start of a concatenation ({{- '<|start_of_role|>' + expr + ... }}).
    # Replace the literal emission with the skip block; the rest of the
    # expression stays.  Handles sites 77 and 79 directly.
    modified_chat_template = re.sub(
        r"\{\{-\s*'<\|start_of_role\|>'\s*\+\s*",
        skip_once_block + "\n        {{- ",
        modified_chat_template,
    )

    # Case B: '<|start_of_role|>ROLE<|end_of_role|>' merged literal (with or
    # without trailing concatenation). Split the literal so only the
    # '<|start_of_role|>' prefix goes through the skip block and the rest
    # ('ROLE<|end_of_role|>' + anything) emits normally.
    # Pattern: {{- 'literal_starting_with_start_of_role' (+ expr | ) }}
    def _split_merged(match: "re.Match") -> str:
        remainder = match.group(1)  # text after <|start_of_role|> up to end of literal
        tail = match.group(2)  # trailing + expr or empty
        return skip_once_block + "\n        {{- '" + remainder + "'" + tail + " }}"

    # Merged literal like '<|start_of_role|>system<|end_of_role|>' followed by
    # optional " + expr + ...". The first group captures everything inside the
    # literal after <|start_of_role|>; the second captures any trailing
    # concatenation up to the closing }}.
    modified_chat_template = re.sub(
        r"\{\{-\s*'<\|start_of_role\|>([^']*)'((?:\s*\+\s*[^}]+?)?)\s*\}\}",
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
    print("Adapter token insertion logic added:")
    print("  - LoRA tokens: inserted at BEGINNING of sequence")
    print(
        "  - ALoRA tokens (user-message invocation): before invocation text in last user message"
    )
    print("  - ALoRA tokens (role-token invocation): before generation prompt")
