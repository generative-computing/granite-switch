# SPDX-License-Identifier: Apache-2.0
"""The conversation suites' adapter-map parser against the template the composer emits.

``_parse_adapter_map`` reads name -> (token, type, invocation_text) back out of a
composed chat template. Both conversation suites (prefix-cache and concurrency)
depend on it, and both are GPU-gated, so a parser that silently stops matching
shows up only as "need >=2 alora adapters" on a GPU run that never happens.

That is what happened: configure_chat_template started writing
``'invocation_drop_chars': N`` after ``invocation_text`` on every aLoRA entry, the
regex still required ``}`` right after ``invocation_text``, and every aLoRA entry
dropped out. These tests build the map with the real ``configure_chat_template``
rather than a hand-written string, so the next format change fails here, on CPU.
"""

from types import SimpleNamespace
from unittest.mock import patch

from granite_switch.composer.tokenizer_setup import configure_chat_template
from tests.integration._conversation_prefix_cache_worker import _parse_adapter_map

_PATCH_TARGET = "granite_switch.composer.tokenizer_setup._decode_alora_invocation_text"
_INVOCATION = "<|start_of_role|>assistant<|end_of_role|>"

# A minimal Granite role-marker template: enough for configure_chat_template to
# detect the format and inject its adapter map.
_TEMPLATE = (
    "{%- for message in messages %}"
    "{{- '<|start_of_role|>' + message['role'] + '<|end_of_role|>' "
    "+ message['content'] + '<|end_of_text|>\\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|start_of_role|>assistant<|end_of_role|>' }}"
    "{%- endif %}"
)


class _Tokenizer(SimpleNamespace):
    """encode/decode just enough to size the aLoRA first-token drop."""

    def encode(self, text, add_special_tokens=False):
        return [7, 1]

    def decode(self, token_ids, skip_special_tokens=False):
        return "<|start_of_role|>"


def _composed_template(adapters):
    tok = _Tokenizer(chat_template=_TEMPLATE)
    with patch(_PATCH_TARGET, return_value=_INVOCATION):
        configure_chat_template(tok, adapters)
    return tok.chat_template


def test_alora_entry_with_drop_chars_is_parsed():
    template = _composed_template([("/path/a", "answerability", "alora")])
    assert "invocation_drop_chars" in template, (
        "the composer no longer writes invocation_drop_chars; this test's premise "
        "is stale"
    )
    amap = _parse_adapter_map(template)
    assert amap == {"answerability": ("<|answerability|>", "alora", _INVOCATION)}


def test_lora_and_alora_entries_are_both_parsed():
    template = _composed_template(
        [
            ("/path/a", "answerability", "alora"),
            ("/path/b", "certainty", "alora"),
            ("/path/c", "guardian", "lora"),
        ]
    )
    amap = _parse_adapter_map(template)
    assert amap == {
        "answerability": ("<|answerability|>", "alora", _INVOCATION),
        "certainty": ("<|certainty|>", "alora", _INVOCATION),
        "guardian": ("<|guardian|>", "lora", None),
    }
