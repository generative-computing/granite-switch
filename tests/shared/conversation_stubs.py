# SPDX-License-Identifier: Apache-2.0
"""A hermetic tokenizer stub for Conversation tests.

Uses the REAL Granite chat template fixture and the REAL
``configure_chat_template`` injection, so the delta-by-subtraction logic is
exercised against genuine Jinja rather than a hand-written approximation. Only
the tokenizer's encode/decode is stubbed, and only as much as
``Conversation`` uses: ``apply_chat_template``, ``__call__``, ``decode``.

The stub's encoding is deliberately simple but must keep the one property the
real tokenizers have and ``Conversation`` depends on: **special tokens are
atomic**, so a join that lands on one cannot be merged across.
"""

import os
import re
import zlib
from unittest.mock import patch

from jinja2.sandbox import ImmutableSandboxedEnvironment

from granite_switch.composer.tokenizer_setup import configure_chat_template

_TESTS_DIR = os.path.dirname(os.path.dirname(__file__))
_FIXTURES = os.path.join(_TESTS_DIR, "composer", "fixtures")
_SPECIAL_RE = re.compile(r"<\|[^|>]*\|>")
_PATCH_TARGET = "granite_switch.composer.tokenizer_setup._decode_alora_invocation_text"

# Ids are arbitrary but must be stable and disjoint: specials from 900 up,
# ordinary text hashed into [1000, 1999].
#
# Everything stays under 2000 so these sequences can also be fed to the vLLM
# switch worker, whose mock config uses vocab_size=2000 (an id at or above it
# would index past the token-exchange LUT). Collisions inside the text range are
# possible and harmless: routing depends only on where the control tokens are.
_BASE_SPECIAL_ID = 900
_TEXT_ID_BASE = 1000
_TEXT_ID_SPAN = 1000


class StubTokenizer:
    """Minimal tokenizer over the real injected chat template."""

    def __init__(self, template, special_tokens):
        self.chat_template = template
        self._special_ids = {
            text: _BASE_SPECIAL_ID + i for i, text in enumerate(special_tokens)
        }
        self._id_to_special = {v: k for k, v in self._special_ids.items()}

    # ── the three methods Conversation uses ───────────────────────────────────
    def apply_chat_template(
        self, messages, add_generation_prompt=False, tokenize=False, **kw
    ):
        assert tokenize is False, "the stub only renders text"
        # trim_blocks/lstrip_blocks match how transformers builds its own
        # environment (utils/chat_template_utils.py:489). A bare Environment()
        # leaks Jinja block indentation, which the ChatML fixture has plenty of:
        # a three-message render came out 193 characters instead of 170, and the
        # extra whitespace landed between the history and the assistant marker --
        # exactly where the turn terminator is derived and the delta is cut.
        return (
            ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
            .from_string(self.chat_template)
            .render(
                messages=messages, add_generation_prompt=add_generation_prompt, **kw
            )
        )

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self._encode(text)}

    def convert_ids_to_tokens(self, tid):
        """Real tokenizers have this, and Conversation's diagnostics use it.

        Without it the placement diagnostic in ``_explain_no_prefix`` degrades
        silently to a generic message, so a test asserting the specific one would
        pass for the wrong reason.
        """
        return self._id_to_special.get(tid) or self._words.get(tid)

    def decode(self, ids, skip_special_tokens=False):
        out = []
        for i in ids:
            if i in self._id_to_special:
                if not skip_special_tokens:
                    out.append(self._id_to_special[i])
            else:
                out.append(self._words.get(i, ""))
        return "".join(out)

    # ── helpers ───────────────────────────────────────────────────────────────
    _words: dict = {}

    def token_id(self, tokentext):
        """Id of a special token, e.g. ``"<|gsm8k|>"``."""
        return self._special_ids[tokentext]

    def _encode(self, text):
        ids = []
        pos = 0
        for m in _SPECIAL_RE.finditer(text):
            ids += self._encode_plain(text[pos : m.start()])
            special = m.group(0)
            # Unknown <|...|> spellings fall back to plain text, mirroring a real
            # tokenizer that only knows the tokens actually added to its vocab.
            if special in self._special_ids:
                ids.append(self._special_ids[special])
            else:
                ids += self._encode_plain(special)
            pos = m.end()
        ids += self._encode_plain(text[pos:])
        return ids

    def _encode_plain(self, text):
        ids = []
        for chunk in re.findall(r"\s+|\w+|[^\s\w]", text):
            # crc32, not hash(): str hashing is salted per process, and ids that
            # change between runs would make any failure irreproducible.
            tid = _TEXT_ID_BASE + (zlib.crc32(chunk.encode()) % _TEXT_ID_SPAN)
            type(self)._words[tid] = chunk
            ids.append(tid)
        return ids


def make_stub_tokenizer(adapters, chatml=False):
    """Build a StubTokenizer whose template knows ``adapters``.

    Args:
        adapters: list of ``(name, technology, invocation_text)``. ``technology``
            is ``"alora"`` or ``"lora"``; ``invocation_text`` is ignored for lora.
        chatml: use the ChatML fixture instead of the role-marker one.
    """
    name = "granite_chatml_template.jinja" if chatml else "granite_chat_template.jinja"
    with open(os.path.join(_FIXTURES, name)) as f:
        template = f.read()

    holder = type("H", (), {"chat_template": template})()
    invocations = [inv for _n, tech, inv in adapters if tech == "alora"]
    discovered = [(f"/path/{n}", n, tech, None) for n, tech, _inv in adapters]
    with patch(_PATCH_TARGET, side_effect=invocations):
        configure_chat_template(holder, discovered)

    specials = [f"<|{n}|>" for n, _t, _i in adapters]
    # Role markers etc. must also be atomic; collect every <|...|> the template
    # can emit so the stub treats them as single tokens.
    specials += sorted(set(_SPECIAL_RE.findall(holder.chat_template)) - set(specials))
    return StubTokenizer(holder.chat_template, specials)


class StubConfig:
    """Stand-in for GraniteSwitchConfig, for the guards."""

    def __init__(self, adapter_token_ids, switch_type="multi"):
        self.adapter_token_ids = list(adapter_token_ids)
        self.switch_type = switch_type
