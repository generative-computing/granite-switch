# SPDX-License-Identifier: Apache-2.0
"""Unit tests for tokenizer setup functions."""

import json
from unittest.mock import patch

import pytest

from granite_switch.composer.tokenizer_setup import (
    _decode_alora_invocation_text,
    add_control_tokens,
    build_substitute_token_ids,
    configure_chat_template,
)

_PATCH_TARGET = "granite_switch.composer.tokenizer_setup._decode_alora_invocation_text"


class MockTokenizer:
    """Mock tokenizer for testing token addition."""

    def __init__(
        self,
        initial_vocab_size: int = 100,
        decode_map: dict | None = None,
        encode_map: dict | None = None,
    ):
        self._vocab = {}
        self._vocab_size = initial_vocab_size
        self._special_tokens = []
        self.chat_template = None
        self._decode_map = decode_map or {}
        self._encode_map = encode_map or {}

    def __len__(self):
        return self._vocab_size

    def add_special_tokens(self, special_tokens_dict):
        """Add special tokens and return count added."""
        tokens = special_tokens_dict.get("additional_special_tokens", [])
        num_added = 0
        for token in tokens:
            if token not in self._vocab:
                self._vocab[token] = self._vocab_size
                self._vocab_size += 1
                self._special_tokens.append(token)
                num_added += 1
        return num_added

    def convert_tokens_to_ids(self, token):
        """Convert token to ID."""
        return self._vocab.get(token, -1)

    def decode(self, token_ids, skip_special_tokens=False):
        """Decode token IDs to string."""
        return self._decode_map.get(
            tuple(token_ids), "".join(f"<tok{t}>" for t in token_ids)
        )

    def encode(self, text, add_special_tokens=False):
        """Encode a string to token IDs, one id per whitespace-free chunk."""
        if text in self._encode_map:
            return self._encode_map[text]
        return [ord(c) for c in text]


class TestDecodeAloraInvocationText:
    """Tests for the _decode_alora_invocation_text helper."""

    def test_decodes_invocation_tokens(self, tmp_path):
        """Returns the decoded string of alora_invocation_tokens."""
        (tmp_path / "adapter_config.json").write_text(
            json.dumps({"alora_invocation_tokens": [1000, 1001]})
        )
        tokenizer = MockTokenizer(decode_map={(1000, 1001): "<requirements>"})
        assert (
            _decode_alora_invocation_text(str(tmp_path), tokenizer) == "<requirements>"
        )

    def test_missing_config_raises(self, tmp_path):
        """FileNotFoundError when adapter_config.json is absent."""
        with pytest.raises(FileNotFoundError):
            _decode_alora_invocation_text(str(tmp_path), MockTokenizer())

    def test_missing_key_raises(self, tmp_path):
        """ValueError when alora_invocation_tokens key is absent from config."""
        (tmp_path / "adapter_config.json").write_text(
            json.dumps({"peft_type": "ALORA"})
        )
        with pytest.raises(ValueError, match="alora_invocation_tokens"):
            _decode_alora_invocation_text(str(tmp_path), MockTokenizer())

    def test_empty_list_raises(self, tmp_path):
        """ValueError when alora_invocation_tokens is an empty list."""
        (tmp_path / "adapter_config.json").write_text(
            json.dumps({"alora_invocation_tokens": []})
        )
        with pytest.raises(ValueError, match="alora_invocation_tokens"):
            _decode_alora_invocation_text(str(tmp_path), MockTokenizer())


class TestAddControlTokens:
    """Tests for add_control_tokens function."""

    def test_add_single_control_token(self, capsys):
        """Add one adapter control token to tokenizer."""
        tokenizer = MockTokenizer(initial_vocab_size=100)
        adapters = [("/path/to/adapter", "rag", "alora")]

        token_ids, special_tokens = add_control_tokens(tokenizer, adapters)

        assert len(token_ids) == 1
        assert token_ids[0] == 100
        assert special_tokens == ["<|rag|>"]
        assert len(tokenizer) == 101

    def test_add_multiple_control_tokens(self, capsys):
        """Add multiple adapter control tokens."""
        tokenizer = MockTokenizer(initial_vocab_size=200)
        adapters = [
            ("/path/to/rag", "rag", "alora"),
            ("/path/to/code", "code", "lora"),
            ("/path/to/math", "math", "alora"),
        ]

        token_ids, special_tokens = add_control_tokens(tokenizer, adapters)

        assert len(token_ids) == 3
        assert token_ids == [200, 201, 202]
        assert special_tokens == ["<|rag|>", "<|code|>", "<|math|>"]
        assert len(tokenizer) == 203

    def test_control_token_ids_sequential(self, capsys):
        """Verify token IDs are assigned sequentially."""
        tokenizer = MockTokenizer(initial_vocab_size=50)
        adapters = [
            ("/a", "alpha", "alora"),
            ("/b", "beta", "lora"),
            ("/c", "gamma", "alora"),
            ("/d", "delta", "lora"),
        ]

        token_ids, _ = add_control_tokens(tokenizer, adapters)

        assert token_ids == [50, 51, 52, 53]

    def test_empty_adapters_list(self, capsys):
        """Handle empty adapter list."""
        tokenizer = MockTokenizer(initial_vocab_size=100)
        token_ids, special_tokens = add_control_tokens(tokenizer, [])
        assert token_ids == []
        assert special_tokens == []
        assert len(tokenizer) == 100

    def test_idempotent_token_addition(self, capsys):
        """Adding same token twice should not duplicate."""
        tokenizer = MockTokenizer(initial_vocab_size=100)
        adapters = [("/path/to/rag", "rag", "alora")]

        token_ids1, _ = add_control_tokens(tokenizer, adapters)
        size_after_first = len(tokenizer)
        token_ids2, _ = add_control_tokens(tokenizer, adapters)

        assert token_ids1 == token_ids2
        assert len(tokenizer) == size_after_first

    def test_token_format(self, capsys):
        """Verify token format is <|adapter_name|>."""
        tokenizer = MockTokenizer()
        _, special_tokens = add_control_tokens(
            tokenizer, [("/path", "my_adapter", "alora")]
        )
        assert special_tokens[0] == "<|my_adapter|>"


class TestBaseResetToken:
    """``base_reset=True`` prepends a return-to-base control token.

    The engine reads ``adapter_token_ids[0]`` as the base-reset slot when the
    list has ``num_adapters + 1`` entries (``_expert_id_offset = 0``), so the
    token MUST come first — a trailing one would fire adapter 1, not base.
    """

    def test_base_reset_token_is_first(self, capsys):
        """With base_reset=True the list is num_adapters+1 long, base first."""
        tokenizer = MockTokenizer(initial_vocab_size=100)
        adapters = [("/a", "rag", "alora"), ("/b", "code", "lora")]

        token_ids, special_tokens = add_control_tokens(
            tokenizer, adapters, base_reset=True
        )

        assert special_tokens == ["<|base_reset|>", "<|rag|>", "<|code|>"]
        assert len(token_ids) == len(adapters) + 1
        assert token_ids[0] == tokenizer.convert_tokens_to_ids("<|base_reset|>")

    def test_default_emits_no_base_reset_token(self, capsys):
        """Default is unchanged: one token per adapter, no base slot."""
        tokenizer = MockTokenizer(initial_vocab_size=100)
        adapters = [("/a", "rag", "alora"), ("/b", "code", "lora")]

        token_ids, special_tokens = add_control_tokens(tokenizer, adapters)

        assert special_tokens == ["<|rag|>", "<|code|>"]
        assert len(token_ids) == len(adapters)

    def test_adapter_named_base_reset_raises(self, capsys):
        """A collision would make two entries share one token id.

        ``add_special_tokens`` de-duplicates, so the base-reset slot and the
        adapter would resolve to the SAME id: the config's uniqueness check
        would reject the checkpoint, or worse the LUT would collapse both to
        one entry. Fail at compose time with a name the user can act on.
        """
        tokenizer = MockTokenizer(initial_vocab_size=100)
        adapters = [("/a", "base_reset", "lora")]

        with pytest.raises(ValueError, match="base_reset"):
            add_control_tokens(tokenizer, adapters, base_reset=True)


class TestBuildSubstituteTokenIds:
    """Substitutes must stay index-aligned with ``adapter_token_ids``.

    ``config.py`` requires equal lengths and the token-exchange LUT zips the two
    lists, so a missing leading entry would shift every adapter's substitute by
    one — silently swapping each control token to the wrong embedding.
    """

    _ALORA_GETTER = (
        "granite_switch.composer.tokenizer_setup.get_alora_first_invocation_token_id"
    )

    def test_alora_uses_invocation_token_lora_uses_probe(self):
        adapters = [("/a", "rag", "alora", None), ("/b", "code", "lora", None)]

        with patch(self._ALORA_GETTER, return_value=77):
            subs = build_substitute_token_ids(adapters, lora_substitute_id=42)

        assert subs == [77, 42]

    def test_base_reset_prepends_the_probed_substitute(self):
        """The base-reset slot takes the sequence-start token, like LoRA does.

        It is emitted at a turn boundary, where the role-open marker is exactly
        what the decoder expects to see at that position after the swap.
        """
        adapters = [("/a", "rag", "alora", None), ("/b", "code", "lora", None)]

        with patch(self._ALORA_GETTER, return_value=77):
            subs = build_substitute_token_ids(
                adapters, lora_substitute_id=42, base_reset=True
            )

        assert subs == [42, 77, 42]
        assert len(subs) == len(adapters) + 1

    def test_builtin_technology_uses_the_probed_substitute(self):
        """Built-in (empty LoRA) slots are not aLoRA and must not be probed."""
        adapters = [(None, "spare", "builtin", None)]

        subs = build_substitute_token_ids(adapters, lora_substitute_id=42)

        assert subs == [42]


class TestConfigureChatTemplate:
    """Structural tests for configure_chat_template — verify template assembly."""

    def test_no_template_warning(self, capsys):
        """Warn when tokenizer has no chat template."""
        tokenizer = MockTokenizer()
        tokenizer.chat_template = None
        configure_chat_template(tokenizer, [("/path", "rag", "alora")])
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "does not have a chat template" in captured.out

    def test_unrecognized_template_format_raises(self):
        """Raise ValueError when the template has neither role marker.

        A marker-less template cannot receive control-token injection, so the
        composer must fail loudly rather than ship a checkpoint whose adapters
        can never be activated.
        """
        tokenizer = MockTokenizer()
        # Valid Jinja, but uses no recognized role marker (no <|start_of_role|>
        # and no <|im_start|>), so detect_template_format returns None.
        tokenizer.chat_template = (
            "{%- for message in messages %}\n{{- message.content }}\n{%- endfor %}\n"
            "{%- if add_generation_prompt %}\n{{- 'assistant:' }}\n{%- endif %}"
        )
        # Use a lora adapter so mapping construction skips the alora invocation
        # decode (which would read a nonexistent adapter_config.json) and the
        # call reaches detect_template_format, which is what we're exercising.
        with pytest.raises(ValueError, match="unrecognized role-marker format"):
            configure_chat_template(tokenizer, [("/path/code", "code", "lora")])

    def test_adapter_map_in_template(self, capsys):
        """Template contains adapter_map with token and type entries."""
        tokenizer = MockTokenizer()
        # Include a granite-format role marker so detect_template_format recognizes the
        # template; the rest is minimal since this test only checks adapter_map.
        tokenizer.chat_template = (
            "{%- if messages[0] %}\n{{- '<|start_of_role|>' }}{{- messages[0] }}\n{%- endif %}\n"
            "{%- if add_generation_prompt %}\n{{- 'assistant:' }}\n{%- endif %}"
        )
        with patch(_PATCH_TARGET, return_value="<requirements>"):
            configure_chat_template(
                tokenizer,
                [("/path/rag", "rag", "alora"), ("/path/code", "code", "lora")],
            )
        assert "adapter_map" in tokenizer.chat_template
        assert "'rag'" in tokenizer.chat_template
        assert "'code'" in tokenizer.chat_template
        assert "<|rag|>" in tokenizer.chat_template
        assert "<|code|>" in tokenizer.chat_template
        assert "'type': 'alora'" in tokenizer.chat_template
        assert "'type': 'lora'" in tokenizer.chat_template

    def test_alora_invocation_text_in_template(self, capsys):
        """ALoRA adapter entries include invocation_text; LoRA entries do not."""
        tokenizer = MockTokenizer()
        tokenizer.chat_template = (
            "{%- if messages[0] %}\n{{- '<|start_of_role|>' }}{{- messages[0] }}\n{%- endif %}\n"
            "{%- if add_generation_prompt %}\n{{- 'end' }}\n{%- endif %}"
        )
        with patch(_PATCH_TARGET, return_value="<requirements>"):
            configure_chat_template(
                tokenizer,
                [("/path/rag", "rag", "alora"), ("/path/code", "code", "lora")],
            )
        assert "'invocation_text': '<requirements>'" in tokenizer.chat_template
        # LoRA entries must NOT have invocation_text
        code_entry_start = tokenizer.chat_template.index("'code'")
        code_entry_end = tokenizer.chat_template.index("}", code_entry_start)
        assert (
            "invocation_text"
            not in tokenizer.chat_template[code_entry_start:code_entry_end]
        )

    def test_namespace_merge_includes_alora_fields(self, capsys):
        """ns namespace gets adapter_token, adapter_type, adapter_invocation_text, alora_target_idx."""
        tokenizer = MockTokenizer()
        tokenizer.chat_template = (
            "{%- set ns = namespace(found=false) %}\n"
            "{%- for message in messages %}\n{{- '<|start_of_role|>' }}{{- message }}\n{%- endfor %}\n"
            "{%- if add_generation_prompt %}\n{{- 'gen' }}\n{%- endif %}"
        )
        with patch(_PATCH_TARGET, return_value="<requirements>"):
            configure_chat_template(tokenizer, [("/path/rag", "rag", "alora")])

        assert "adapter_token=adapter_token" in tokenizer.chat_template
        assert "adapter_type=adapter_type" in tokenizer.chat_template
        assert (
            "adapter_invocation_text=adapter_invocation_text" in tokenizer.chat_template
        )
        assert "alora_target_idx=-1" in tokenizer.chat_template
        assert "ns.adapter_token" in tokenizer.chat_template


_GRANITE_TEMPLATE_WITH_ANCHOR = (
    "{%- for message in messages %}\n"
    "{{- '<|start_of_role|>' + message.role + '<|end_of_role|>' + message.content }}\n"
    "{%- endfor %}\n"
    "{%- if add_generation_prompt %}\n"
    "{{- '<|start_of_role|>assistant<|end_of_role|>' }}\n"
    "{%- endif %}"
)


def _write_sr_config(
    directory,
    anchor: str = "<|end_of_role|>",
    anchor_id: int | None = 49153,
) -> str:
    """Write a feature/stock-peft-sr shaped adapter_config.json, return its dir."""
    directory.mkdir(parents=True, exist_ok=True)
    config = {
        "peft_type": "LORA",
        "r": 32,
        "target_modules": ["q_proj", "o_proj", "cross_stream"],
        "rank_pattern": {"cross_stream": 96},
        "last_context_token": anchor,
        "last_token": "<|end_of_text|>",
        "last_token_id": 0,
    }
    if anchor_id is not None:
        config["last_context_token_id"] = anchor_id
    (directory / "adapter_config.json").write_text(json.dumps(config))
    return str(directory)


class TestShadowResidualAnchorMode:
    """configure_chat_template with technology='sr' (feature/stock-peft-sr)."""

    def test_adapter_map_entry_is_sr_without_invocation_text(self, tmp_path, capsys):
        """SR entries carry type 'sr' and no invocation_text.

        SR's control token *replaces* the anchor, so there is nothing to insert
        before — invocation_text would be meaningless and the Jinja would try to
        match it inside the last user message.
        """
        path = _write_sr_config(tmp_path / "answerability")
        tokenizer = MockTokenizer(encode_map={"<|end_of_role|>": [49153]})
        tokenizer.chat_template = _GRANITE_TEMPLATE_WITH_ANCHOR

        configure_chat_template(tokenizer, [(path, "answerability", "sr")])

        assert "'type': 'sr'" in tokenizer.chat_template
        entry_start = tokenizer.chat_template.index("'answerability'")
        entry_end = tokenizer.chat_template.index("}", entry_start)
        assert "invocation_text" not in tokenizer.chat_template[entry_start:entry_end]

    def test_generation_prompt_anchor_becomes_a_substitution(self, tmp_path):
        """The anchor emission in the generation prompt is wrapped, not deleted.

        Non-SR renders (and no-adapter renders) must still emit the anchor, so
        the rewrite is a conditional substitution with the anchor as the else
        branch.
        """
        path = _write_sr_config(tmp_path / "answerability")
        tokenizer = MockTokenizer(encode_map={"<|end_of_role|>": [49153]})
        tokenizer.chat_template = _GRANITE_TEMPLATE_WITH_ANCHOR

        configure_chat_template(tokenizer, [(path, "answerability", "sr")])

        assert (
            "{%- if ns.adapter_token and ns.adapter_type == 'sr' %}"
            "{{- ns.adapter_token }}" in tokenizer.chat_template
        )
        # The else branch re-emits the anchor, and the assistant role text that
        # preceded it in the merged literal survives as its own emission.
        assert "{%- else %}{{- '<|end_of_role|>' }}{%- endif %}" in (
            tokenizer.chat_template
        )
        assert "{{- 'assistant' }}" in tokenizer.chat_template

    def test_placement_summary_names_anchor_replacement(self, tmp_path, capsys):
        """The printed placement line distinguishes SR from the aLoRA fallback."""
        path = _write_sr_config(tmp_path / "answerability")
        tokenizer = MockTokenizer(encode_map={"<|end_of_role|>": [49153]})
        tokenizer.chat_template = _GRANITE_TEMPLATE_WITH_ANCHOR

        configure_chat_template(tokenizer, [(path, "answerability", "sr")])

        out = capsys.readouterr().out
        assert "replacing '<|end_of_role|>' in the generation prompt" in out

    def test_sr_label_with_alora_metadata_raises(self, tmp_path):
        """A weights-classified SR adapter carrying aLoRA metadata is a conflict.

        The two conventions place the control token in different positions, so
        guessing would silently activate at the wrong index.
        """
        adapter = tmp_path / "answerability"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text(
            json.dumps({"alora_invocation_tokens": [1000, 1001]})
        )
        tokenizer = MockTokenizer(decode_map={(1000, 1001): "<requirements>"})
        tokenizer.chat_template = _GRANITE_TEMPLATE_WITH_ANCHOR

        with pytest.raises(ValueError, match="classified as Shadow Residual"):
            configure_chat_template(tokenizer, [(str(adapter), "answerability", "sr")])

    def test_sr_adapters_must_share_one_anchor(self, tmp_path):
        """Two SR adapters with different anchors cannot share a chat template."""
        first = _write_sr_config(tmp_path / "a")
        second = _write_sr_config(tmp_path / "b", anchor="</think>", anchor_id=49154)
        tokenizer = MockTokenizer(
            encode_map={"<|end_of_role|>": [49153], "</think>": [49154]}
        )
        tokenizer.chat_template = _GRANITE_TEMPLATE_WITH_ANCHOR

        with pytest.raises(ValueError, match="must share a"):
            configure_chat_template(
                tokenizer, [(first, "a", "sr"), (second, "b", "sr")]
            )

    def test_anchor_absent_from_generation_prompt_raises(self, tmp_path):
        """A template whose generation prompt lacks the anchor fails loudly.

        Silently leaving the control token unplaced would produce a checkpoint
        whose adapter never activates.
        """
        path = _write_sr_config(
            tmp_path / "answerability", anchor="</think>", anchor_id=49154
        )
        tokenizer = MockTokenizer(encode_map={"</think>": [49154]})
        tokenizer.chat_template = _GRANITE_TEMPLATE_WITH_ANCHOR

        with pytest.raises(ValueError, match="never emits"):
            configure_chat_template(tokenizer, [(path, "answerability", "sr")])

    def test_missing_both_key_sets_raises(self, tmp_path):
        """The error names both conventions so the fix is obvious."""
        adapter = tmp_path / "answerability"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA"}))
        tokenizer = MockTokenizer()
        tokenizer.chat_template = _GRANITE_TEMPLATE_WITH_ANCHOR

        with pytest.raises(ValueError, match="last_context_token"):
            configure_chat_template(tokenizer, [(str(adapter), "answerability", "sr")])

    def test_anchor_id_mismatch_raises(self, tmp_path):
        """A recorded id that disagrees with the tokenizer is a compose-time crash."""
        path = _write_sr_config(tmp_path / "answerability", anchor_id=12345)
        tokenizer = MockTokenizer(encode_map={"<|end_of_role|>": [49153]})
        tokenizer.chat_template = _GRANITE_TEMPLATE_WITH_ANCHOR

        with pytest.raises(ValueError, match="last_context_token_id=12345"):
            configure_chat_template(tokenizer, [(path, "answerability", "sr")])

    def test_multi_token_anchor_raises(self, tmp_path):
        """The swap replaces one embedding, so a multi-token anchor is rejected."""
        path = _write_sr_config(tmp_path / "answerability", anchor_id=None)
        tokenizer = MockTokenizer(encode_map={"<|end_of_role|>": [1, 2, 3]})
        tokenizer.chat_template = _GRANITE_TEMPLATE_WITH_ANCHOR

        with pytest.raises(ValueError, match="encodes to 3 tokens"):
            configure_chat_template(tokenizer, [(path, "answerability", "sr")])
