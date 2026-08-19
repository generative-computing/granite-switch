# SPDX-License-Identifier: Apache-2.0
"""Unit tests for tokenizer setup functions."""

import json
from unittest.mock import patch

import pytest

from granite_switch.composer.tokenizer_setup import (
    _decode_alora_invocation_text,
    add_audio_token,
    add_control_tokens,
    configure_chat_template,
)

_PATCH_TARGET = "granite_switch.composer.tokenizer_setup._decode_alora_invocation_text"


class MockTokenizer:
    """Mock tokenizer for testing token addition."""

    def __init__(self, initial_vocab_size: int = 100, decode_map: dict | None = None):
        self._vocab = {}
        self._vocab_size = initial_vocab_size
        self._special_tokens = []
        self.chat_template = None
        self._decode_map = decode_map or {}

    def __len__(self):
        return self._vocab_size

    def add_special_tokens(self, special_tokens_dict):
        """Add special tokens and return count added.

        Mirrors transformers: tokens new to the vocabulary are appended and keep
        their id, but ``additional_special_tokens`` is *replaced* by the list
        passed in rather than extended. A second call therefore drops whatever
        the first one registered unless the caller re-passes it.
        """
        tokens = special_tokens_dict.get("additional_special_tokens", [])
        num_added = 0
        for token in tokens:
            if token not in self._vocab:
                self._vocab[token] = self._vocab_size
                self._vocab_size += 1
                num_added += 1
        self._special_tokens = list(tokens)
        return num_added

    def convert_tokens_to_ids(self, token):
        """Convert token to ID."""
        return self._vocab.get(token, -1)

    def decode(self, token_ids, skip_special_tokens=False):
        """Decode token IDs to string."""
        return self._decode_map.get(
            tuple(token_ids), "".join(f"<tok{t}>" for t in token_ids)
        )


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


class TestAddAudioToken:
    """add_audio_token must not evict the control tokens added before it.

    ``add_special_tokens`` replaces ``additional_special_tokens`` instead of
    extending it, and transformers exposes no way to read the current list back,
    so the marker call has to re-pass whatever it wants to keep.
    """

    _ADAPTERS = [
        ("/path/to/rag", "rag", "alora"),
        ("/path/to/code", "code", "lora"),
    ]

    def test_marker_takes_the_next_free_id(self, capsys):
        """The marker is added after the control tokens, so it gets the next id."""
        tokenizer = MockTokenizer(initial_vocab_size=100)
        _, special_tokens = add_control_tokens(tokenizer, self._ADAPTERS)

        token_id = add_audio_token(tokenizer, keep_special_tokens=special_tokens)

        assert token_id == 102
        assert len(tokenizer) == 103

    def test_control_tokens_survive_the_marker(self, capsys):
        """Both the controls and the marker stay registered as special."""
        tokenizer = MockTokenizer(initial_vocab_size=100)
        _, special_tokens = add_control_tokens(tokenizer, self._ADAPTERS)

        add_audio_token(tokenizer, keep_special_tokens=special_tokens)

        assert tokenizer._special_tokens == ["<|rag|>", "<|code|>", "<|audio|>"]

    def test_control_token_ids_are_unchanged(self, capsys):
        """Re-passing an added token keeps its id and does not grow the vocab."""
        tokenizer = MockTokenizer(initial_vocab_size=100)
        ctrl_ids, special_tokens = add_control_tokens(tokenizer, self._ADAPTERS)

        add_audio_token(tokenizer, keep_special_tokens=special_tokens)

        assert [tokenizer.convert_tokens_to_ids(t) for t in special_tokens] == ctrl_ids

    def test_omitting_keep_evicts_the_control_tokens(self, capsys):
        """Witness for why keep_special_tokens exists.

        Without it the marker call drops every control token from the
        additional-special-tokens list — silently, since the ids and the
        vocabulary are untouched.
        """
        tokenizer = MockTokenizer(initial_vocab_size=100)
        add_control_tokens(tokenizer, self._ADAPTERS)

        add_audio_token(tokenizer)

        assert tokenizer._special_tokens == ["<|audio|>"]

    def test_marker_not_duplicated_when_already_kept(self, capsys):
        """Passing the marker in keep_special_tokens must not register it twice."""
        tokenizer = MockTokenizer(initial_vocab_size=100)

        add_audio_token(tokenizer, keep_special_tokens=["<|audio|>", "<|rag|>"])

        assert tokenizer._special_tokens == ["<|rag|>", "<|audio|>"]
        assert len(tokenizer) == 102

    def test_custom_marker(self, capsys):
        """A non-default marker is honored."""
        tokenizer = MockTokenizer(initial_vocab_size=100)

        token_id = add_audio_token(tokenizer, marker="<|sound|>")

        assert tokenizer.convert_tokens_to_ids("<|sound|>") == token_id


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

    def test_adapter_map_in_template(self, capsys):
        """Template contains adapter_map with token and type entries."""
        tokenizer = MockTokenizer()
        tokenizer.chat_template = (
            "{%- if messages[0] %}\n{{- messages[0] }}\n{%- endif %}\n"
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
            "{%- if messages[0] %}\n{{- messages[0] }}\n{%- endif %}\n"
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
            "{%- for message in messages %}\n{{- message }}\n{%- endfor %}\n"
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
