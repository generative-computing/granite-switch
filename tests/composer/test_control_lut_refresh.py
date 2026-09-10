# SPDX-License-Identifier: Apache-2.0
"""Compose's control-LUT refresh must work on every switch engine.

This is the crash site, not a proxy for it: ``refresh_switch_control_lut`` is
the block at the end of ``build()``, extracted so it can run over both engines
without composing a real 3B checkpoint.

The bug it guards: the refresh used to call
``switch.rebuild_control_to_substitute_lut(config)``, a method defined only on
``SingleSwitch``, while ``create_switch`` returns ``MultiSwitch`` for
``switch_type == "multi"``. Reaching it with a multi engine raised
``AttributeError`` and killed the compose.

Reaching it needs the audio marker. The table is sized
``max(base_vocab_size, max_ctrl_id + 1)`` and control ids are appended, so after
N control tokens it already equals ``len(tokenizer)`` and the refresh is a no-op.
``<|audio|>`` adds a token that is *not* a control token, pushing the vocabulary
one past the last control id — which is why the trigger is
``--switch-type multi --enable-audio`` and not multi alone.

Why nothing caught it: every test that runs a real ``compose --switch-type
multi`` is gated behind ``GRANITE_SWITCH_E2E_MODELS=1``, which CI never sets, and
none of them passes ``--enable-audio`` anyway. These tests are deliberately
CPU-only and ungated so the intersection is covered on every run.
"""

import pytest
import torch

from granite_switch.composer.compose_granite_switch import refresh_switch_control_lut
from granite_switch.config import GraniteSwitchConfig
from granite_switch.hf import GraniteSwitchForCausalLM

SWITCH_TYPES = ["multi"]

# Two adapters -> control ids 200, 201 appended past a 200-token base vocab.
_BASE_VOCAB = 200
_CTRL_IDS = [200, 201]
_SUB_IDS = [5, 7]
# What compose leaves behind: N control tokens, so len(tokenizer) == 202 and the
# table already agrees. Plus <|audio|> -> 203, one past the last control id.
_VOCAB_NO_AUDIO = 202
_VOCAB_WITH_AUDIO = 203


def _config(switch_type):
    return GraniteSwitchConfig(
        vocab_size=_BASE_VOCAB,
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        intermediate_size=64,
        shared_intermediate_size=64,
        max_position_embeddings=64,
        mamba_n_heads=1,
        mamba_expand=1,
        num_adapters=2,
        adapter_ranks=[4, 4],
        max_lora_rank=4,
        adapter_token_ids=list(_CTRL_IDS),
        adapter_names=["a", "b"],
        adapter_substitute_token_ids=list(_SUB_IDS),
        switch_type=switch_type,
        torch_dtype=torch.float32,
    )


def _model(switch_type):
    model = GraniteSwitchForCausalLM(_config(switch_type)).eval()
    expected = "MultiSwitch" if switch_type == "multi" else "SingleSwitch"
    assert type(model.model.switch).__name__ == expected
    return model


@pytest.mark.parametrize("switch_type", SWITCH_TYPES)
def test_audio_marker_makes_the_table_stale_and_refresh_fixes_it(switch_type):
    """The --enable-audio case: this is the compose that used to crash."""
    model = _model(switch_type)
    model.resize_token_embeddings(_VOCAB_WITH_AUDIO)

    # Precondition: the marker pushed vocab_size past the table.
    assert model.model.switch.control_to_substitute_lut.numel() == _VOCAB_NO_AUDIO
    assert model.config.vocab_size == _VOCAB_WITH_AUDIO

    assert refresh_switch_control_lut(model) is True

    lut = model.model.switch.control_to_substitute_lut
    assert lut.numel() == model.config.vocab_size
    assert lut[_CTRL_IDS[0]].item() == _SUB_IDS[0]
    assert lut[_CTRL_IDS[1]].item() == _SUB_IDS[1]
    assert int((lut >= 0).sum()) == len(_CTRL_IDS)
    # Must remain a buffer, or save_pretrained drops it from the checkpoint.
    assert "control_to_substitute_lut" in dict(model.model.switch.named_buffers())


@pytest.mark.parametrize("switch_type", SWITCH_TYPES)
def test_without_audio_the_table_already_agrees(switch_type):
    """Documents why text-only multi compose never hit the bug.

    Control ids are appended, so N control tokens leave the table exactly
    len(tokenizer) long and the refresh reports no work. If this ever starts
    returning True, the sizing rule moved and the reasoning above is stale.
    """
    model = _model(switch_type)
    model.resize_token_embeddings(_VOCAB_NO_AUDIO)

    assert model.model.switch.control_to_substitute_lut.numel() == _VOCAB_NO_AUDIO
    assert model.config.vocab_size == _VOCAB_NO_AUDIO
    assert refresh_switch_control_lut(model) is False


@pytest.mark.parametrize("switch_type", SWITCH_TYPES)
def test_refresh_is_idempotent(switch_type):
    """A second call finds nothing to do — compose must not depend on run count."""
    model = _model(switch_type)
    model.resize_token_embeddings(_VOCAB_WITH_AUDIO)

    assert refresh_switch_control_lut(model) is True
    assert refresh_switch_control_lut(model) is False


@pytest.mark.parametrize("switch_type", SWITCH_TYPES)
def test_no_mapping_is_not_an_error(switch_type):
    """A checkpoint without substitute ids has no table to refresh."""
    config = _config(switch_type)
    config.adapter_substitute_token_ids = None
    model = GraniteSwitchForCausalLM(config).eval()

    assert model.model.switch.control_to_substitute_lut is None
    assert refresh_switch_control_lut(model) is False
