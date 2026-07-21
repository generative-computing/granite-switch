# SPDX-License-Identifier: Apache-2.0
"""Config validation tests for GraniteSwitchConfig.

Covers the validators in __init__, default values, and the config
fields that survived the legacy-hiding removal.
"""

import pytest

from granite_switch.config import GraniteSwitchConfig


# ── Helper ────────────────────────────────────────────────────────────


def _valid_kwargs(num_adapters=2, **overrides):
    """Return kwargs for a valid token-exchange config."""
    adapter_names = [f"adapter_{i}" for i in range(num_adapters)]
    base = dict(
        vocab_size=300,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_adapters=num_adapters,
        adapter_token_ids=list(range(500, 500 + num_adapters)),
        adapter_substitute_token_ids=[1] * num_adapters,
        adapter_names=adapter_names,
        max_lora_rank=8,
        adapter_ranks=[8] * num_adapters,
    )
    base.update(overrides)
    return base


# ════════════════════════════════════════════════════════════════════
# 1. Config validation — every ValueError path
# ════════════════════════════════════════════════════════════════════


class TestConfigValidation:

    def test_negative_num_adapters_raises(self):
        with pytest.raises(ValueError, match="num_adapters must be >= 0"):
            GraniteSwitchConfig(**_valid_kwargs(num_adapters=-1, adapter_ranks=None))

    def test_adapter_token_ids_wrong_length_raises(self):
        with pytest.raises(ValueError, match="adapter_token_ids length"):
            GraniteSwitchConfig(**_valid_kwargs(adapter_token_ids=[500]))

    def test_substitute_ids_required_when_adapters_present(self):
        with pytest.raises(ValueError, match="adapter_substitute_token_ids is required"):
            GraniteSwitchConfig(
                **_valid_kwargs(adapter_substitute_token_ids=None)
            )

    def test_substitute_ids_wrong_length_raises(self):
        with pytest.raises(ValueError, match="adapter_substitute_token_ids length"):
            GraniteSwitchConfig(
                **_valid_kwargs(adapter_substitute_token_ids=[1])
            )

    def test_substitute_ids_negative_raises(self):
        with pytest.raises(ValueError, match=">= 0"):
            GraniteSwitchConfig(
                **_valid_kwargs(adapter_substitute_token_ids=[-1, 1])
            )

    def test_duplicate_adapter_token_ids_raises(self):
        with pytest.raises(ValueError, match="adapter_token_ids must be unique"):
            GraniteSwitchConfig(**_valid_kwargs(adapter_token_ids=[500, 500]))

    def test_adapter_ranks_required(self):
        with pytest.raises(ValueError, match="adapter_ranks must be provided"):
            GraniteSwitchConfig(**_valid_kwargs(adapter_ranks=None))

    def test_adapter_ranks_wrong_length(self):
        with pytest.raises(ValueError, match="adapter_ranks length"):
            GraniteSwitchConfig(**_valid_kwargs(adapter_ranks=[8]))

    def test_max_lora_rank_must_match(self):
        with pytest.raises(ValueError, match="max_lora_rank"):
            GraniteSwitchConfig(**_valid_kwargs(max_lora_rank=4))


# ════════════════════════════════════════════════════════════════════
# 2. Defaults
# ════════════════════════════════════════════════════════════════════


class TestConfigDefaults:

    def test_zero_adapter_default(self):
        cfg = GraniteSwitchConfig(num_adapters=0)
        assert cfg.num_adapters == 0
        assert cfg.adapter_token_ids is None
        assert cfg.adapter_substitute_token_ids is None

    def test_projection_head_dim_inferred_from_hidden_size(self):
        cfg = GraniteSwitchConfig(**_valid_kwargs())
        assert cfg.projection_head_dim == 64 // 4


# ════════════════════════════════════════════════════════════════════
# 3. Audio (ASR) preprocessing fields
# ════════════════════════════════════════════════════════════════════


class TestAudioConfig:

    def test_asr_defaults_off(self):
        cfg = GraniteSwitchConfig(num_adapters=0)
        assert cfg.asr_enabled is False
        assert cfg.asr_model_id is None
        assert cfg.asr_device == "cpu"
        assert cfg.asr_pipeline_kwargs is None
        assert cfg.asr_generate_kwargs is None

    def test_longaudio_defaults(self):
        cfg = GraniteSwitchConfig(num_adapters=0)
        assert cfg.asr_max_audio_clips == 32
        assert cfg.asr_generation_reserve_tokens == 8192
        assert cfg.asr_chunk_length_s == 30.0
        assert cfg.asr_chunk_overlap_s == 5.0
        assert cfg.asr_self_chunks is True

    def test_longaudio_round_trip(self, tmp_path):
        cfg = GraniteSwitchConfig(
            num_adapters=0,
            asr_enabled=True,
            asr_max_audio_clips=4,
            asr_generation_reserve_tokens=4096,
            asr_chunk_length_s=20.0,
            asr_chunk_overlap_s=3.0,
            asr_self_chunks=False,
        )
        cfg.save_pretrained(tmp_path)
        loaded = GraniteSwitchConfig.from_pretrained(tmp_path)
        assert loaded.asr_max_audio_clips == 4
        assert loaded.asr_generation_reserve_tokens == 4096
        assert loaded.asr_chunk_length_s == 20.0
        assert loaded.asr_chunk_overlap_s == 3.0
        assert loaded.asr_self_chunks is False

    def test_invalid_max_audio_clips_raises(self):
        with pytest.raises(ValueError, match="asr_max_audio_clips"):
            GraniteSwitchConfig(num_adapters=0, asr_max_audio_clips=0)

    def test_overlap_ge_window_raises(self):
        with pytest.raises(ValueError, match="asr_chunk_overlap_s"):
            GraniteSwitchConfig(
                num_adapters=0, asr_chunk_length_s=10.0, asr_chunk_overlap_s=10.0
            )

    def test_asr_kwargs_round_trip(self, tmp_path):
        # Pipeline/generate kwargs must survive save_pretrained → from_pretrained
        # so the checkpoint stays self-describing about its ASR front-end.
        cfg = GraniteSwitchConfig(
            num_adapters=0,
            asr_enabled=True,
            asr_model_id="openai/whisper-large-v3",
            asr_pipeline_kwargs={"chunk_length_s": 15, "batch_size": 4},
            asr_generate_kwargs={"language": "de", "task": "transcribe"},
        )
        cfg.save_pretrained(tmp_path)
        loaded = GraniteSwitchConfig.from_pretrained(tmp_path)
        assert loaded.asr_enabled is True
        assert loaded.asr_model_id == "openai/whisper-large-v3"
        assert loaded.asr_pipeline_kwargs == {"chunk_length_s": 15, "batch_size": 4}
        assert loaded.asr_generate_kwargs == {"language": "de", "task": "transcribe"}
