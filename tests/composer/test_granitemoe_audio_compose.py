# SPDX-License-Identifier: Apache-2.0
"""The audio cascade over a pure sparse MoE base, on both switch engines.

Nothing on the audio path is architecture-specific by design: the compose-time
gate is ``model_type.startswith("granite")`` and the marker injection keys off
the *detected template family*, not the architecture. That is exactly why the
intersection had no coverage — each side is tested thoroughly on its own and
neither suite crosses into the other:

* every ``tests/vllm/`` and composer MoE fixture sets ``num_local_experts=0``
  or a dense ``shared_mlp``;
* every audio test runs on a dense base.

So a regression that only shows up when both hold — a shared-MLP-shaped
assumption in the marker fixup, or a control-LUT size derived from something the
expert bank changes — would pass both suites. These cases are CPU-only,
synthetic and ungated for the same reason
``tests/composer/test_control_lut_refresh.py`` is: gating is what hid the
``--switch-type multi --enable-audio`` crash.

``shared_intermediate_size == 0`` is upstream's own encoding for "no shared
MLP", so it is a *meaningful* value and never a falsy one.
"""

import pytest
import torch

from granite_switch.composer.compose_granite_switch import (
    initialize_control_token_output_rows,
    refresh_switch_control_lut,
)
from granite_switch.config import GraniteSwitchConfig
from granite_switch.hf import GraniteSwitchForCausalLM
from tests.shared.granitemoe_compose import (
    CONTROL_TOKEN_ID,
    DEFAULT_GEOMETRY,
    VOCAB_SIZE,
    create_base_model,
    create_lora_adapter,
)

SWITCH_TYPES = ["single", "multi"]

# The marker is added after the control tokens and is NOT one of them, so it
# pushes the vocabulary one past what the control table was sized for. That
# off-by-one is what makes a stale table specific to --enable-audio.
_AUDIO_MARKER_ID = VOCAB_SIZE
_VOCAB_WITH_AUDIO = VOCAB_SIZE + 1
# Stand-in for a reserved <|unused_N|> row. The lookup that finds a real one is
# covered against real tokenizers in test_audio_marker_output_row.py; here the
# id only has to be a row the fixup can copy from.
_RESERVED_ID = 260
_BYSTANDER_ID = 261


@pytest.fixture(scope="module", params=SWITCH_TYPES)
def audio_moe_model(request, tmp_path_factory):
    """A composed pure sparse MoE switch model with audio recorded in its config.

    Composed through ``GraniteSwitchComposer`` rather than the compose CLI: the
    CLI needs a tokenizer in the base directory, and the synthetic base
    deliberately ships none. The audio-specific steps the CLI would then run —
    the marker's embedding row, its output-row fixup and the control-LUT
    refresh — are applied by the tests themselves, against the same functions
    ``build()`` calls.
    """
    from granite_switch.composer import GraniteSwitchComposer

    root = tmp_path_factory.mktemp(f"moe_audio_{request.param}")
    base_path = root / "base"
    adapter_path = root / "adapter"
    create_base_model(base_path, DEFAULT_GEOMETRY)
    create_lora_adapter(adapter_path, DEFAULT_GEOMETRY)

    model = GraniteSwitchComposer.from_base_and_adapters(
        base_model_name_or_path=str(base_path),
        adapter_paths=[str(adapter_path)],
        adapter_token_ids=[CONTROL_TOKEN_ID],
        adapter_substitute_token_ids=[1],
        adapter_names=["a"],
        switch_type=request.param,
        asr_enabled=True,
    ).eval()

    expected = "MultiSwitch" if request.param == "multi" else "SingleSwitch"
    assert type(model.model.switch).__name__ == expected
    return model


def _fresh(audio_moe_model):
    """A private copy, so a mutating test cannot leak into the next one."""
    config = GraniteSwitchConfig(**audio_moe_model.config.to_dict())
    return GraniteSwitchForCausalLM(config).eval()


@pytest.mark.audio
class TestComposedConfig:
    def test_audio_does_not_resurrect_the_shared_mlp(self, audio_moe_model):
        """``asr_enabled`` must not perturb the no-shared-MLP encoding."""
        config = audio_moe_model.config
        assert config.asr_enabled is True
        assert config.shared_intermediate_size == 0
        layer = audio_moe_model.model.layers[-1]
        assert layer.shared_mlp is None
        assert layer.has_shared_mlp is False
        assert layer.has_experts is True

    def test_audio_does_not_widen_the_adapter_surface(self, audio_moe_model):
        """The adapter surface stays attention-only.

        A pure sparse base has no MLP-side LoRA at all, and enabling audio is
        not a reason for one to appear: the marker is a tokenizer/embedding
        concern, not an adaptation site.
        """
        assert set(audio_moe_model.config.lora_target_modules) == {
            "qkv_proj",
            "o_proj",
        }

    def test_no_zero_width_parameter(self, audio_moe_model):
        """Skipped, not sized to zero.

        ``nn.Linear(H, 0)`` still registers a ``[0, H]`` weight, which would then
        be demanded of the base checkpoint.
        """
        offenders = [
            name
            for name, param in audio_moe_model.named_parameters()
            if 0 in tuple(param.shape)
        ]
        assert offenders == []


@pytest.mark.audio
class TestControlLutWithMarker:
    """The audio off-by-one, on a pure sparse base, on both engines."""

    def test_marker_makes_the_table_stale_and_refresh_fixes_it(self, audio_moe_model):
        model = _fresh(audio_moe_model)
        before = model.model.switch.control_to_substitute_lut.numel()

        model.resize_token_embeddings(_VOCAB_WITH_AUDIO)

        # Guard the guard: if the marker ever stops pushing vocab_size past the
        # table, this test silently stops covering the bug it exists for.
        assert model.config.vocab_size == _VOCAB_WITH_AUDIO
        assert before < _VOCAB_WITH_AUDIO
        assert model.model.switch.control_to_substitute_lut.numel() == before

        assert refresh_switch_control_lut(model) is True

        lut = model.model.switch.control_to_substitute_lut
        assert lut.numel() == model.config.vocab_size
        assert lut[CONTROL_TOKEN_ID].item() == 1
        assert int((lut >= 0).sum()) == 1
        # Must remain a buffer, or save_pretrained drops it from the checkpoint.
        assert "control_to_substitute_lut" in dict(model.model.switch.named_buffers())

    def test_refresh_is_idempotent(self, audio_moe_model):
        model = _fresh(audio_moe_model)
        model.resize_token_embeddings(_VOCAB_WITH_AUDIO)
        assert refresh_switch_control_lut(model) is True
        assert refresh_switch_control_lut(model) is False

    def test_validator_accepts_the_refreshed_table(self, audio_moe_model):
        from granite_switch.composer.validator import validate_control_lut

        model = _fresh(audio_moe_model)
        model.resize_token_embeddings(_VOCAB_WITH_AUDIO)
        refresh_switch_control_lut(model)
        validate_control_lut(model)  # raises on a one-row-short table


@pytest.mark.audio
class TestMarkerOutputRow:
    """The never-emitted row fixup is indifferent to the MLP topology."""

    def test_marker_row_matches_reserved_row(self, audio_moe_model):
        model = _fresh(audio_moe_model)
        model.resize_token_embeddings(_VOCAB_WITH_AUDIO)
        head = model.get_output_embeddings().weight
        reserved_before = head[_RESERVED_ID].clone()
        assert not torch.equal(head[_AUDIO_MARKER_ID], reserved_before)

        initialize_control_token_output_rows(model, [_AUDIO_MARKER_ID], _RESERVED_ID)

        head = model.get_output_embeddings().weight
        assert torch.equal(head[_AUDIO_MARKER_ID], reserved_before)
        assert torch.equal(head[_RESERVED_ID], reserved_before)

    def test_marker_and_control_rows_share_the_fixup(self, audio_moe_model):
        """Compose passes control ids and the marker through one call."""
        model = _fresh(audio_moe_model)
        model.resize_token_embeddings(_VOCAB_WITH_AUDIO)
        head = model.get_output_embeddings().weight
        reserved_before = head[_RESERVED_ID].clone()
        bystander_before = head[_BYSTANDER_ID].clone()

        initialize_control_token_output_rows(
            model, [CONTROL_TOKEN_ID, _AUDIO_MARKER_ID], _RESERVED_ID
        )

        head = model.get_output_embeddings().weight
        assert torch.equal(head[CONTROL_TOKEN_ID], reserved_before)
        assert torch.equal(head[_AUDIO_MARKER_ID], reserved_before)
        assert torch.equal(head[_BYSTANDER_ID], bystander_before)


@pytest.mark.audio
class TestRoundtrip:
    def test_audio_and_no_shared_mlp_survive_save_load(self, audio_moe_model, tmp_path):
        """A strict reload: no ``ignore_mismatched_sizes`` crutch.

        The marker grows the vocabulary after construction, so this is also the
        case where the persistent control-LUT buffer and the saved config must
        agree — a shape disagreement is discarded on load and leaves the buffer
        uninitialized rather than raising.
        """
        model = _fresh(audio_moe_model)
        model.resize_token_embeddings(_VOCAB_WITH_AUDIO)
        initialize_control_token_output_rows(model, [_AUDIO_MARKER_ID], _RESERVED_ID)
        refresh_switch_control_lut(model)

        save_dir = tmp_path / "composed"
        model.save_pretrained(str(save_dir))
        reloaded = GraniteSwitchForCausalLM.from_pretrained(str(save_dir)).eval()

        assert reloaded.config.asr_enabled is True
        assert reloaded.config.shared_intermediate_size == 0
        assert reloaded.config.vocab_size == _VOCAB_WITH_AUDIO
        assert reloaded.model.layers[-1].shared_mlp is None

        lut = reloaded.model.switch.control_to_substitute_lut
        assert lut.numel() == _VOCAB_WITH_AUDIO
        assert lut[CONTROL_TOKEN_ID].item() == 1

        head = reloaded.get_output_embeddings().weight
        assert torch.equal(
            head[_AUDIO_MARKER_ID],
            model.get_output_embeddings().weight[_AUDIO_MARKER_ID],
        )
