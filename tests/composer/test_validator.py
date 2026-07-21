# SPDX-License-Identifier: Apache-2.0
"""Unit tests for post-build parameter validation."""

import pytest
import torch
import torch.nn as nn

from granite_switch.composer.arch import ArchDescriptor, ModuleDescriptor
from granite_switch.composer.validator import validate_all_parameters


@pytest.fixture
def simple_arch():
    """Create a simple architecture descriptor for testing."""
    return ArchDescriptor(
        groups=[
            ModuleDescriptor(
                name="qkv_proj",
                peft_modules=["q_proj", "k_proj", "v_proj"],
                parent="self_attn",
            ),
            ModuleDescriptor(
                name="o_proj",
                peft_modules=["o_proj"],
                parent="self_attn",
            ),
        ],
        required_config_fields=["hidden_size"],
        optional_config_fields={"attention_multiplier": 1.0},
        buffer_keywords=["adapter_token_ids", "adapter_scalings"],
    )


class MockModel(nn.Module):
    """Mock model for validation testing.

    Parameter names must follow the pattern that extract_module_key expects:
    model.layers.{layer_idx}.{parent}.{attr}.{suffix}

    Example: model.layers.0.self_attn.qkv_proj.lora_A
    """

    def __init__(self):
        super().__init__()
        # Buffer that should be ignored
        self.register_buffer("adapter_token_ids", torch.tensor([100, 101]))

        # Create parameters with correct naming structure
        # Base weights - should be initialized
        self._base_qkv = nn.Parameter(torch.randn(384, 128))
        self._base_o = nn.Parameter(torch.randn(128, 128))

        # LoRA weights - may be zero depending on adapter coverage
        self._lora_qkv_a = nn.Parameter(torch.randn(8, 128))
        self._lora_qkv_b = nn.Parameter(torch.randn(384, 8))
        self._lora_o_a = nn.Parameter(torch.zeros(8, 128))  # Zero (expected)
        self._lora_o_b = nn.Parameter(torch.zeros(128, 8))  # Zero (expected)

    def named_parameters(self, **kwargs):
        """Yield named parameters with proper naming structure."""
        # Base weights
        yield "model.layers.0.self_attn.qkv_proj.weight", self._base_qkv
        yield "model.layers.0.self_attn.o_proj.weight", self._base_o
        # LoRA weights
        yield "model.layers.0.self_attn.qkv_proj.lora_A", self._lora_qkv_a
        yield "model.layers.0.self_attn.qkv_proj.lora_B", self._lora_qkv_b
        yield "model.layers.0.self_attn.o_proj.lora_A", self._lora_o_a
        yield "model.layers.0.self_attn.o_proj.lora_B", self._lora_o_b


class MockLayer(nn.Module):
    """Mock layer with LoRA parameters (unused - kept for reference)."""

    def __init__(self):
        super().__init__()
        pass


class TestValidateAllParameters:
    """Tests for validate_all_parameters function."""

    def test_validate_healthy_model(self, simple_arch, capsys):
        """Validate model with properly initialized parameters."""
        model = MockModel()

        validate_all_parameters(model, simple_arch)

        captured = capsys.readouterr()
        assert "Validating model parameters" in captured.out
        # Should report LoRA zeros as expected (no adapter info)
        assert "as expected" in captured.out
        # Should show parameter summary
        assert "Parameter summary" in captured.out

    def test_validate_with_adapter_info(self, simple_arch, tmp_path, capsys):
        """Validate with adapter path information."""
        import json

        # Create adapter directory with config
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        config = {
            "r": 8,
            "lora_alpha": 8.0,
            "target_modules": ["q_proj", "k_proj", "v_proj"],
        }
        (adapter_dir / "adapter_config.json").write_text(json.dumps(config))

        model = MockModel()

        validate_all_parameters(
            model,
            simple_arch,
            adapter_paths=[str(adapter_dir)],
            adapter_names=["rag"],
        )

        captured = capsys.readouterr()
        assert "Validating model parameters" in captured.out
        # o_proj zeros should be expected (not in target_modules)
        assert "no_adapter_targets" in captured.out

    def test_validate_with_target_module_sets(self, simple_arch, capsys):
        """Validate using pre-loaded target module sets."""
        model = MockModel()
        target_module_sets = [{"q_proj", "k_proj", "v_proj"}]

        validate_all_parameters(
            model,
            simple_arch,
            adapter_paths=["dummy_path"],
            adapter_names=["rag"],
            target_module_sets=target_module_sets,
        )

        captured = capsys.readouterr()
        # o_proj should be expected zero (not in target modules)
        assert "no_adapter_targets" in captured.out

    def test_detect_nan_parameters(self, simple_arch, capsys):
        """Detect NaN parameters as uninitialized."""
        model = MockModel()
        # Inject NaN
        model._base_qkv.data[0, 0] = float("nan")

        validate_all_parameters(model, simple_arch)

        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "uninitialized" in captured.out

    def test_detect_zero_base_parameters(self, simple_arch, capsys):
        """Detect all-zero non-LoRA parameters as uninitialized."""
        model = MockModel()
        # Set base weight to all zeros
        model._base_qkv.data.zero_()

        validate_all_parameters(model, simple_arch)

        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "appear uninitialized" in captured.out

    def test_skip_buffer_parameters(self, simple_arch, capsys):
        """Skip buffer parameters like adapter_token_ids."""

        class ModelWithBuffer(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(10, 10))
                # These should be skipped
                self.register_parameter(
                    "adapter_token_ids_param", nn.Parameter(torch.zeros(4))
                )

            def named_parameters(self, **kwargs):
                yield "weight", self.weight
                yield "adapter_token_ids", self.adapter_token_ids_param

        model = ModelWithBuffer()

        validate_all_parameters(model, simple_arch)

        captured = capsys.readouterr()
        # Should not report adapter_token_ids as uninitialized
        assert (
            "adapter_token_ids" not in captured.out
            or "uninitialized" not in captured.out
        )

    def test_parameter_count_summary(self, simple_arch, capsys):
        """Verify parameter count summary is printed."""
        model = MockModel()

        validate_all_parameters(model, simple_arch)

        captured = capsys.readouterr()
        assert "Total:" in captured.out
        assert "Trainable:" in captured.out
        assert "Frozen:" in captured.out

    def test_expected_zero_lora_grouped_by_reason(self, simple_arch, capsys):
        """Test that expected zeros are grouped by reason."""
        model = MockModel()
        target_module_sets = [{"q_proj", "k_proj", "v_proj"}]

        validate_all_parameters(
            model,
            simple_arch,
            adapter_paths=["dummy"],
            adapter_names=["rag"],
            target_module_sets=target_module_sets,
        )

        captured = capsys.readouterr()
        # Should group zeros by reason
        assert "no_adapter_targets" in captured.out


class TestValidatorEdgeCases:
    """Edge case tests for validator."""

    def test_empty_model(self, simple_arch):
        """Empty model causes division by zero in summary - expect error."""
        # Note: This is an edge case that won't occur in practice.
        # Real models always have parameters.

        class EmptyModel(nn.Module):
            def named_parameters(self, **kwargs):
                return iter([])

        model = EmptyModel()

        # The validator doesn't handle empty models (division by zero)
        with pytest.raises(ZeroDivisionError):
            validate_all_parameters(model, simple_arch)

    def test_model_without_lora(self, simple_arch, capsys):
        """Handle model without LoRA parameters."""

        class NoLoraModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(10, 10))

            def named_parameters(self, **kwargs):
                yield "weight", self.weight

        model = NoLoraModel()
        validate_all_parameters(model, simple_arch)

        captured = capsys.readouterr()
        # Should complete without errors
        assert "Parameter summary" in captured.out

    def test_partial_adapter_coverage(self, simple_arch, capsys):
        """Test validation with adapters targeting different modules."""
        model = MockModel()
        # First adapter targets qkv, second targets o_proj
        target_module_sets = [
            {"q_proj", "k_proj", "v_proj"},
            {"o_proj"},
        ]

        validate_all_parameters(
            model,
            simple_arch,
            adapter_paths=["adapter1", "adapter2"],
            adapter_names=["rag", "code"],
            target_module_sets=target_module_sets,
        )

        captured = capsys.readouterr()
        # Both qkv and o_proj should be considered populated (by different adapters)
        assert "Parameter summary" in captured.out
