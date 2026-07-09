# SPDX-License-Identifier: Apache-2.0
"""Unit tests for adapter loading and discovery functions."""

import json

import pytest
import torch
import yaml

from granite_switch.composer.adapter_discovery import (
    discover_adapters,
    discover_adapters_from_yaml,
)
from granite_switch.composer.adapter_loader import (
    analyze_source_adapters,
    detect_lora_config,
    detect_present_modules,
    load_adapter_config,
    load_adapter_files,
    load_adapter_target_modules,
)
from granite_switch.composer.arch import ArchDescriptor, ModuleDescriptor, resolve_arch


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
    )


@pytest.fixture
def mock_adapter_dir(tmp_path):
    """Create a mock adapter directory with config and weights."""
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()

    # Create adapter_config.json
    config = {
        "r": 8,
        "lora_alpha": 8.0,
        "target_modules": ["q_proj", "k_proj", "v_proj"],
    }
    config_file = adapter_dir / "adapter_config.json"
    config_file.write_text(json.dumps(config))

    # Create adapter weights (safetensors format)
    weights = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(
            8, 128
        ),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.randn(
            128, 8
        ),
        "base_model.model.model.layers.0.self_attn.k_proj.lora_A.weight": torch.randn(
            8, 128
        ),
        "base_model.model.model.layers.0.self_attn.k_proj.lora_B.weight": torch.randn(
            128, 8
        ),
        "base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight": torch.randn(
            8, 128
        ),
        "base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight": torch.randn(
            128, 8
        ),
    }
    from safetensors.torch import save_file

    save_file(weights, str(adapter_dir / "adapter_model.safetensors"))

    return adapter_dir


class TestLoadAdapterConfig:
    """Tests for load_adapter_config function."""

    def test_load_valid_config(self, tmp_path):
        """Load valid adapter config."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        config = {"r": 8, "lora_alpha": 8.0}
        config_file = adapter_dir / "adapter_config.json"
        config_file.write_text(json.dumps(config))

        result = load_adapter_config(str(adapter_dir))

        assert result["r"] == 8
        assert result["lora_alpha"] == 8.0

    def test_load_missing_config(self, tmp_path):
        """Raise FileNotFoundError for missing config."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()

        with pytest.raises(FileNotFoundError):
            load_adapter_config(str(adapter_dir))


class TestDetectLoraConfig:
    """Tests for detect_lora_config function."""

    def test_detect_single_adapter(self, tmp_path, capsys):
        """Detect config from single adapter."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        config = {"r": 8, "lora_alpha": 16.0}
        (adapter_dir / "adapter_config.json").write_text(json.dumps(config))

        max_rank, default_alpha, ranks, alphas = detect_lora_config([str(adapter_dir)])

        assert max_rank == 8
        assert default_alpha == 16.0
        assert ranks == [8]
        assert alphas == [16.0]

    def test_detect_uniform_adapters(self, tmp_path, capsys):
        """Detect uniform config across multiple adapters."""
        adapters = []
        for i in range(3):
            adapter_dir = tmp_path / f"adapter_{i}"
            adapter_dir.mkdir()
            config = {"r": 8, "lora_alpha": 8.0}
            (adapter_dir / "adapter_config.json").write_text(json.dumps(config))
            adapters.append(str(adapter_dir))

        max_rank, default_alpha, ranks, alphas = detect_lora_config(adapters)

        assert max_rank == 8
        assert default_alpha == 8.0
        assert ranks == [8, 8, 8]

        captured = capsys.readouterr()
        assert "Uniform configuration" in captured.out

    def test_detect_variable_rank_adapters(self, tmp_path, capsys):
        """Detect variable rank adapters."""
        adapters = []
        for i, rank in enumerate([4, 8, 16]):
            adapter_dir = tmp_path / f"adapter_{i}"
            adapter_dir.mkdir()
            config = {"r": rank, "lora_alpha": float(rank)}
            (adapter_dir / "adapter_config.json").write_text(json.dumps(config))
            adapters.append(str(adapter_dir))

        max_rank, default_alpha, ranks, alphas = detect_lora_config(adapters)

        assert max_rank == 16
        assert ranks == [4, 8, 16]

        captured = capsys.readouterr()
        assert "Variable rank" in captured.out

    def test_missing_rank_raises(self, tmp_path):
        """Raise ValueError when rank is missing."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        config = {"lora_alpha": 8.0}  # Missing "r"
        (adapter_dir / "adapter_config.json").write_text(json.dumps(config))

        with pytest.raises(ValueError, match="rank"):
            detect_lora_config([str(adapter_dir)])

    def test_missing_alpha_raises(self, tmp_path):
        """Raise ValueError when lora_alpha is missing."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        config = {"r": 8}  # Missing "lora_alpha"
        (adapter_dir / "adapter_config.json").write_text(json.dumps(config))

        with pytest.raises(ValueError, match="lora_alpha"):
            detect_lora_config([str(adapter_dir)])


class TestLoadAdapterTargetModules:
    """Tests for load_adapter_target_modules function."""

    def test_load_list_target_modules(self, tmp_path):
        """Load target_modules as list."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        config = {
            "r": 8,
            "lora_alpha": 8.0,
            "target_modules": ["q_proj", "k_proj"],
        }
        (adapter_dir / "adapter_config.json").write_text(json.dumps(config))

        result = load_adapter_target_modules([str(adapter_dir)])

        assert len(result) == 1
        assert result[0] == {"q_proj", "k_proj"}

    def test_load_string_target_modules(self, tmp_path):
        """String target_modules pattern returns empty set (validated via weights instead)."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        config = {
            "r": 8,
            "lora_alpha": 8.0,
            "target_modules": ".*proj",
        }
        (adapter_dir / "adapter_config.json").write_text(json.dumps(config))

        result = load_adapter_target_modules([str(adapter_dir)])
        assert result == [set()]

    def test_load_missing_config(self, tmp_path):
        """Missing adapter_config.json raises FileNotFoundError."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()

        with pytest.raises(FileNotFoundError):
            load_adapter_target_modules([str(adapter_dir)])

    def test_load_multiple_adapters(self, tmp_path):
        """Load target modules from multiple adapters."""
        adapters = []
        for i, targets in enumerate([["q_proj"], ["q_proj", "k_proj", "v_proj"]]):
            adapter_dir = tmp_path / f"adapter_{i}"
            adapter_dir.mkdir()
            config = {"r": 8, "lora_alpha": 8.0, "target_modules": targets}
            (adapter_dir / "adapter_config.json").write_text(json.dumps(config))
            adapters.append(str(adapter_dir))

        result = load_adapter_target_modules(adapters)

        assert len(result) == 2
        assert result[0] == {"q_proj"}
        assert result[1] == {"q_proj", "k_proj", "v_proj"}


class TestLoadAdapterFiles:
    """Tests for load_adapter_files function."""

    def test_load_safetensors(self, mock_adapter_dir, capsys):
        """Load adapter from safetensors format."""
        result = load_adapter_files([str(mock_adapter_dir)])

        assert len(result) == 1
        state_dict = result[0]
        assert (
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
            in state_dict
        )

    def test_load_pytorch_bin(self, tmp_path, capsys):
        """Load adapter from pytorch bin format."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()

        weights = {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(
                8, 128
            ),
        }
        torch.save(weights, str(adapter_dir / "adapter_model.bin"))

        result = load_adapter_files([str(adapter_dir)])

        assert len(result) == 1
        assert (
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
            in result[0]
        )

    def test_load_missing_files_raises(self, tmp_path):
        """Raise FileNotFoundError when no weight file exists."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        # No weight files

        with pytest.raises(FileNotFoundError):
            load_adapter_files([str(adapter_dir)])

    def test_load_multiple_adapters(self, tmp_path, capsys):
        """Load multiple adapter files."""
        adapters = []
        for i in range(2):
            adapter_dir = tmp_path / f"adapter_{i}"
            adapter_dir.mkdir()
            weights = {f"layer.{i}.weight": torch.randn(8, 8)}
            from safetensors.torch import save_file

            save_file(weights, str(adapter_dir / "adapter_model.safetensors"))
            adapters.append(str(adapter_dir))

        result = load_adapter_files(adapters)

        assert len(result) == 2
        assert "layer.0.weight" in result[0]
        assert "layer.1.weight" in result[1]


class TestAnalyzeSourceAdapters:
    """Tests for analyze_source_adapters function."""

    def test_analyze_populated_adapter(self, mock_adapter_dir, capsys):
        """Analyze adapter with populated weights."""
        peft_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        result = analyze_source_adapters(
            [str(mock_adapter_dir)],
            peft_modules=peft_modules,
        )

        assert "adapter_names" in result
        assert "module_types" in result
        assert "status" in result
        assert "adapter_ranks" in result
        assert "file_info" in result

        # Check that q_proj.lora_A is populated
        assert (
            result["status"]["q_proj.lora_A"][result["adapter_names"][0]] == "populated"
        )

        # Check that o_proj is not-targeted (not in target_modules)
        assert (
            result["status"]["o_proj.lora_A"][result["adapter_names"][0]]
            == "not-targeted"
        )

    def test_analyze_with_custom_names(self, mock_adapter_dir, capsys):
        """Analyze adapter with custom adapter names."""
        result = analyze_source_adapters(
            [str(mock_adapter_dir)],
            peft_modules=["q_proj"],
            adapter_names=["my_custom_name"],
        )

        assert result["adapter_names"] == ["my_custom_name"]

    def test_analyze_missing_weight_file(self, tmp_path, capsys):
        """Handle missing weight file gracefully."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        config = {"r": 8, "lora_alpha": 8.0, "target_modules": ["q_proj"]}
        (adapter_dir / "adapter_config.json").write_text(json.dumps(config))
        # No weight file

        result = analyze_source_adapters(
            [str(adapter_dir)],
            peft_modules=["q_proj"],
        )

        # Status should be "no-file" for all modules
        adapter_name = result["adapter_names"][0]
        assert result["status"]["q_proj.lora_A"][adapter_name] == "no-file"

    def test_analyze_zero_weights(self, tmp_path, capsys):
        """Detect all-zero weights."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        config = {"r": 8, "lora_alpha": 8.0, "target_modules": ["q_proj"]}
        (adapter_dir / "adapter_config.json").write_text(json.dumps(config))

        # Create weights with all zeros
        weights = {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros(
                8, 128
            ),
        }
        from safetensors.torch import save_file

        save_file(weights, str(adapter_dir / "adapter_model.safetensors"))

        result = analyze_source_adapters(
            [str(adapter_dir)],
            peft_modules=["q_proj"],
        )

        adapter_name = result["adapter_names"][0]
        assert result["status"]["q_proj.lora_A"][adapter_name] == "zero*"


class TestDetectPresentModules:
    """Tests for detect_present_modules function."""

    def test_detect_qkv_present(self, mock_adapter_dir, simple_arch, capsys):
        """Detect QKV modules as present."""
        present_groups, analysis = detect_present_modules(
            [str(mock_adapter_dir)],
            simple_arch,
        )

        # qkv_proj should be present (q_proj, k_proj, v_proj are in weights)
        assert "qkv_proj" in present_groups
        # o_proj should not be present
        assert "o_proj" not in present_groups

    def test_detect_absent_modules_reported(
        self, mock_adapter_dir, simple_arch, capsys
    ):
        """Report absent modules in output."""
        present_groups, _ = detect_present_modules(
            [str(mock_adapter_dir)],
            simple_arch,
        )

        captured = capsys.readouterr()
        assert "Absent module groups" in captured.out
        assert "o_proj" in captured.out

    def test_incompatible_adapter_raises(self, tmp_path, capsys):
        """Adapter with weights for modules unknown to the architecture is rejected.

        Simulates loading a Granite 3.x adapter (gate_proj, up_proj, down_proj)
        against a Granite 4 architecture (input_linear, output_linear).
        """
        from safetensors.torch import save_file

        # Granite 4 arch — knows only input_linear / output_linear for MLP
        granite4_arch = ArchDescriptor(
            groups=[
                ModuleDescriptor(
                    name="qkv_proj",
                    peft_modules=["q_proj", "k_proj", "v_proj"],
                    parent="self_attn",
                ),
                ModuleDescriptor(
                    name="shared_input_linear",
                    peft_modules=["input_linear"],
                    parent="shared_mlp",
                    attr_name="input_linear",
                    num_switch_slices=2,
                ),
                ModuleDescriptor(
                    name="shared_output_linear",
                    peft_modules=["output_linear"],
                    parent="shared_mlp",
                    attr_name="output_linear",
                ),
            ],
            required_config_fields=["hidden_size"],
            optional_config_fields={},
        )

        # Granite 3.x adapter with gate_proj / up_proj / down_proj weights
        adapter_dir = tmp_path / "granite3_adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps(
                {
                    "r": 8,
                    "lora_alpha": 8.0,
                    "target_modules": ["gate_proj", "up_proj", "down_proj"],
                }
            )
        )
        save_file(
            {
                "base_model.model.layers.0.mlp.gate_proj.lora_A.weight": torch.randn(
                    8, 64
                ),
                "base_model.model.layers.0.mlp.gate_proj.lora_B.weight": torch.randn(
                    64, 8
                ),
                "base_model.model.layers.0.mlp.up_proj.lora_A.weight": torch.randn(
                    8, 64
                ),
                "base_model.model.layers.0.mlp.up_proj.lora_B.weight": torch.randn(
                    64, 8
                ),
                "base_model.model.layers.0.mlp.down_proj.lora_A.weight": torch.randn(
                    8, 64
                ),
                "base_model.model.layers.0.mlp.down_proj.lora_B.weight": torch.randn(
                    64, 8
                ),
            },
            str(adapter_dir / "adapter_model.safetensors"),
        )

        with pytest.raises(
            ValueError, match="not recognized by the current architecture"
        ):
            detect_present_modules([str(adapter_dir)], granite4_arch)


class TestAdapterLoadingFromYAML:
    def test_fallback_precedence_and_yaml_parity(self, tmp_path):
        target_model = "granite-4.0-micro"
        adapter_name = "unified-test-adapter"

        # 1. Setup: Create a standard 'lora' folder
        lora_dir = tmp_path / adapter_name / target_model / "lora"
        lora_dir.mkdir(parents=True)
        (lora_dir / "io.yaml").write_text("content: standard-lora")
        (lora_dir / "adapter_config.json").write_text(json.dumps({"r": 8}))
        (lora_dir / "adapter_model.safetensors").write_bytes(b"\x00")

        # 2. Setup: Create a custom-named folder (NOT alora/lora)
        # We will promote this to 'alora' via the fallback
        custom_dir = (
            tmp_path / f"{adapter_name}-latest" / target_model / "experimental-v3"
        )
        custom_dir.mkdir(parents=True)
        (custom_dir / "io.yaml").write_text("content: fallback-promoted-alora")
        (custom_dir / "adapter_config.json").write_text(json.dumps({"r": 16}))
        (custom_dir / "adapter_model.safetensors").write_bytes(b"\x01")

        # --- ACTION: DISCOVERY MODE ---
        # input_path is the directory
        input_path = str(tmp_path)
        arch = resolve_arch("ibm-granite/granite-4.0-micro")
        adapters = discover_adapters(  # default is to prefer alora
            input_path, "granite-4.0-micro", arch, technology_fallback="alora"
        )
        assert len(adapters) == 2

        # --- ACTION: YAML MODE ---
        # Create a manifest pointing to the SAME custom folder
        manifest_file = tmp_path / "manifest.yaml"
        manifest_data = {
            adapter_name: {"path": str(lora_dir.absolute()), "type": "lora"},
            f"{adapter_name}-latest": {
                "path": str(custom_dir.absolute()),
                "type": "alora",
            },
        }
        with open(manifest_file, "w") as f:
            yaml.dump(manifest_data, f)

        input_path = str(manifest_file)
        yaml_adapters = discover_adapters_from_yaml(input_path)

        # Verification of Parity: compare (path, name, technology) - source differs by design
        # discover_adapters returns source=None, discover_adapters_from_yaml returns source=manifest_path
        # Compare order-independently: discover_adapters walks the filesystem (scan
        # order varies by OS/filesystem) while the YAML path follows manifest order,
        # so parity means the same set of adapters, not the same ordering.
        adapters_without_source = [(p, n, t) for p, n, t, _ in adapters]
        yaml_without_source = [(p, n, t) for p, n, t, _ in yaml_adapters]
        assert sorted(adapters_without_source) == sorted(yaml_without_source)
