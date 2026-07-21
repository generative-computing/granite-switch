# SPDX-License-Identifier: Apache-2.0
"""End-to-end compose test for ``python -m granite_switch.composer.compose_granite_switch``.

Runs the exact README command via subprocess, then performs sanity checks on the
output directory.  The compose is expensive (~2.5 min, ~7 GB output) so a
module-scoped fixture composes once and shares the output across all test methods.

Markers: slow, requires_model (both defined in pyproject.toml).
"""

import json
import subprocess
import sys

import pytest

# ---------------------------------------------------------------------------
# Constants (verified against a known-good build of granite-4.0-micro +
# granite-lib-rag-r1.0)
# ---------------------------------------------------------------------------

BASE_MODEL = "ibm-granite/granite-4.0-micro"
ADAPTER_LIBRARY = "ibm-granite/granite-lib-rag-r1.0"
BASE_VOCAB_SIZE = 100352
BASE_NUM_LAYERS = 41  # switch model: base 40 + 1 switch layer
BASE_PARAM_COUNT = 3_402_836_480
BUILD_TIMEOUT = 3600  # 60 min (parallel xdist workers compete for downloads)


# ---------------------------------------------------------------------------
# Fixture: run the build once, share output across all tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def build_output(tmp_path_factory):
    """Run the default build command and return the output directory Path.

    Uses ``tmp_path_factory`` so pytest manages cleanup.  The fixture is
    module-scoped so the ~2.5 min build happens only once.
    """
    output_dir = tmp_path_factory.mktemp("build") / "granite-with-all-aloras"

    cmd = [
        sys.executable,
        "-m",
        "granite_switch.composer.compose_granite_switch",
        "--adapters",
        ADAPTER_LIBRARY,
        "--output",
        str(output_dir),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=BUILD_TIMEOUT,
    )

    # Print stdout/stderr for debugging regardless of outcome
    if result.stdout:
        print(result.stdout[-3000:])  # last 3k chars to avoid flooding
    if result.stderr:
        print("STDERR:", result.stderr[-2000:])

    assert result.returncode == 0, (
        f"Build failed with exit code {result.returncode}.\n"
        f"STDOUT (last 1000 chars):\n{result.stdout[-1000:]}\n"
        f"STDERR (last 1000 chars):\n{result.stderr[-1000:]}"
    )

    return output_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

pytestmark = [pytest.mark.slow, pytest.mark.requires_model]


@pytest.mark.xdist_group("compose_e2e")
class TestBuildE2E:
    """Sanity checks on the output of a default build."""

    def test_output_files_exist(self, build_output):
        """Check that all expected output files are present."""
        # Safetensors shard(s)
        safetensors = list(build_output.glob("*.safetensors"))
        assert safetensors, "No .safetensors files found"

        # Core metadata files
        for name in [
            "config.json",
            "adapter_index.json",
            "compose_report.json",
            "generation_config.json",
        ]:
            assert (build_output / name).exists(), f"Missing {name}"

        # Tokenizer files (at least one of these)
        tokenizer_files = [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ]
        found = [f for f in tokenizer_files if (build_output / f).exists()]
        assert found, "No tokenizer files found"

        # io_configs directory with per-adapter io.yaml files
        io_configs = build_output / "io_configs"
        assert io_configs.is_dir(), "Missing io_configs/ directory"
        yaml_files = list(io_configs.glob("*/io.yaml"))
        assert len(yaml_files) >= 1, (
            f"Expected at least 1 io.yaml file, found {len(yaml_files)}"
        )

    def test_config_correctness(self, build_output):
        """Verify key config.json fields match expected values."""
        config = json.loads((build_output / "config.json").read_text())

        assert config["model_type"] == "granite_switch"
        assert config["architectures"] == ["GraniteSwitchForCausalLM"]
        assert config["num_adapters"] >= 1
        assert config["hidden_size"] == 2560
        assert config["num_hidden_layers"] == BASE_NUM_LAYERS

        # Control token ID lists must match num_adapters
        num_adapters = config["num_adapters"]
        assert len(config["adapter_token_ids"]) == num_adapters

    def test_parameter_count_increased(self, build_output):
        """Verify total parameter count exceeds the base model's 3.4B."""
        from safetensors import safe_open

        total_elements = 0
        for sf_path in sorted(build_output.glob("*.safetensors")):
            with safe_open(str(sf_path), framework="pt") as f:
                for key in f.keys():
                    total_elements += f.get_tensor(key).numel()

        overhead = total_elements - BASE_PARAM_COUNT
        overhead_pct = (overhead / BASE_PARAM_COUNT) * 100

        print(f"\nParameter count: {total_elements:,}")
        print(f"Base count:      {BASE_PARAM_COUNT:,}")
        print(f"Overhead:        {overhead:,} ({overhead_pct:.2f}%)")

        assert total_elements > BASE_PARAM_COUNT, (
            f"Expected > {BASE_PARAM_COUNT:,} params, got {total_elements:,}"
        )

    def test_vocabulary_expanded(self, build_output):
        """Verify vocab_size in config reflects the added control tokens."""
        config = json.loads((build_output / "config.json").read_text())
        num_adapters = config["num_adapters"]
        expected_vocab = BASE_VOCAB_SIZE + num_adapters  # 1 control token per adapter
        assert config["vocab_size"] == expected_vocab, (
            f"Expected vocab_size={expected_vocab} "
            f"(base {BASE_VOCAB_SIZE} + {num_adapters}), "
            f"got {config['vocab_size']}"
        )

    def test_adapter_index(self, build_output):
        """Verify adapter_index.json structure and content."""
        index = json.loads((build_output / "adapter_index.json").read_text())

        adapters = index["adapters"]
        assert len(adapters) >= 1, (
            f"Expected at least 1 adapter in adapter_index.json, got {len(adapters)}"
        )

        for entry in adapters:
            # Required top-level keys
            assert "adapter_index" in entry
            assert "adapter_name" in entry
            assert "io_config" in entry

            # Control token sub-object
            ctrl = entry["control_token"]
            assert "token" in ctrl
            assert "id" in ctrl

            # Token names follow <|name|> pattern
            assert ctrl["token"].startswith("<")
            assert ctrl["token"].endswith(">")

            # IDs are valid token IDs (positive integers)
            assert isinstance(ctrl["id"], int) and ctrl["id"] > 0

    def test_compose_report_valid(self, build_output):
        """Verify compose_report.json is valid and has expected top-level keys."""
        report = json.loads((build_output / "compose_report.json").read_text())

        for key in ["metadata", "base_model_mapping", "adapter_mapping"]:
            assert key in report, f"Missing key '{key}' in compose_report.json"
            assert report[key], f"Empty value for '{key}' in compose_report.json"

    def test_tokenizer_has_control_tokens(self, build_output):
        """Verify the saved tokenizer recognizes all control tokens."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(build_output))
        config = json.loads((build_output / "config.json").read_text())
        expected_vocab = BASE_VOCAB_SIZE + config["num_adapters"]
        assert len(tokenizer) == expected_vocab, (
            f"Expected tokenizer len={expected_vocab}, got {len(tokenizer)}"
        )

        # Cross-check with adapter_index.json
        index = json.loads((build_output / "adapter_index.json").read_text())
        for entry in index["adapters"]:
            ctrl = entry["control_token"]

            # Adapter token
            adapter_id = tokenizer.convert_tokens_to_ids(ctrl["token"])
            assert adapter_id == ctrl["id"], (
                f"Token {ctrl['token']}: expected ID {ctrl['id']}, "
                f"tokenizer returned {adapter_id}"
            )

    def test_generation_config_preserved(self, build_output):
        """Verify generation_config.json is preserved from the base model."""
        gen_config_path = build_output / "generation_config.json"
        assert gen_config_path.exists(), "Missing generation_config.json"

        gen_config = json.loads(gen_config_path.read_text())

        # Must contain essential generation parameters from the base model
        assert "eos_token_id" in gen_config, (
            "generation_config.json missing eos_token_id"
        )
        assert "bos_token_id" in gen_config, (
            "generation_config.json missing bos_token_id"
        )

    def test_model_loads(self, build_output):
        """Verify the model loads with device_map='meta' (no memory used)."""
        from transformers import AutoModelForCausalLM

        # Register our custom model class
        import granite_switch.hf  # noqa: F401

        model = AutoModelForCausalLM.from_pretrained(
            str(build_output),
            device_map="meta",
        )

        assert model is not None
        assert model.config.model_type == "granite_switch"
        assert model.config.num_adapters >= 1
        assert model.config.num_hidden_layers == BASE_NUM_LAYERS
