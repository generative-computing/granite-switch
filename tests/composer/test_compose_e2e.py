# SPDX-License-Identifier: Apache-2.0
"""End-to-end compose test for ``python -m granite_switch.composer.compose_granite_switch``.

Runs the README command via subprocess, then performs sanity checks on the output
directory. A module-scoped fixture composes once per source and shares the output
across all test methods.

Two sources (tests/shared/model_sources.py): ``fake`` on every PR run, a tiny base
and adapter library built from captured copies of the real ones; ``real``, the
published granite-4.0-micro and granitelib-rag-r1.0, in the release tier. Expected
sizes are read off the base each source composes, so both run the same checks.
"""

import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from granite_switch.config import SWITCH_CACHE_LAYERS
from tests.shared import model_sources

BASE_NAME = "granite-4.0-micro"
ADAPTER_LIBRARY = "ibm-granite/granitelib-rag-r1.0"
BUILD_TIMEOUT = 3600  # 60 min (parallel xdist workers compete for downloads)


def _base_facts(base):
    """Vocab, hidden size, layer and parameter counts of the base being composed.

    Read from the base rather than hard-coded, so the fake and real sources run
    the same checks. Parameter counts come from safetensors headers; no weights
    are loaded.
    """
    from huggingface_hub import get_safetensors_metadata
    from safetensors import safe_open
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(base)
    if Path(base).is_dir():
        params = 0
        for sf in Path(base).glob("*.safetensors"):
            with safe_open(str(sf), framework="pt") as f:
                params += sum(math.prod(f.get_slice(k).get_shape()) for k in f.keys())
    else:
        params = sum(get_safetensors_metadata(base).parameter_count.values())
    return SimpleNamespace(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_hidden_layers + SWITCH_CACHE_LAYERS,
        param_count=params,
    )


# ---------------------------------------------------------------------------
# Fixture: run the build once, share output across all tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", params=model_sources.sources())
def build(request, tmp_path_factory):
    """Run the build once per source; returns the output dir and the base's facts."""
    source = request.param
    base = model_sources.base_model(source, BASE_NAME)
    library = model_sources.adapter_library(source, ADAPTER_LIBRARY, BASE_NAME)
    output_dir = tmp_path_factory.mktemp(f"build-{source}") / "granite-switch"

    cmd = [
        sys.executable,
        "-m",
        "granite_switch.composer.compose_granite_switch",
        "--base-model",
        base,
        "--adapters",
        library,
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

    return SimpleNamespace(output=output_dir, base=_base_facts(base))


@pytest.fixture(scope="module")
def build_output(build):
    return build.output


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

pytestmark = [pytest.mark.slow]


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

    def test_config_correctness(self, build_output, build):
        """Verify key config.json fields match expected values."""
        config = json.loads((build_output / "config.json").read_text())

        assert config["model_type"] == "granite_switch"
        assert config["architectures"] == ["GraniteSwitchForCausalLM"]
        assert config["num_adapters"] >= 1
        assert config["hidden_size"] == build.base.hidden_size
        assert config["num_hidden_layers"] == build.base.num_layers

        # Control token ID lists must match num_adapters
        num_adapters = config["num_adapters"]
        assert len(config["adapter_token_ids"]) == num_adapters

    def test_parameter_count_increased(self, build_output, build):
        """Verify total parameter count exceeds the base model's."""
        from safetensors import safe_open

        total_elements = 0
        for sf_path in sorted(build_output.glob("*.safetensors")):
            with safe_open(str(sf_path), framework="pt") as f:
                for key in f.keys():
                    total_elements += f.get_tensor(key).numel()

        base_params = build.base.param_count
        overhead = total_elements - base_params
        overhead_pct = (overhead / base_params) * 100

        print(f"\nParameter count: {total_elements:,}")
        print(f"Base count:      {base_params:,}")
        print(f"Overhead:        {overhead:,} ({overhead_pct:.2f}%)")

        assert total_elements > base_params, (
            f"Expected > {base_params:,} params, got {total_elements:,}"
        )

    def test_vocabulary_expanded(self, build_output, build):
        """Verify vocab_size in config reflects the added control tokens."""
        config = json.loads((build_output / "config.json").read_text())
        num_adapters = config["num_adapters"]
        base_vocab = build.base.vocab_size
        expected_vocab = base_vocab + num_adapters  # 1 control token per adapter
        assert config["vocab_size"] == expected_vocab, (
            f"Expected vocab_size={expected_vocab} "
            f"(base {base_vocab} + {num_adapters}), "
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

    def test_tokenizer_has_control_tokens(self, build_output, build):
        """Verify the saved tokenizer recognizes all control tokens."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(build_output))
        config = json.loads((build_output / "config.json").read_text())
        expected_vocab = build.base.vocab_size + config["num_adapters"]
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

    def test_model_loads(self, build_output, build):
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
        assert model.config.num_hidden_layers == build.base.num_layers
