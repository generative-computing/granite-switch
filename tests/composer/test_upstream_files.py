# SPDX-License-Identifier: Apache-2.0
"""Auxiliary file preservation tests for Granite models.

Verifies that ``_copy_upstream_auxiliary_files()`` works correctly for
supported Granite model families, and that a full built-in-adapter build
completes correctly.

Two test tiers:
  1. Copy logic — calls ``_copy_upstream_auxiliary_files()`` directly against
     resolved base models.  Fast (no model loading, no GPU).
  2. Full build E2E — runs the build CLI with ``--built-in-adapters base``.
     Tests the full pipeline including ``save_pretrained()`` overwrite
     behavior, model loading, and tokenizer.

Two sources (tests/shared/model_sources.py): ``fake`` on every PR run, a tiny base
built from the captured files of the real repo (including the files the copy must
skip); ``real``, the published granite-4.0-micro, in the release tier.

Markers: slow (full build tests only).
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from granite_switch.composer.compose_granite_switch import (
    _copy_upstream_auxiliary_files,
    _resolve_base_model_path,
)
from tests.shared import model_sources

# ---------------------------------------------------------------------------
# Model lists (Granite only)
# ---------------------------------------------------------------------------

#: (family, base name in tests/shared/fake_bases.py). Smallest representative.
FAMILY_MODELS = [
    ("granitemoehybrid", "granite-4.0-micro"),
]


def _cases(models):
    """Each (family, base) once per source; the real source is release-tier."""
    return [
        pytest.param(
            (family, base, src.values[0]),
            id=f"{family}-{src.values[0]}",
            marks=src.marks,
        )
        for family, base in models
        for src in model_sources.sources()
    ]


BUILD_TIMEOUT = 3600  # 60 min (parallel xdist workers compete for downloads)

# Extensions that must never be copied
WEIGHT_EXTENSIONS = {".safetensors", ".bin", ".pt", ".ckpt"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", params=_cases(FAMILY_MODELS))
def resolved_model(request):
    """Resolve each base to a local path (downloads the real one if needed).

    Returns ``(family, model_id, local_path)``.
    """
    family, base, source = request.param
    model_id = model_sources.base_model(source, base)
    local_path = _resolve_base_model_path(model_id)
    return family, model_id, local_path


@pytest.fixture(scope="module", params=_cases(FAMILY_MODELS))
def built_in_build_output(request, tmp_path_factory):
    """Run a built-in-only build for each base and source.

    Returns ``(family, model_id, output_dir)``.
    """
    family, base, source = request.param
    model_id = model_sources.base_model(source, base)
    output_dir = tmp_path_factory.mktemp(f"build_{family}_{source}") / "output"

    cmd = [
        sys.executable,
        "-m",
        "granite_switch.composer.compose_granite_switch",
        "--base-model",
        model_id,
        "--built-in-adapters",
        "base",
        "--output",
        str(output_dir),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=BUILD_TIMEOUT,
    )

    if result.stdout:
        print(result.stdout[-3000:])
    if result.stderr:
        print("STDERR:", result.stderr[-2000:])

    assert result.returncode == 0, (
        f"Build failed for {model_id} with exit code {result.returncode}.\n"
        f"STDOUT (last 1000 chars):\n{result.stdout[-1000:]}\n"
        f"STDERR (last 1000 chars):\n{result.stderr[-1000:]}"
    )

    return family, model_id, output_dir


# ---------------------------------------------------------------------------
# Copy logic tests
# ---------------------------------------------------------------------------


class TestUpstreamFileCopy:
    """Verify _copy_upstream_auxiliary_files() across Granite model families."""

    def test_generation_config_copied(self, resolved_model, tmp_path):
        """generation_config.json should be copied for all families."""
        _family, _model_id, local_path = resolved_model
        src = Path(local_path)

        if not (src / "generation_config.json").exists():
            pytest.skip("upstream model has no generation_config.json")

        copied = _copy_upstream_auxiliary_files(local_path, str(tmp_path))
        assert "generation_config.json" in copied

    def test_upstream_has_chat_template(self, resolved_model):
        """Upstream tokenizer should have a chat template."""
        _family, _model_id, local_path = resolved_model
        src = Path(local_path)

        has_standalone = (src / "chat_template.jinja").exists()

        has_embedded = False
        tc_path = src / "tokenizer_config.json"
        if tc_path.exists():
            tc = json.loads(tc_path.read_text())
            has_embedded = bool(tc.get("chat_template"))

        assert has_standalone or has_embedded, (
            f"Upstream model {_model_id} has no chat template "
            f"(checked tokenizer_config.json and chat_template.jinja)"
        )

    def test_chat_template_survives_token_addition(self, resolved_model, tmp_path):
        """Chat template must be preserved verbatim after adding tokens and
        round-tripping through save_pretrained/from_pretrained."""
        from transformers import AutoTokenizer

        _family, _model_id, local_path = resolved_model

        upstream = AutoTokenizer.from_pretrained(local_path)
        upstream_template = upstream.chat_template
        if not upstream_template:
            pytest.skip(f"upstream model {_model_id} has no chat template")

        upstream.add_special_tokens({"additional_special_tokens": ["<|__test__|>"]})
        upstream.save_pretrained(str(tmp_path))

        reloaded = AutoTokenizer.from_pretrained(str(tmp_path))
        assert reloaded.chat_template == upstream_template, (
            f"Chat template changed after token addition + save round-trip "
            f"for {_model_id}.\n"
            f"Upstream (first 200 chars): {upstream_template[:200]}\n"
            f"Reloaded (first 200 chars): {reloaded.chat_template[:200] if reloaded.chat_template else '<None>'}"
        )

    def test_weight_files_excluded(self, resolved_model, tmp_path):
        """No weight files (.safetensors/.bin/.pt/.ckpt) should be copied."""
        _family, _model_id, local_path = resolved_model
        copied = _copy_upstream_auxiliary_files(local_path, str(tmp_path))

        for name in copied:
            ext = Path(name).suffix
            assert ext not in WEIGHT_EXTENSIONS, (
                f"Weight file '{name}' should not have been copied"
            )

    def test_config_json_excluded(self, resolved_model, tmp_path):
        """config.json must not be copied (replaced by GraniteSwitchConfig)."""
        _family, _model_id, local_path = resolved_model
        copied = _copy_upstream_auxiliary_files(local_path, str(tmp_path))
        assert "config.json" not in copied

    def test_dotfiles_excluded(self, resolved_model, tmp_path):
        """Dotfiles (.gitattributes etc.) must not be copied."""
        _family, _model_id, local_path = resolved_model
        copied = _copy_upstream_auxiliary_files(local_path, str(tmp_path))

        for name in copied:
            assert not name.startswith("."), (
                f"Dotfile '{name}' should not have been copied"
            )

    def test_no_unexpected_files(self, resolved_model, tmp_path):
        """Every copied file must exist in the source directory."""
        _family, _model_id, local_path = resolved_model
        src = Path(local_path)
        copied = _copy_upstream_auxiliary_files(local_path, str(tmp_path))

        for name in copied:
            assert (src / name).exists(), (
                f"Copied file '{name}' not found in source {local_path}"
            )
            assert (tmp_path / name).exists(), (
                f"Copied file '{name}' not found in output {tmp_path}"
            )


# ---------------------------------------------------------------------------
# Full build E2E tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.xdist_group("upstream_build_e2e")
class TestBuiltInBuildE2E:
    """Full build E2E for Granite models with --built-in-adapters."""

    def test_build_succeeds(self, built_in_build_output):
        """Build output directory should exist with safetensors."""
        _family, _model_id, output_dir = built_in_build_output
        safetensors = list(output_dir.glob("*.safetensors"))
        assert safetensors, "No .safetensors files found in build output"

    def test_generation_config_preserved(self, built_in_build_output):
        """generation_config.json should exist with eos/bos token IDs."""
        _family, _model_id, output_dir = built_in_build_output
        gen_config_path = output_dir / "generation_config.json"
        assert gen_config_path.exists(), "Missing generation_config.json"

        gen_config = json.loads(gen_config_path.read_text())
        assert "eos_token_id" in gen_config
        assert "bos_token_id" in gen_config

    def test_config_is_granite_switch(self, built_in_build_output):
        """config.json should have model_type=granite_switch and num_adapters=1."""
        _family, _model_id, output_dir = built_in_build_output
        config = json.loads((output_dir / "config.json").read_text())

        assert config["model_type"] == "granite_switch"
        assert config["num_adapters"] == 1

    def test_model_loads(self, built_in_build_output):
        """Model should load with device_map='meta' (no memory used)."""
        from transformers import AutoModelForCausalLM

        import granite_switch.hf  # noqa: F401

        _family, _model_id, output_dir = built_in_build_output
        model = AutoModelForCausalLM.from_pretrained(
            str(output_dir),
            device_map="meta",
        )

        assert model is not None
        assert model.config.model_type == "granite_switch"
        assert model.config.num_adapters == 1

    def test_tokenizer_loads(self, built_in_build_output):
        """Tokenizer should load and have control tokens."""
        from transformers import AutoTokenizer

        _family, _model_id, output_dir = built_in_build_output
        tokenizer = AutoTokenizer.from_pretrained(str(output_dir))
        assert tokenizer is not None

        config = json.loads((output_dir / "config.json").read_text())
        for token_id in config["adapter_token_ids"]:
            token = tokenizer.convert_ids_to_tokens(token_id)
            assert token is not None, (
                f"Adapter token ID {token_id} not in tokenizer vocabulary"
            )

    def test_chat_template_enriched(self, built_in_build_output):
        """Granite chat template should be enriched with adapter mappings."""
        from transformers import AutoTokenizer

        _family, model_id, output_dir = built_in_build_output

        upstream_tokenizer = AutoTokenizer.from_pretrained(model_id)
        upstream_template = upstream_tokenizer.chat_template
        if not upstream_template:
            pytest.skip(f"upstream model {model_id} has no chat template")

        output_tokenizer = AutoTokenizer.from_pretrained(str(output_dir))
        output_template = output_tokenizer.chat_template

        assert output_template, "Output tokenizer has no chat template"
        assert "adapter_map" in output_template, (
            "Granite output template missing adapter_map mapping"
        )
        assert "adapter_token" in output_template, (
            "Granite output template missing adapter_token lookup logic"
        )
