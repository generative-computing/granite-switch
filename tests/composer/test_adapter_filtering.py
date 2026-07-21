# SPDX-License-Identifier: Apache-2.0
"""Unit tests for adapter filtering and listing functions."""

import json

import pytest

from granite_switch.composer.adapter_discovery import (
    discover_adapters,
    filter_adapters,
    list_available_adapters,
)
from granite_switch.composer.arch import resolve_arch

# -- Fixtures ----------------------------------------------------------------


@pytest.fixture
def sample_discovered():
    """Simulates output of discover_adapters(): list of (path, name, tech)."""
    return [
        ("/adapters/answerability/alora", "answerability", "alora"),
        ("/adapters/citations/alora", "citations", "alora"),
        ("/adapters/context_relevance/alora", "context_relevance", "alora"),
        ("/adapters/hallucination_detection/alora", "hallucination_detection", "alora"),
        ("/adapters/query_rewrite/alora", "query_rewrite", "alora"),
        ("/adapters/query_clarification/alora", "query_clarification", "alora"),
    ]


@pytest.fixture
def mixed_tech_discovered():
    """Discovered adapters with mixed technologies (after tech_filter skipped)."""
    return [
        ("/adapters/adapt_a/alora", "adapt_a", "alora"),
        ("/adapters/adapt_b/lora", "adapt_b", "lora"),
        ("/adapters/adapt_c/alora", "adapt_c", "alora"),
        ("/adapters/adapt_d/lora", "adapt_d", "lora"),
    ]


def _make_adapter_library(tmp_path, target_model, adapters):
    """Create a minimal adapter library directory structure.

    *adapters* is a list of ``(adapter_name, technologies)`` where
    *technologies* is a list of ``"alora"`` / ``"lora"`` strings.
    """
    for name, techs in adapters:
        for tech in techs:
            d = tmp_path / name / target_model / tech
            d.mkdir(parents=True)
            (d / "io.yaml").write_text(f"{name}-{tech}---\n")
            (d / "adapter_config.json").write_text(json.dumps({"r": 8}))
            (d / "adapter_model.safetensors").write_bytes(b"\x00")
    return str(tmp_path)


class TestTechnologyFilterAdapters:
    def test_test_prefer_alora(self, tmp_path):
        # Load base config early for arch resolution.
        arch = resolve_arch("ibm-granite/granite-4.0-micro")
        root = _make_adapter_library(
            tmp_path,
            "granite-4.0-micro",
            [
                ("answerability", ["alora", "lora"]),
                ("citations", ["alora"]),
            ],
        )
        adapters = discover_adapters(  # default is to prefer alora
            root,
            "granite-4.0-micro",
            arch,
            technology_fallback=None,
            technology_filter=None,
        )
        assert len(adapters) == 2
        assert len([found for found in adapters if found[2] == "alora"]) == 2
        adapters = discover_adapters(  # default is to prefer alora
            root,
            "granite-4.0-micro",
            arch,
            technology_fallback=None,
            technology_filter="lora",
        )
        assert len(adapters) == 1
        assert len([found for found in adapters if found[2] == "lora"]) == 1
        adapters = discover_adapters(
            root,
            "granite-4.0-micro",
            arch,
            technology_fallback="lora",
            technology_filter="lora",
        )
        print(adapters)
        assert len(adapters) == 1
        assert len([found for found in adapters if found[2] == "lora"]) == 1

    def test_filter_and_override(self, tmp_path):
        arch = resolve_arch("ibm-granite/granite-4.0-micro")
        root = _make_adapter_library(
            tmp_path,
            "granite-4.0-micro",
            [
                ("answerability", ["alora", "lora"]),
                ("citations", ["alora"]),
            ],
        )
        adapters = discover_adapters(  # default is to prefer alora
            root, "granite-4.0-micro", arch, technology_filter="lora"
        )
        assert len(adapters) == 1
        other_dir = tmp_path / "new_adapter" / "granite-4.0-micro" / "v3-prototype"
        other_dir.mkdir(parents=True)
        (other_dir / "io.yaml").write_text("source: v3-prototype-data")
        (other_dir / "adapter_config.json").write_text(json.dumps({"r": 16}))
        (other_dir / "adapter_model.safetensors").write_bytes(b"\x01")
        adapters = discover_adapters(
            root, "granite-4.0-micro", arch, technology_fallback="alora"
        )
        assert len(adapters) == 3
        adapters = discover_adapters(
            root,
            "granite-4.0-micro",
            arch,
            technology_fallback="alora",
            technology_filter="lora",
        )
        assert len(adapters) == 1


# -- filter_adapters tests ---------------------------------------------------


class TestFilterAdapters:
    def test_no_filters_passthrough(self, sample_discovered):
        result = filter_adapters(sample_discovered)
        assert result == sample_discovered

    def test_include_exact_name(self, sample_discovered):
        result = filter_adapters(sample_discovered, include=["answerability"])
        assert len(result) == 1
        assert result[0][1] == "answerability"

    def test_include_multiple_exact(self, sample_discovered):
        result = filter_adapters(
            sample_discovered, include=["answerability", "citations"]
        )
        names = [r[1] for r in result]
        assert names == ["answerability", "citations"]

    def test_include_glob_pattern(self, sample_discovered):
        result = filter_adapters(sample_discovered, include=["query_*"])
        names = [r[1] for r in result]
        assert set(names) == {"query_rewrite", "query_clarification"}

    def test_include_mixed_exact_and_glob(self, sample_discovered):
        result = filter_adapters(
            sample_discovered, include=["answerability", "query_*"]
        )
        names = [r[1] for r in result]
        assert "answerability" in names
        assert "query_rewrite" in names
        assert "query_clarification" in names
        assert len(names) == 3

    def test_exclude_exact_name(self, sample_discovered):
        result = filter_adapters(sample_discovered, exclude=["hallucination_detection"])
        names = [r[1] for r in result]
        assert "hallucination_detection" not in names
        assert len(names) == 5

    def test_exclude_glob_pattern(self, sample_discovered):
        result = filter_adapters(sample_discovered, exclude=["query_*"])
        names = [r[1] for r in result]
        assert "query_rewrite" not in names
        assert "query_clarification" not in names
        assert len(names) == 4

    def test_include_then_exclude(self, sample_discovered):
        result = filter_adapters(
            sample_discovered,
            include=["query_*", "answerability"],
            exclude=["query_clarification"],
        )
        names = [r[1] for r in result]
        assert set(names) == {"answerability", "query_rewrite"}

    def test_empty_result(self, sample_discovered):
        result = filter_adapters(sample_discovered, include=["nonexistent"])
        assert result == []

    def test_exclude_all(self, sample_discovered):
        result = filter_adapters(sample_discovered, exclude=["*"])
        assert result == []

    def test_warn_unmatched_include_pattern(self, sample_discovered, capsys):
        filter_adapters(sample_discovered, include=["no_match_*"])
        captured = capsys.readouterr()
        assert "no_match_*" in captured.out
        assert "matched nothing" in captured.out

    def test_preserves_tuple_structure(self, sample_discovered):
        result = filter_adapters(sample_discovered, include=["answerability"])
        assert result[0] == ("/adapters/answerability/alora", "answerability", "alora")

    def test_preserves_order(self, sample_discovered):
        result = filter_adapters(
            sample_discovered,
            include=["query_clarification", "answerability"],
        )
        names = [r[1] for r in result]
        assert names == ["answerability", "query_clarification"]


# -- list_available_adapters tests -------------------------------------------


class TestListAvailableAdapters:
    def test_lists_all_technologies(self, tmp_path):
        root = _make_adapter_library(
            tmp_path,
            "granite-4.0-micro",
            [
                ("answerability", ["alora", "lora"]),
                ("citations", ["alora"]),
            ],
        )
        result = list_available_adapters(root, "granite-4.0-micro")
        assert len(result) == 2
        ans = next(a for a in result if a["name"] == "answerability")
        assert ans["technologies"] == ["alora", "lora"]
        cit = next(a for a in result if a["name"] == "citations")
        assert cit["technologies"] == ["alora"]

    def test_filters_by_target_model(self, tmp_path):
        root = _make_adapter_library(
            tmp_path,
            "granite-4.0-micro",
            [
                ("answerability", ["alora"]),
            ],
        )
        _make_adapter_library(tmp_path, "granite-4.1-3b", [("other_adapter", ["lora"])])
        result = list_available_adapters(root, "granite-4.0-micro")
        names = [a["name"] for a in result]
        assert "answerability" in names
        assert "other_adapter" not in names

    def test_empty_library(self, tmp_path):
        result = list_available_adapters(str(tmp_path), "granite-4.0-micro")
        assert result == []

    def test_sorted_by_name(self, tmp_path):
        root = _make_adapter_library(
            tmp_path,
            "granite-4.0-micro",
            [
                ("zebra", ["alora"]),
                ("alpha", ["alora"]),
                ("middle", ["lora"]),
            ],
        )
        result = list_available_adapters(root, "granite-4.0-micro")
        names = [a["name"] for a in result]
        assert names == ["alpha", "middle", "zebra"]

    def test_skips_incomplete_adapters(self, tmp_path):
        root = _make_adapter_library(
            tmp_path,
            "granite-4.0-micro",
            [
                ("complete", ["alora"]),
            ],
        )
        incomplete_dir = tmp_path / "incomplete" / "granite-4.0-micro" / "alora"
        incomplete_dir.mkdir(parents=True)
        (incomplete_dir / "io.yaml").write_text("---\n")
        # Missing adapter_model.safetensors and adapter_config.json

        result = list_available_adapters(root, "granite-4.0-micro")
        names = [a["name"] for a in result]
        assert names == ["complete"]
