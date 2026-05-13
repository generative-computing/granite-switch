# SPDX-License-Identifier: Apache-2.0
"""Tests for selective HuggingFace download behavior.

Covers the metadata-first path introduced to fix #3: adapter libraries are
filtered *before* download via ``allow_patterns`` derived from
``--base-model``, ``--include-adapters``, and ``--exclude-adapters``. HF Hub
calls (``list_repo_tree``, ``snapshot_download``) are mocked so tests run
offline, except the ``TestRealHubMetadata`` class which hits the real Hub.
"""

from unittest.mock import patch, MagicMock

import pytest
from huggingface_hub.hf_api import RepoFile, RepoFolder
from huggingface_hub.errors import EntryNotFoundError

from granite_switch.composer.adapter_discovery import (
    _build_allow_patterns,
    _resolve_technology,
    list_repo_adapters_remote,
    resolve_repo_path,
)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _folder(path):
    return RepoFolder(path=path, oid="deadbeef")


def _file(path):
    return RepoFile(path=path, oid="deadbeef", size=1)


def _tree_response(tree_map):
    """Return a ``list_repo_tree`` side_effect that serves *tree_map*.

    *tree_map* maps ``path_in_repo`` (``""`` for root) to the list of
    folder/file names that live at that path. Unknown paths raise
    ``EntryNotFoundError`` — matching real HF behavior for missing folders.
    """
    def _side_effect(repo_id, repo_type="model", path_in_repo=None):
        key = path_in_repo or ""
        if key not in tree_map:
            raise EntryNotFoundError(f"Path not found: {key}")
        entries = []
        for entry in tree_map[key]:
            full_path = f"{key}/{entry}" if key else entry
            if entry.endswith(".txt") or entry.endswith(".md") or entry.endswith(".json"):
                entries.append(_file(full_path))
            else:
                entries.append(_folder(full_path))
        return entries
    return _side_effect


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_tree():
    """Three-adapter library with an alora/lora mix for granite-4.1-3b."""
    return _tree_response({
        "": ["answerability", "citations", "query_rewrite"],
        "answerability/granite-4.1-3b": ["alora", "lora"],
        "citations/granite-4.1-3b": ["lora"],
        "query_rewrite/granite-4.1-3b": ["alora"],
    })


@pytest.fixture
def core_tree():
    """ibm-granite/granitelib-core-r1.0-like layout with an 8b variant."""
    return _tree_response({
        "": ["context-attribution", "requirement-check", "uncertainty"],
        "context-attribution/granite-4.1-3b": ["lora"],
        "context-attribution/granite-4.1-8b": ["lora"],
        "requirement-check/granite-4.1-3b": ["alora"],
        "uncertainty/granite-4.1-3b": ["alora"],
    })


@pytest.fixture
def rag_tree():
    """ibm-granite/granitelib-rag-r1.0-like layout with an 8b variant."""
    return _tree_response({
        "": [
            "query_rewrite", "answerability",
            "citations", "hallucination_detection",
        ],
        "query_rewrite/granite-4.1-3b": ["alora"],
        "query_rewrite/granite-4.1-8b": ["alora"],
        "answerability/granite-4.1-3b": ["alora"],
        "citations/granite-4.1-3b": ["lora"],
        "hallucination_detection/granite-4.1-3b": ["lora"],
    })


# ---------------------------------------------------------------------------
# _resolve_technology — alora is preferred over lora
# ---------------------------------------------------------------------------


class TestResolveTechnology:
    def test_prefers_alora_when_both_exist(self):
        tree = _tree_response({
            "answerability/granite-4.1-3b": ["alora", "lora"],
        })
        with patch("huggingface_hub.list_repo_tree", side_effect=tree):
            assert _resolve_technology(
                "org/repo", "answerability", "granite-4.1-3b"
            ) == "alora"

    def test_returns_none_when_target_model_missing(self):
        # Adapter exists, but only for a different model size.
        tree = _tree_response({
            "answerability/granite-4.1-8b": ["alora"],
        })
        with patch("huggingface_hub.list_repo_tree", side_effect=tree):
            assert _resolve_technology(
                "org/repo", "answerability", "granite-4.1-3b"
            ) is None


# ---------------------------------------------------------------------------
# _build_allow_patterns — pattern construction from filters
# ---------------------------------------------------------------------------


class TestBuildAllowPatterns:
    def test_all_adapters_with_target_model(self, default_tree):
        with patch("huggingface_hub.list_repo_tree", side_effect=default_tree):
            patterns = _build_allow_patterns(
                "org/repo", target_model_name="granite-4.1-3b",
            )
        assert patterns == [
            "answerability/granite-4.1-3b/alora/**",
            "citations/granite-4.1-3b/lora/**",
            "query_rewrite/granite-4.1-3b/alora/**",
        ]

    def test_include_filter_applies(self, default_tree):
        with patch("huggingface_hub.list_repo_tree", side_effect=default_tree):
            patterns = _build_allow_patterns(
                "org/repo",
                target_model_name="granite-4.1-3b",
                include_adapters=["answerability"],
            )
        assert patterns == ["answerability/granite-4.1-3b/alora/**"]

    def test_technology_filter_overrides_alora_preference(self, default_tree):
        """``technology_filter='lora'`` must force lora paths even for
        adapters that have an alora variant — otherwise discovery (which
        also respects the filter) drops them and the final model is missing
        adapters."""
        with patch("huggingface_hub.list_repo_tree", side_effect=default_tree):
            patterns = _build_allow_patterns(
                "org/repo",
                target_model_name="granite-4.1-3b",
                technology_filter="lora",
            )
        assert patterns == [
            "answerability/granite-4.1-3b/lora/**",
            "citations/granite-4.1-3b/lora/**",
            "query_rewrite/granite-4.1-3b/lora/**",
        ]


# ---------------------------------------------------------------------------
# resolve_repo_path — integration with snapshot_download
# ---------------------------------------------------------------------------


class TestResolveRepoPathSelectiveDownload:
    def test_local_path_returns_as_is_without_download(self, tmp_path):
        (tmp_path / "adapter_config.json").write_text("{}")
        with patch("huggingface_hub.snapshot_download") as mock_dl:
            result = resolve_repo_path(
                str(tmp_path),
                target_model_name="granite-4.1-3b",
                include_adapters=["anything"],
            )
        mock_dl.assert_not_called()
        assert result == str(tmp_path)

    def test_hf_repo_passes_allow_patterns(self, tmp_path, default_tree):
        mock_dl = MagicMock(return_value=str(tmp_path))
        with patch(
            "huggingface_hub.list_repo_tree", side_effect=default_tree,
        ), patch("huggingface_hub.snapshot_download", mock_dl):
            resolve_repo_path(
                "org/repo", target_model_name="granite-4.1-3b",
            )
        assert mock_dl.call_args.kwargs["allow_patterns"] == [
            "answerability/granite-4.1-3b/alora/**",
            "citations/granite-4.1-3b/lora/**",
            "query_rewrite/granite-4.1-3b/alora/**",
        ]

    def test_hf_repo_without_filters_downloads_full(self, tmp_path):
        """No filters → allow_patterns not passed (matches pre-fix behavior)."""
        mock_dl = MagicMock(return_value=str(tmp_path))
        with patch("huggingface_hub.snapshot_download", mock_dl):
            resolve_repo_path("org/repo")
        mock_dl.assert_called_once()
        assert "allow_patterns" not in mock_dl.call_args.kwargs

    def test_pattern_build_failure_falls_back_to_full_download(self, tmp_path):
        """If metadata pass raises, warn and continue with a full download."""
        def _boom(*args, **kwargs):
            raise RuntimeError("HF Hub down")

        mock_dl = MagicMock(return_value=str(tmp_path))
        with patch(
            "huggingface_hub.list_repo_tree", side_effect=_boom,
        ), patch("huggingface_hub.snapshot_download", mock_dl):
            resolve_repo_path(
                "org/repo", target_model_name="granite-4.1-3b",
            )
        assert "allow_patterns" not in mock_dl.call_args.kwargs

    def test_shared_include_filter_across_repos_downloads_disjoint_subsets(
        self, tmp_path, core_tree, rag_tree,
    ):
        """Issue #3 scenario: the same ``--include-adapters`` applied to two
        repos downloads only each repo's matching subset (no 8b variants, and
        the correct technology per adapter)."""
        target_model = "granite-4.1-3b"
        include = ["query_rewrite", "context-attribution"]

        mock_dl = MagicMock(return_value=str(tmp_path))

        with patch(
            "huggingface_hub.list_repo_tree", side_effect=core_tree,
        ), patch("huggingface_hub.snapshot_download", mock_dl):
            resolve_repo_path(
                "ibm-granite/granitelib-core-r1.0",
                target_model_name=target_model,
                include_adapters=include,
            )
        assert mock_dl.call_args.kwargs["allow_patterns"] == [
            "context-attribution/granite-4.1-3b/lora/**",
        ]

        mock_dl.reset_mock()
        with patch(
            "huggingface_hub.list_repo_tree", side_effect=rag_tree,
        ), patch("huggingface_hub.snapshot_download", mock_dl):
            resolve_repo_path(
                "ibm-granite/granitelib-rag-r1.0",
                target_model_name=target_model,
                include_adapters=include,
            )
        assert mock_dl.call_args.kwargs["allow_patterns"] == [
            "query_rewrite/granite-4.1-3b/alora/**",
        ]


# ---------------------------------------------------------------------------
# Real HuggingFace Hub metadata calls (no file downloads)
# ---------------------------------------------------------------------------
#
# Uses the real HF Hub API via ``list_repo_tree`` to verify the helpers
# against an actual repo layout. ``snapshot_download`` is still mocked so
# nothing heavy is pulled to disk. Marked ``slow`` because they require
# network; skip with ``-m "not slow"``.


@pytest.mark.slow
class TestRealHubMetadata:
    REPO = "ibm-granite/granitelib-core-r1.0"
    TARGET_MODEL = "granite-4.1-3b"

    def test_resolve_technology_matches_published_build(self):
        # context-attribution is documented as 'lora' in the published BUILD.md;
        # requirement-check as 'alora'.
        assert _resolve_technology(
            self.REPO, "context-attribution", self.TARGET_MODEL,
        ) == "lora"
        assert _resolve_technology(
            self.REPO, "requirement-check", self.TARGET_MODEL,
        ) == "alora"

    def test_list_repo_adapters_remote_includes_known_adapters(self):
        known = {"context-attribution", "requirement-check", "uncertainty"}
        result = list_repo_adapters_remote(self.REPO, self.TARGET_MODEL)
        names = {entry["name"] for entry in result}
        assert known.issubset(names), (
            f"Missing adapters. Expected ⊇ {known}, got {names}"
        )

    def test_build_allow_patterns_against_real_repo(self, tmp_path):
        mock_dl = MagicMock(return_value=str(tmp_path))
        with patch("huggingface_hub.snapshot_download", mock_dl):
            resolve_repo_path(
                self.REPO,
                target_model_name=self.TARGET_MODEL,
                include_adapters=["context-attribution"],
            )
        assert mock_dl.call_args.kwargs["allow_patterns"] == [
            "context-attribution/granite-4.1-3b/lora/**",
        ]
