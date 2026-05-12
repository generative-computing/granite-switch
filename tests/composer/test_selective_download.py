# SPDX-License-Identifier: Apache-2.0
"""Tests for selective HuggingFace download behavior.

Covers the metadata-first path introduced to fix #3: adapter libraries are
filtered *before* download via ``allow_patterns`` derived from
``--base-model``, ``--include-adapters``, and ``--exclude-adapters``. HF Hub
calls (``list_repo_tree``, ``snapshot_download``) are mocked so tests run
offline.
"""

from unittest.mock import patch, MagicMock

import pytest
from huggingface_hub.hf_api import RepoFile, RepoFolder
from huggingface_hub.errors import EntryNotFoundError

from granite_switch.composer.adapter_discovery import (
    _build_allow_patterns,
    _list_repo_adapter_names,
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
# _list_repo_adapter_names
# ---------------------------------------------------------------------------


class TestListRepoAdapterNames:
    def test_returns_folder_names_only(self):
        tree = _tree_response({
            "": ["answerability", "citations", "README.md", "config.json"],
        })
        with patch(
            "huggingface_hub.list_repo_tree",
            side_effect=tree,
        ):
            names = _list_repo_adapter_names("org/repo")
        assert names == ["answerability", "citations"]

    def test_skips_underscore_prefixed(self):
        tree = _tree_response({
            "": ["answerability", "_ollama", "_internal", "citations"],
        })
        with patch(
            "huggingface_hub.list_repo_tree",
            side_effect=tree,
        ):
            names = _list_repo_adapter_names("org/repo")
        assert names == ["answerability", "citations"]

    def test_empty_repo(self):
        tree = _tree_response({"": []})
        with patch(
            "huggingface_hub.list_repo_tree",
            side_effect=tree,
        ):
            assert _list_repo_adapter_names("org/repo") == []


# ---------------------------------------------------------------------------
# _resolve_technology
# ---------------------------------------------------------------------------


class TestResolveTechnology:
    def test_prefers_alora_when_both_exist(self):
        tree = _tree_response({
            "answerability/granite-4.1-3b": ["alora", "lora"],
        })
        with patch(
            "huggingface_hub.list_repo_tree",
            side_effect=tree,
        ):
            assert _resolve_technology(
                "org/repo", "answerability", "granite-4.1-3b"
            ) == "alora"

    def test_returns_lora_when_only_lora(self):
        tree = _tree_response({
            "citations/granite-4.1-3b": ["lora"],
        })
        with patch(
            "huggingface_hub.list_repo_tree",
            side_effect=tree,
        ):
            assert _resolve_technology(
                "org/repo", "citations", "granite-4.1-3b"
            ) == "lora"

    def test_returns_alora_when_only_alora(self):
        tree = _tree_response({
            "answerability/granite-4.1-3b": ["alora"],
        })
        with patch(
            "huggingface_hub.list_repo_tree",
            side_effect=tree,
        ):
            assert _resolve_technology(
                "org/repo", "answerability", "granite-4.1-3b"
            ) == "alora"

    def test_returns_none_when_target_model_missing(self):
        # The adapter/model path does not exist in the repo
        tree = _tree_response({})
        with patch(
            "huggingface_hub.list_repo_tree",
            side_effect=tree,
        ):
            assert _resolve_technology(
                "org/repo", "answerability", "granite-99b"
            ) is None

    def test_returns_none_when_no_technology_dirs(self):
        # Model dir exists but contains unexpected entries
        tree = _tree_response({
            "answerability/granite-4.1-3b": ["other"],
        })
        with patch(
            "huggingface_hub.list_repo_tree",
            side_effect=tree,
        ):
            assert _resolve_technology(
                "org/repo", "answerability", "granite-4.1-3b"
            ) is None


# ---------------------------------------------------------------------------
# _build_allow_patterns
# ---------------------------------------------------------------------------


class TestBuildAllowPatterns:
    def _default_tree(self):
        """Typical three-adapter library tree for mocking."""
        return _tree_response({
            "": ["answerability", "citations", "query_rewrite"],
            "answerability/granite-4.1-3b": ["alora", "lora"],
            "citations/granite-4.1-3b": ["lora"],
            "query_rewrite/granite-4.1-3b": ["alora"],
        })

    def test_all_adapters_with_target_model(self):
        with patch(
            "huggingface_hub.list_repo_tree",
            side_effect=self._default_tree(),
        ):
            patterns = _build_allow_patterns(
                "org/repo", target_model_name="granite-4.1-3b",
            )
        assert patterns == [
            "answerability/granite-4.1-3b/alora/**",
            "citations/granite-4.1-3b/lora/**",
            "query_rewrite/granite-4.1-3b/alora/**",
        ]

    def test_include_filter_applies(self):
        with patch(
            "huggingface_hub.list_repo_tree",
            side_effect=self._default_tree(),
        ):
            patterns = _build_allow_patterns(
                "org/repo",
                target_model_name="granite-4.1-3b",
                include_adapters=["answerability"],
            )
        assert patterns == ["answerability/granite-4.1-3b/alora/**"]

    def test_include_filter_supports_fnmatch_glob(self):
        with patch(
            "huggingface_hub.list_repo_tree",
            side_effect=self._default_tree(),
        ):
            patterns = _build_allow_patterns(
                "org/repo",
                target_model_name="granite-4.1-3b",
                include_adapters=["query_*"],
            )
        assert patterns == ["query_rewrite/granite-4.1-3b/alora/**"]

    def test_exclude_filter_applies(self):
        with patch(
            "huggingface_hub.list_repo_tree",
            side_effect=self._default_tree(),
        ):
            patterns = _build_allow_patterns(
                "org/repo",
                target_model_name="granite-4.1-3b",
                exclude_adapters=["citations"],
            )
        assert patterns == [
            "answerability/granite-4.1-3b/alora/**",
            "query_rewrite/granite-4.1-3b/alora/**",
        ]

    def test_include_and_exclude_combined(self):
        # include keeps {answerability, citations}, then exclude drops citations
        with patch(
            "huggingface_hub.list_repo_tree",
            side_effect=self._default_tree(),
        ):
            patterns = _build_allow_patterns(
                "org/repo",
                target_model_name="granite-4.1-3b",
                include_adapters=["answerability", "citations"],
                exclude_adapters=["citations"],
            )
        assert patterns == ["answerability/granite-4.1-3b/alora/**"]

    def test_target_model_missing_keeps_broad_pattern(self):
        # Adapter directory exists at top level, but target model dir
        # doesn't exist → pattern has no tech segment so discovery can
        # report it downstream.
        tree = _tree_response({
            "": ["answerability"],
            # Note: no entry for "answerability/granite-4.1-3b"
        })
        with patch(
            "huggingface_hub.list_repo_tree",
            side_effect=tree,
        ):
            patterns = _build_allow_patterns(
                "org/repo", target_model_name="granite-4.1-3b",
            )
        assert patterns == ["answerability/granite-4.1-3b/**"]

    def test_no_target_model_falls_back_to_full_adapter_dirs(self):
        tree = _tree_response({
            "": ["answerability", "citations"],
        })
        with patch(
            "huggingface_hub.list_repo_tree",
            side_effect=tree,
        ):
            patterns = _build_allow_patterns(
                "org/repo", target_model_name=None,
            )
        assert patterns == ["answerability/**", "citations/**"]

    def test_include_drops_everything_falls_back_to_target_model_glob(self):
        # When include filter removes every adapter but target_model_name
        # is still set, the builder falls back to ``*/<model>/**``. This
        # is a known edge case: downstream ``filter_adapters`` will still
        # drop unmatched names, but the snapshot download is broader than
        # strictly necessary.
        with patch(
            "huggingface_hub.list_repo_tree",
            side_effect=self._default_tree(),
        ):
            patterns = _build_allow_patterns(
                "org/repo",
                target_model_name="granite-4.1-3b",
                include_adapters=["nonexistent"],
            )
        assert patterns == ["*/granite-4.1-3b/**"]

    def test_empty_repo_returns_none(self):
        tree = _tree_response({"": []})
        with patch(
            "huggingface_hub.list_repo_tree",
            side_effect=tree,
        ):
            patterns = _build_allow_patterns(
                "org/repo", target_model_name=None,
            )
        assert patterns is None


# ---------------------------------------------------------------------------
# list_repo_adapters_remote
# ---------------------------------------------------------------------------


class TestListRepoAdaptersRemote:
    def test_returns_sorted_adapters_with_technologies(self):
        tree = _tree_response({
            "": ["zeta", "alpha"],
            "zeta/granite-4.1-3b": ["alora"],
            "alpha/granite-4.1-3b": ["alora", "lora"],
        })
        with patch(
            "huggingface_hub.list_repo_tree",
            side_effect=tree,
        ):
            result = list_repo_adapters_remote("org/repo", "granite-4.1-3b")
        assert result == [
            {"name": "alpha", "technologies": ["alora", "lora"]},
            {"name": "zeta", "technologies": ["alora"]},
        ]

    def test_skips_adapters_missing_target_model(self):
        tree = _tree_response({
            "": ["answerability", "stale_adapter"],
            "answerability/granite-4.1-3b": ["alora"],
            # stale_adapter/granite-4.1-3b is missing — should be skipped
        })
        with patch(
            "huggingface_hub.list_repo_tree",
            side_effect=tree,
        ):
            result = list_repo_adapters_remote("org/repo", "granite-4.1-3b")
        assert result == [{"name": "answerability", "technologies": ["alora"]}]

    def test_skips_adapter_with_no_known_technologies(self):
        # Adapter has a target-model dir but no alora/lora subdirs inside
        tree = _tree_response({
            "": ["weird_adapter"],
            "weird_adapter/granite-4.1-3b": ["experimental"],
        })
        with patch(
            "huggingface_hub.list_repo_tree",
            side_effect=tree,
        ):
            result = list_repo_adapters_remote("org/repo", "granite-4.1-3b")
        assert result == []


# ---------------------------------------------------------------------------
# resolve_repo_path — integration with snapshot_download
# ---------------------------------------------------------------------------


class TestResolveRepoPathSelectiveDownload:
    def test_local_path_returns_as_is_without_download(self, tmp_path):
        (tmp_path / "adapter_config.json").write_text("{}")
        with patch(
            "huggingface_hub.snapshot_download"
        ) as mock_dl:
            result = resolve_repo_path(
                str(tmp_path),
                target_model_name="granite-4.1-3b",
                include_adapters=["anything"],
            )
        mock_dl.assert_not_called()
        assert result == str(tmp_path)

    def test_hf_repo_passes_allow_patterns(self, tmp_path):
        tree = _tree_response({
            "": ["answerability", "citations"],
            "answerability/granite-4.1-3b": ["alora"],
            "citations/granite-4.1-3b": ["lora"],
        })
        mock_dl = MagicMock(return_value=str(tmp_path))
        with patch(
            "huggingface_hub.list_repo_tree",
            side_effect=tree,
        ), patch(
            "huggingface_hub.snapshot_download", mock_dl,
        ):
            resolve_repo_path(
                "org/repo", target_model_name="granite-4.1-3b",
            )
        mock_dl.assert_called_once()
        kwargs = mock_dl.call_args.kwargs
        assert kwargs["repo_id"] == "org/repo"
        assert kwargs["repo_type"] == "model"
        assert kwargs["allow_patterns"] == [
            "answerability/granite-4.1-3b/alora/**",
            "citations/granite-4.1-3b/lora/**",
        ]

    def test_hf_repo_passes_include_and_exclude_through(self, tmp_path):
        tree = _tree_response({
            "": ["answerability", "citations", "query_rewrite"],
            "answerability/granite-4.1-3b": ["alora"],
            "citations/granite-4.1-3b": ["lora"],
            "query_rewrite/granite-4.1-3b": ["alora"],
        })
        mock_dl = MagicMock(return_value=str(tmp_path))
        with patch(
            "huggingface_hub.list_repo_tree",
            side_effect=tree,
        ), patch(
            "huggingface_hub.snapshot_download", mock_dl,
        ):
            resolve_repo_path(
                "org/repo",
                target_model_name="granite-4.1-3b",
                include_adapters=["answerability", "query_rewrite"],
                exclude_adapters=["query_rewrite"],
            )
        kwargs = mock_dl.call_args.kwargs
        assert kwargs["allow_patterns"] == [
            "answerability/granite-4.1-3b/alora/**",
        ]

    def test_hf_repo_without_filters_downloads_full(self, tmp_path):
        """No filters → allow_patterns not passed (matches pre-fix behavior)."""
        mock_dl = MagicMock(return_value=str(tmp_path))
        with patch("huggingface_hub.snapshot_download", mock_dl):
            resolve_repo_path("org/repo")
        kwargs = mock_dl.call_args.kwargs
        assert "allow_patterns" not in kwargs

    def test_pattern_build_failure_falls_back_to_full_download(self, tmp_path):
        """If metadata pass raises, warn and continue with a full download."""
        def _boom(*args, **kwargs):
            raise RuntimeError("HF Hub down")

        mock_dl = MagicMock(return_value=str(tmp_path))
        with patch(
            "huggingface_hub.list_repo_tree",
            side_effect=_boom,
        ), patch(
            "huggingface_hub.snapshot_download", mock_dl,
        ):
            resolve_repo_path(
                "org/repo", target_model_name="granite-4.1-3b",
            )
        # Fell back to a full snapshot_download (no allow_patterns).
        kwargs = mock_dl.call_args.kwargs
        assert "allow_patterns" not in kwargs

    def test_nonexistent_path_without_slash_raises(self):
        with pytest.raises(ValueError, match="doesn't appear to be a HuggingFace repo"):
            resolve_repo_path("not-a-repo-or-path")


# ---------------------------------------------------------------------------
# End-to-end: exact scenario from issue #3
# ---------------------------------------------------------------------------


class TestIssue3ReproScenario:
    """Mirrors the "Steps to reproduce" from issue #3 verbatim.

    Invocation::

        python -m granite_switch.composer.compose_granite_switch \\
          --base-model ibm-granite/granite-4.1-3b \\
          --adapters ibm-granite/granitelib-core-r1.0 \\
                     ibm-granite/granitelib-rag-r1.0 \\
          --include-adapters query_rewrite context-attribution

    Issue's "Expected":
        Only files under ``query_rewrite/granite-4.1-3b/`` and
        ``context-attribution/granite-4.1-3b/`` are downloaded from each repo.

    This test verifies that the two ``snapshot_download`` calls carry
    ``allow_patterns`` restricted to exactly those adapter/model paths —
    no 8b/30b variants, and only the correct technology per adapter.
    """

    def _core_tree(self):
        # granitelib-core-r1.0 hosts context-attribution (lora, per BUILD.md)
        # plus other adapters for multiple model sizes.
        return _tree_response({
            "": ["context-attribution", "requirement-check", "uncertainty"],
            "context-attribution/granite-4.1-3b": ["lora"],
            "context-attribution/granite-4.1-8b": ["lora"],
            "requirement-check/granite-4.1-3b": ["alora"],
            "uncertainty/granite-4.1-3b": ["alora"],
        })

    def _rag_tree(self):
        # granitelib-rag-r1.0 hosts query_rewrite (alora) plus others.
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

    def test_only_requested_adapter_model_paths_are_downloaded(self, tmp_path):
        target_model = "granite-4.1-3b"
        include = ["query_rewrite", "context-attribution"]

        mock_dl = MagicMock(return_value=str(tmp_path))

        # --- Call 1: granitelib-core-r1.0 ---
        with patch(
            "huggingface_hub.list_repo_tree",
            side_effect=self._core_tree(),
        ), patch("huggingface_hub.snapshot_download", mock_dl):
            resolve_repo_path(
                "ibm-granite/granitelib-core-r1.0",
                target_model_name=target_model,
                include_adapters=include,
            )

        core_kwargs = mock_dl.call_args.kwargs
        assert core_kwargs["repo_id"] == "ibm-granite/granitelib-core-r1.0"
        # Only context-attribution/granite-4.1-3b/lora — no 8b, no other adapters
        assert core_kwargs["allow_patterns"] == [
            "context-attribution/granite-4.1-3b/lora/**",
        ]

        # --- Call 2: granitelib-rag-r1.0 ---
        mock_dl.reset_mock()
        with patch(
            "huggingface_hub.list_repo_tree",
            side_effect=self._rag_tree(),
        ), patch("huggingface_hub.snapshot_download", mock_dl):
            resolve_repo_path(
                "ibm-granite/granitelib-rag-r1.0",
                target_model_name=target_model,
                include_adapters=include,
            )

        rag_kwargs = mock_dl.call_args.kwargs
        assert rag_kwargs["repo_id"] == "ibm-granite/granitelib-rag-r1.0"
        # Only query_rewrite/granite-4.1-3b/alora — no 8b, no other adapters,
        # and only alora (not lora)
        assert rag_kwargs["allow_patterns"] == [
            "query_rewrite/granite-4.1-3b/alora/**",
        ]


# ---------------------------------------------------------------------------
# Real HuggingFace Hub metadata calls (no file downloads)
# ---------------------------------------------------------------------------
#
# These tests hit the real HF Hub API via ``list_repo_tree`` to verify that
# the helpers work against actual repo layouts — not just our mock model of
# them. ``snapshot_download`` is still mocked so nothing heavy is pulled to
# disk (metadata calls are cheap: tens of KB each). Marked ``slow`` because
# they require network and can be rate-limited; skip with ``-m "not slow"``.


@pytest.mark.slow
class TestRealHubMetadata:
    REPO = "ibm-granite/granitelib-core-r1.0"
    TARGET_MODEL = "granite-4.1-3b"
    # Adapters known to exist in granitelib-core-r1.0 for granite-4.1-3b
    # (source: BUILD.md of the published ibm-granite/granite-switch-4.1-3b-preview).
    # This set may grow as IBM adds adapters — we assert containment, not equality.
    KNOWN_ADAPTERS = {"context-attribution", "requirement-check", "uncertainty"}

    def test_list_repo_adapter_names_against_real_repo(self):
        names = _list_repo_adapter_names(self.REPO)
        # All known adapters should be present
        assert self.KNOWN_ADAPTERS.issubset(set(names)), (
            f"Expected adapters {self.KNOWN_ADAPTERS} not found. Got: {names}"
        )
        # No underscore-prefixed folders (e.g., _ollama) should leak through
        assert all(not n.startswith("_") for n in names)

    def test_resolve_technology_matches_published_build(self):
        # context-attribution is documented as 'lora' in the published BUILD.md
        tech = _resolve_technology(
            self.REPO, "context-attribution", self.TARGET_MODEL,
        )
        assert tech == "lora"

        # requirement-check is documented as 'alora'
        tech = _resolve_technology(
            self.REPO, "requirement-check", self.TARGET_MODEL,
        )
        assert tech == "alora"

    def test_resolve_technology_returns_none_for_nonexistent_target(self):
        # No granite-99b exists in the real repo
        tech = _resolve_technology(
            self.REPO, "context-attribution", "granite-99b",
        )
        assert tech is None

    def test_list_repo_adapters_remote_includes_known_adapters(self):
        result = list_repo_adapters_remote(self.REPO, self.TARGET_MODEL)
        names = {entry["name"] for entry in result}
        assert self.KNOWN_ADAPTERS.issubset(names), (
            f"Missing adapters. Expected ⊇ {self.KNOWN_ADAPTERS}, got {names}"
        )
        # Each entry must list at least one technology
        for entry in result:
            assert entry["technologies"], (
                f"{entry['name']} has no technologies"
            )
            assert all(
                t in ("alora", "lora") for t in entry["technologies"]
            ), f"Unknown tech for {entry['name']}: {entry['technologies']}"

    def test_build_allow_patterns_against_real_repo(self, tmp_path):
        # Construct patterns from the real repo, mock snapshot_download so
        # no weights are actually fetched.
        mock_dl = MagicMock(return_value=str(tmp_path))
        with patch("huggingface_hub.snapshot_download", mock_dl):
            resolve_repo_path(
                self.REPO,
                target_model_name=self.TARGET_MODEL,
                include_adapters=["context-attribution"],
            )

        kwargs = mock_dl.call_args.kwargs
        patterns = kwargs["allow_patterns"]
        # Exactly one pattern, scoped to context-attribution/3b/lora
        assert patterns == ["context-attribution/granite-4.1-3b/lora/**"]
