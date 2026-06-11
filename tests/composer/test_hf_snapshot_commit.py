# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``_extract_hf_snapshot_commit``.

The helper gates commit-SHA extraction on the adapter path being contained
under ``huggingface_hub.constants.HF_HUB_CACHE``. Tests monkeypatch that
constant to a controlled temp directory and build realistic HF-style cache
paths under it.
"""

import os

import pytest

from granite_switch.composer.compose_granite_switch import (
    _extract_hf_snapshot_commit,
)

pytestmark = pytest.mark.local_fast


VALID_SHA = "6e4a75e35f1cb272e8d15b4615fb0a123398d1cf"
SHORT_SHA = VALID_SHA[:8]


def _patch_hf_cache(monkeypatch, cache_root):
    """Point huggingface_hub.constants.HF_HUB_CACHE at *cache_root*.

    The helper imports HF_HUB_CACHE lazily, so we patch the attribute on the
    already-imported module.
    """
    from huggingface_hub import constants
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache_root))


class TestExtractHfSnapshotCommit:
    def test_none_input(self):
        assert _extract_hf_snapshot_commit(None) is None

    def test_empty_string(self):
        assert _extract_hf_snapshot_commit("") is None

    def test_real_hf_cache_path(self, tmp_path, monkeypatch):
        cache = tmp_path / "hf_cache"
        adapter = (
            cache
            / "models--ibm-granite--granite-lib-rag-r1.0"
            / "snapshots"
            / VALID_SHA
            / "answerability"
            / "granite-4.1-3b"
            / "alora"
        )
        adapter.mkdir(parents=True)
        _patch_hf_cache(monkeypatch, cache)
        assert _extract_hf_snapshot_commit(str(adapter)) == VALID_SHA

    def test_path_outside_hf_cache_returns_none(self, tmp_path, monkeypatch):
        # Deliberately construct a decoy path with a snapshots/<sha> segment
        # that lives OUTSIDE the configured HF cache. Should be rejected.
        cache = tmp_path / "hf_cache"
        cache.mkdir()
        decoy = (
            tmp_path
            / "my_local_adapters"
            / "some-lib"
            / "snapshots"
            / VALID_SHA
            / "adapter"
        )
        decoy.mkdir(parents=True)
        _patch_hf_cache(monkeypatch, cache)
        assert _extract_hf_snapshot_commit(str(decoy)) is None

    def test_hf_cache_but_no_snapshot_segment(self, tmp_path, monkeypatch):
        cache = tmp_path / "hf_cache"
        weird = cache / "models--foo--bar" / "blobs" / "deadbeef"
        weird.mkdir(parents=True)
        _patch_hf_cache(monkeypatch, cache)
        assert _extract_hf_snapshot_commit(str(weird)) is None

    def test_hf_cache_but_invalid_sha_segment(self, tmp_path, monkeypatch):
        # snapshots/<not-40-hex> must be rejected
        cache = tmp_path / "hf_cache"
        bad = cache / "models--foo--bar" / "snapshots" / "not-a-sha" / "file"
        bad.mkdir(parents=True)
        _patch_hf_cache(monkeypatch, cache)
        assert _extract_hf_snapshot_commit(str(bad)) is None

    def test_hf_cache_but_sha_wrong_length(self, tmp_path, monkeypatch):
        # 39 hex chars (one short)
        cache = tmp_path / "hf_cache"
        short = cache / "models--foo--bar" / "snapshots" / ("a" * 39) / "file"
        short.mkdir(parents=True)
        _patch_hf_cache(monkeypatch, cache)
        assert _extract_hf_snapshot_commit(str(short)) is None

    def test_returns_full_sha_not_short(self, tmp_path, monkeypatch):
        cache = tmp_path / "hf_cache"
        adapter = cache / "models--a--b" / "snapshots" / VALID_SHA / "x"
        adapter.mkdir(parents=True)
        _patch_hf_cache(monkeypatch, cache)
        result = _extract_hf_snapshot_commit(str(adapter))
        assert result == VALID_SHA
        assert len(result) == 40

    def test_sha_with_uppercase_hex_rejected(self, tmp_path, monkeypatch):
        # HF uses lowercase hex; uppercase shouldn't match the validator
        cache = tmp_path / "hf_cache"
        upper_sha = VALID_SHA.upper()
        adapter = cache / "models--a--b" / "snapshots" / upper_sha / "x"
        adapter.mkdir(parents=True)
        _patch_hf_cache(monkeypatch, cache)
        assert _extract_hf_snapshot_commit(str(adapter)) is None
