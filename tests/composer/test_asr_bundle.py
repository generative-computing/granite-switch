# SPDX-License-Identifier: Apache-2.0
"""Compose-side ASR selection and bundling (--asr-model/--asr-revision/--bundle-asr).

No network: bundling is exercised from a tiny local ASR directory built here,
the same path an airgapped compose takes.
"""

import pytest

from granite_switch.composer.compose_granite_switch import (
    _compose_argparser,
    _resolve_asr_model,
    bundle_asr_model,
)
from granite_switch.config import DEFAULT_ASR_MODEL_ID, DEFAULT_ASR_MODEL_REVISION
from tests.shared.tiny_asr import build_tiny_asr

pytestmark = pytest.mark.audio


def _args(*extra):
    return _compose_argparser().parse_args(["--adapters", "x", *extra])


class TestResolveAsrModel:
    def test_default_model_is_pinned(self):
        assert _resolve_asr_model(_args("--enable-audio")) == (
            DEFAULT_ASR_MODEL_ID,
            DEFAULT_ASR_MODEL_REVISION,
        )

    def test_explicit_revision_wins_on_default_model(self):
        args = _args("--enable-audio", "--asr-revision", "abc")
        assert _resolve_asr_model(args) == (DEFAULT_ASR_MODEL_ID, "abc")

    def test_custom_model_without_revision_warns_and_is_unpinned(self, capsys):
        args = _args("--enable-audio", "--asr-model", "openai/whisper-small")
        assert _resolve_asr_model(args) == ("openai/whisper-small", None)
        assert "no --asr-revision" in capsys.readouterr().out

    def test_custom_model_with_revision(self, capsys):
        args = _args(
            "--enable-audio",
            "--asr-model",
            "openai/whisper-small",
            "--asr-revision",
            "r",
        )
        assert _resolve_asr_model(args) == ("openai/whisper-small", "r")
        assert "WARNING" not in capsys.readouterr().out

    def test_bundle_asr_is_opt_in(self):
        assert _args("--enable-audio").bundle_asr is False
        assert _args("--bundle-asr").bundle_asr is True


def _tiny_asr_dir(root):
    """The shared tiny CTC model, plus a legacy weight file to be filtered."""
    build_tiny_asr(root)
    (root / "pytorch_model.bin").write_bytes(b"legacy")
    return root


class TestBundleAsrModel:
    def test_local_source_is_copied_and_validated(self, tmp_path):
        src = _tiny_asr_dir(tmp_path / "src")
        dest = tmp_path / "ckpt" / "asr"
        bundle_asr_model(str(src), None, str(dest))
        assert (dest / "config.json").is_file()
        assert (dest / "model.safetensors").is_file()
        # Legacy weights are not bundled.
        assert not (dest / "pytorch_model.bin").exists()

    def test_source_without_safetensors_is_refused(self, tmp_path):
        src = _tiny_asr_dir(tmp_path / "src")
        (src / "model.safetensors").unlink()
        with pytest.raises(FileNotFoundError, match="safetensors"):
            bundle_asr_model(str(src), None, str(tmp_path / "asr"))

    def test_hub_source_downloads_at_the_revision(self, tmp_path, monkeypatch):
        from granite_switch.composer import compose_granite_switch as cgs

        src = _tiny_asr_dir(tmp_path / "src")
        calls = {}

        def fake_snapshot_download(repo_id, revision, local_dir, allow_patterns):
            calls.update(repo_id=repo_id, revision=revision)
            import shutil

            shutil.copytree(src, local_dir, dirs_exist_ok=True)

        monkeypatch.setattr(cgs, "snapshot_download", fake_snapshot_download)
        bundle_asr_model("org/asr", "abc", str(tmp_path / "asr"))
        assert calls == {"repo_id": "org/asr", "revision": "abc"}

    def test_rebundle_replaces_the_previous_bundle(self, tmp_path):
        dest = tmp_path / "ckpt" / "asr"
        dest.mkdir(parents=True)
        # Left over from an earlier compose with another ASR model.
        (dest / "model-00001-of-00002.safetensors").write_bytes(b"old")
        (dest / "tokenizer.json").write_text("{}")
        bundle_asr_model(str(_tiny_asr_dir(tmp_path / "src")), None, str(dest))
        assert not (dest / "model-00001-of-00002.safetensors").exists()
        assert not (dest / "tokenizer.json").exists()
        assert (dest / "model.safetensors").is_file()

    def test_source_that_is_the_destination_is_refused(self, tmp_path):
        dest = _tiny_asr_dir(tmp_path / "asr")
        with pytest.raises(ValueError, match="bundle destination"):
            bundle_asr_model(str(dest), None, str(dest))
        assert (dest / "model.safetensors").is_file()
