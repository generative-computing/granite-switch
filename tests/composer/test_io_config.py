# SPDX-License-Identifier: Apache-2.0
"""Tests for _copy_io_configs — the --create-ioyaml behavior.

An adapter is expected to ship an ``io.yaml``. When it doesn't:
  * default (create_ioyaml=False): compose fails with a clear FileNotFoundError.
  * --create-ioyaml (create_ioyaml=True): a minimal io.yaml is synthesized.
Built-in adapters (adapter_path is None) are always skipped.
"""

import pytest
import yaml

from granite_switch.composer.compose_granite_switch import (
    _MINIMAL_IO_YAML,
    _copy_io_configs,
)


def _adapter_dir(tmp_path, name, with_io=False):
    d = tmp_path / name
    d.mkdir()
    if with_io:
        (d / "io.yaml").write_text("name: real\nmodel: m\n")
    return (str(d), name, "alora", None)


class TestCopyIoConfigs:
    def test_copies_existing_io_yaml(self, tmp_path, capsys):
        out = tmp_path / "out"
        out.mkdir()
        adapters = [_adapter_dir(tmp_path, "has_io", with_io=True)]
        paths = _copy_io_configs(adapters, str(out))
        assert paths == ["io_configs/has_io/io.yaml"]
        assert (out / "io_configs/has_io/io.yaml").is_file()

    def test_missing_io_yaml_fails_by_default(self, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        adapters = [_adapter_dir(tmp_path, "no_io", with_io=False)]
        with pytest.raises(FileNotFoundError, match="--create-ioyaml"):
            _copy_io_configs(adapters, str(out))

    def test_missing_io_yaml_synthesized_with_flag(self, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        adapters = [_adapter_dir(tmp_path, "no_io", with_io=False)]
        paths = _copy_io_configs(adapters, str(out), create_ioyaml=True)
        assert paths == ["io_configs/no_io/io.yaml"]
        dest = out / "io_configs/no_io/io.yaml"
        assert dest.is_file()
        # Minimal io.yaml: all four fields present and null.
        loaded = yaml.safe_load(dest.read_text())
        assert loaded == {
            "name": None,
            "model": None,
            "response_format": None,
            "transformations": None,
        }
        assert dest.read_text() == _MINIMAL_IO_YAML

    def test_builtin_adapter_skipped(self, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        adapters = [(None, "builtin", "builtin", None)]
        # Built-ins never need an io.yaml, even without the flag.
        paths = _copy_io_configs(adapters, str(out))
        assert paths == [None]
