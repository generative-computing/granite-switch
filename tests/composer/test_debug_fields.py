# SPDX-License-Identifier: Apache-2.0
"""Tests for --debug-fields flag and source propagation in compose reports."""

import pytest

from granite_switch.composer.adapter_discovery import discover_adapters

pytestmark = pytest.mark.local_fast


class TestSourcePropagation:
    """Tests for source propagation in discover_adapters()."""

    def test_source_propagated_in_tuple(self, tmp_path):
        """Source parameter should be included in returned tuples."""
        # Create a minimal adapter library structure
        adapter_dir = tmp_path / "test-adapter" / "granite-4.0-micro" / "alora"
        adapter_dir.mkdir(parents=True)
        (adapter_dir / "io.yaml").write_text("input: test")
        (adapter_dir / "adapter_model.safetensors").write_text("")
        (adapter_dir / "adapter_config.json").write_text("{}")

        # Create a minimal arch descriptor
        class MockGroup:
            name = "test"
            peft_modules = []

        class MockArch:
            groups = [MockGroup()]

        source = "ibm-granite/granitelib-rag-r1.0"
        discovered = discover_adapters(
            str(tmp_path),
            "granite-4.0-micro",
            MockArch(),
            source=source,
        )

        assert len(discovered) == 1
        assert len(discovered[0]) == 4  # (path, name, tech, source)
        assert discovered[0][3] == source

    def test_source_none_when_not_provided(self, tmp_path):
        """Source should be None when not provided."""
        # Create a minimal adapter library structure
        adapter_dir = tmp_path / "test-adapter" / "granite-4.0-micro" / "alora"
        adapter_dir.mkdir(parents=True)
        (adapter_dir / "io.yaml").write_text("input: test")
        (adapter_dir / "adapter_model.safetensors").write_text("")
        (adapter_dir / "adapter_config.json").write_text("{}")

        class MockGroup:
            name = "test"
            peft_modules = []

        class MockArch:
            groups = [MockGroup()]

        discovered = discover_adapters(
            str(tmp_path),
            "granite-4.0-micro",
            MockArch(),
        )

        assert len(discovered) == 1
        assert len(discovered[0]) == 4
        assert discovered[0][3] is None


class TestDebugFieldsFlag:
    """Tests for --debug-fields flag behavior in adapter_index.json."""

    def test_original_path_excluded_by_default(self):
        """original_path should NOT be in adapter_index.json by default."""
        from granite_switch.composer.compose_granite_switch import (
            _create_adapter_index,
        )
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            # 4-tuple: (path, name, tech, source)
            discovered = [
                ("/local/path/to/adapter", "test-adapter", "alora", "org/repo"),
            ]
            io_config_paths = ["io_configs/test-adapter/io.yaml"]
            token_ids = [100]

            index = _create_adapter_index(
                discovered,
                io_config_paths,
                token_ids,
                tmp_dir,
                "ibm-granite/granite-4.0-micro",
                include_debug_fields=False,  # Default
            )

            # original_path should NOT be present
            assert "original_path" not in index["adapters"][0]

    def test_original_path_included_with_debug_flag(self):
        """original_path should be in adapter_index.json with --debug-fields."""
        from granite_switch.composer.compose_granite_switch import (
            _create_adapter_index,
        )
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            # 4-tuple: (path, name, tech, source)
            source = "ibm-granite/granitelib-rag-r1.0"
            discovered = [
                ("/local/path", "answerability", "alora", source),
            ]
            io_config_paths = ["io_configs/answerability/io.yaml"]
            token_ids = [100]

            index = _create_adapter_index(
                discovered,
                io_config_paths,
                token_ids,
                tmp_dir,
                "ibm-granite/granite-4.0-micro",
                include_debug_fields=True,
            )

            # original_path should be present and use source
            assert "original_path" in index["adapters"][0]
            expected = f"{source}/answerability"
            assert index["adapters"][0]["original_path"] == expected

    def test_local_path_used_when_no_source(self):
        """Local path should be used when source is None."""
        from granite_switch.composer.compose_granite_switch import (
            _create_adapter_index,
        )
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            local_path = "/home/user/my-adapters/custom-adapter"
            discovered = [
                (local_path, "custom-adapter", "alora", None),
            ]
            io_config_paths = ["io_configs/custom-adapter/io.yaml"]
            token_ids = [100]

            index = _create_adapter_index(
                discovered,
                io_config_paths,
                token_ids,
                tmp_dir,
                "ibm-granite/granite-4.0-micro",
                include_debug_fields=True,
            )

            # Should fall back to local path when source is None
            assert index["adapters"][0]["original_path"] == local_path

    def test_built_in_adapters_no_original_path(self):
        """Built-in adapters have no original_path even with debug flag."""
        from granite_switch.composer.compose_granite_switch import (
            _create_adapter_index,
        )
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Built-in adapter has None path
            discovered = [
                (None, "base", "builtin", None),
            ]
            io_config_paths = [None]
            token_ids = [100]

            index = _create_adapter_index(
                discovered,
                io_config_paths,
                token_ids,
                tmp_dir,
                "ibm-granite/granite-4.0-micro",
                include_debug_fields=True,
            )

            # Built-in should have built_in=True, not original_path
            assert index["adapters"][0].get("built_in") is True
            assert "original_path" not in index["adapters"][0]
