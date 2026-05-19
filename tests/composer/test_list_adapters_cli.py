# SPDX-License-Identifier: Apache-2.0
"""Tests for the ``--list-adapters`` CLI path in compose_granite_switch.

These are fast, CPU-only tests that mock network calls.  They verify:
1. ``--list-adapters`` exits cleanly (exit code 0) without crashing ``main()``
2. ``--list-adapters`` without ``--adapters`` returns exit code 1
3. ``--list-adapters`` handles remote listing failures gracefully (exit code 1)
"""

import sys
from unittest.mock import patch

import pytest


FAKE_ADAPTERS = [
    {"name": "rag", "technologies": ["alora", "lora"]},
    {"name": "summarize", "technologies": ["lora"]},
]


def _run_main(argv):
    """Invoke main() with mocked sys.argv and return exit code."""
    from granite_switch.composer.compose_granite_switch import main

    with patch.object(sys, "argv", argv):
        return main()


class TestListAdaptersCLI:
    """Verify --list-adapters exits cleanly through main()."""

    def test_list_adapters_exits_zero(self):
        """--list-adapters with a valid (mocked) remote repo returns 0."""
        argv = [
            "compose_granite_switch",
            "--base-model", "ibm-granite/granite-4.1-3b",
            "--adapters", "ibm-granite/some-lib",
            "--list-adapters",
        ]
        with patch(
            "granite_switch.composer.compose_granite_switch.list_repo_adapters_remote",
            return_value=FAKE_ADAPTERS,
        ):
            rc = _run_main(argv)

        assert rc == 0

    def test_list_adapters_no_adapters_flag_exits_one(self):
        """--list-adapters without --adapters returns exit code 1."""
        argv = [
            "compose_granite_switch",
            "--base-model", "ibm-granite/granite-4.1-3b",
            "--list-adapters",
        ]
        rc = _run_main(argv)
        assert rc == 1

    def test_list_adapters_remote_failure_exits_one(self):
        """--list-adapters with a network error returns exit code 1."""
        argv = [
            "compose_granite_switch",
            "--base-model", "ibm-granite/granite-4.1-3b",
            "--adapters", "ibm-granite/some-lib",
            "--list-adapters",
        ]
        with patch(
            "granite_switch.composer.compose_granite_switch.list_repo_adapters_remote",
            side_effect=RuntimeError("network timeout"),
        ):
            rc = _run_main(argv)

        assert rc == 1

    def test_list_adapters_empty_results(self):
        """--list-adapters with no matching adapters still exits 0."""
        argv = [
            "compose_granite_switch",
            "--base-model", "ibm-granite/granite-4.1-3b",
            "--adapters", "ibm-granite/some-lib",
            "--list-adapters",
        ]
        with patch(
            "granite_switch.composer.compose_granite_switch.list_repo_adapters_remote",
            return_value=[],
        ):
            rc = _run_main(argv)

        assert rc == 0
