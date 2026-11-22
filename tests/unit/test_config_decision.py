# SPDX-License-Identifier: Apache-2.0
"""GraniteSwitchConfig's decision fields: what a checkpoint records about its
decision adapters, validated on construction and on load."""

import copy

import pytest

from granite_switch.config import GraniteSwitchConfig
from tests.shared.decision_fixtures import REFERENCE_BLOCK
from tests.unit.test_config_edge_cases import _valid_kwargs


def _specs(*names):
    return {name: copy.deepcopy(REFERENCE_BLOCK) for name in names}


def test_no_decision_fields_by_default():
    cfg = GraniteSwitchConfig(**_valid_kwargs())
    assert cfg.decision_adapter is None
    assert cfg.decision_specs is None


def test_specs_are_normalized_and_survive_a_save_load(tmp_path):
    cfg = GraniteSwitchConfig(
        **_valid_kwargs(
            decision_adapter="adapter_1", decision_specs=_specs("adapter_1")
        )
    )
    spec = cfg.decision_specs["adapter_1"]
    assert spec["option_line"] == "{label}: {text}"
    assert spec["invocation_marker"] == "### DECISION"

    cfg.save_pretrained(tmp_path)
    loaded = GraniteSwitchConfig.from_pretrained(tmp_path)
    assert loaded.decision_adapter == "adapter_1"
    assert loaded.decision_specs == cfg.decision_specs


def test_a_probe_slot_cannot_be_a_decision_adapter():
    with pytest.raises(ValueError, match="probe slot"):
        GraniteSwitchConfig(
            **_valid_kwargs(
                adapter_kinds=["lora", "probe"],
                probe_label_token_ids=[None, [5, 6]],
                decision_adapter="adapter_1",
                decision_specs=_specs("adapter_1"),
            )
        )


def test_spec_for_an_unknown_adapter_is_refused():
    with pytest.raises(ValueError, match="not an adapter of this checkpoint"):
        GraniteSwitchConfig(
            **_valid_kwargs(decision_adapter="nope", decision_specs=_specs("nope"))
        )


def test_default_must_be_one_of_the_specs():
    with pytest.raises(ValueError, match="must name one of"):
        GraniteSwitchConfig(
            **_valid_kwargs(
                decision_adapter="adapter_0", decision_specs=_specs("adapter_1")
            )
        )


def test_default_without_specs_is_refused():
    with pytest.raises(ValueError, match="decision_specs is empty"):
        GraniteSwitchConfig(**_valid_kwargs(decision_adapter="adapter_0"))


def test_invalid_spec_is_refused():
    bad = _specs("adapter_0")
    bad["adapter_0"]["messages"] = []
    with pytest.raises(ValueError, match="non-empty list"):
        GraniteSwitchConfig(
            **_valid_kwargs(decision_adapter="adapter_0", decision_specs=bad)
        )
