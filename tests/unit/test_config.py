# SPDX-License-Identifier: Apache-2.0
"""Config validation tests for GraniteSwitchConfig.

Tests every ValueError path in __init__, default values, and derived properties.
"""

import pytest

from granite_switch.config import GraniteSwitchConfig


# ── Helper ────────────────────────────────────────────────────────────

def _valid_kwargs(num_adapters=2, **overrides):
    """Return kwargs for a valid SingleSwitch config, with optional overrides.

    Default mode: legacy KV hiding (control_dims=32) so existing shape/hiding
    tests keep their original contract. Token-exchange tests override
    control_dims=0 and pass adapter_substitute_token_ids.
    """
    adapter_names = [f"adapter_{i}" for i in range(num_adapters)]
    base = dict(
        vocab_size=300,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_adapters=num_adapters,
        adapter_token_ids=list(range(500, 500 + num_adapters)),
        adapter_names=adapter_names,
        max_lora_rank=8,
        adapter_ranks=[8] * num_adapters,
        control_dims=32,
    )
    base.update(overrides)
    return base


# ════════════════════════════════════════════════════════════════════
# 1. Config validation — every ValueError path
# ════════════════════════════════════════════════════════════════════

class TestConfigValidation:

    def test_negative_num_adapters_raises(self):
        with pytest.raises(ValueError, match="num_adapters must be >= 0"):
            GraniteSwitchConfig(**_valid_kwargs(num_adapters=-1, adapter_ranks=None))

    def test_adapter_token_ids_wrong_length_raises(self):
        with pytest.raises(ValueError, match="adapter_token_ids length"):
            GraniteSwitchConfig(**_valid_kwargs(
                adapter_token_ids=[500, 501, 502],  # length 3, expected 2
            ))

    def test_missing_adapter_ranks_raises(self):
        with pytest.raises(ValueError, match="adapter_ranks must be provided"):
            GraniteSwitchConfig(**_valid_kwargs(adapter_ranks=None))

    def test_adapter_ranks_wrong_length_raises(self):
        with pytest.raises(ValueError, match="adapter_ranks length"):
            GraniteSwitchConfig(**_valid_kwargs(
                adapter_ranks=[8],  # length 1, expected 2
            ))

    def test_max_adapter_ranks_mismatch_raises(self):
        with pytest.raises(ValueError, match="max.*adapter_ranks.*must equal max_lora_rank"):
            GraniteSwitchConfig(**_valid_kwargs(
                adapter_ranks=[4, 4],  # max=4, but max_lora_rank=8
            ))


# ════════════════════════════════════════════════════════════════════
# 2. Config defaults and derived properties
# ════════════════════════════════════════════════════════════════════

class TestConfigDefaults:

    def test_zero_adapters_no_validation(self):
        """Config with 0 adapters should not require adapter_ranks or token_ids."""
        cfg = GraniteSwitchConfig(
            vocab_size=256, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            num_adapters=0,
        )
        assert cfg.num_adapters == 0
        assert cfg.adapter_ranks is None


# ════════════════════════════════════════════════════════════════════
# 3. Hiding groups and policy
# ════════════════════════════════════════════════════════════════════

class TestHidingConfig:

    def test_hiding_groups_none_by_default(self):
        """No hiding groups when not specified."""
        cfg = GraniteSwitchConfig(**_valid_kwargs())
        assert cfg.num_hiding_groups == 0
        assert cfg.hiding_group_names == []

    def test_hiding_groups_count(self):
        """num_hiding_groups reflects configured groups."""
        cfg = GraniteSwitchConfig(**_valid_kwargs(
            hiding_groups={
                "group_a": ["adapter_0"],
                "group_b": ["adapter_1"],
            },
        ))
        assert cfg.num_hiding_groups == 2
        assert cfg.hiding_group_names == ["group_a", "group_b"]

    def test_control_dims_less_than_groups_raises(self):
        """control_dims must be >= number of hiding groups."""
        with pytest.raises(ValueError, match="control_dims.*must be >= number of hiding groups"):
            GraniteSwitchConfig(**_valid_kwargs(
                control_dims=1,
                hiding_groups={
                    "g1": ["adapter_0"],
                    "g2": ["adapter_1"],
                },
            ))

    def test_get_hiding_group_token_ids(self):
        """Token IDs resolved correctly for SingleSwitch."""
        cfg = GraniteSwitchConfig(**_valid_kwargs(
            hiding_groups={"all_controls": ["adapter_0", "adapter_1"]},
        ))
        group_tokens = cfg.get_hiding_group_token_ids()
        # SingleSwitch: no base offset, adapter_0 → token 500, adapter_1 → token 501
        assert group_tokens == {0: [500, 501]}

    def test_get_hiding_group_token_ids_multiple_groups(self):
        """Multiple groups with different adapter assignments."""
        cfg = GraniteSwitchConfig(**_valid_kwargs(
            hiding_groups={
                "group_a": ["adapter_0"],
                "group_b": ["adapter_1"],
            },
        ))
        group_tokens = cfg.get_hiding_group_token_ids()
        assert group_tokens == {0: [500], 1: [501]}

    def test_get_adapter_hiding_policy_matrix(self):
        """Policy matrix built correctly from named config."""
        cfg = GraniteSwitchConfig(**_valid_kwargs(
            hiding_groups={
                "group_a": ["adapter_0"],
                "group_b": ["adapter_1"],
            },
            hiding_policy={
                "base": ["group_a", "group_b"],
                "adapter_0": ["group_b"],
                "adapter_1": ["group_a"],
            },
        ))
        matrix = cfg.get_adapter_hiding_policy_matrix()
        # [base, adapter_0, adapter_1] x [group_a, group_b]
        assert matrix == [
            [True, True],    # base hides both
            [False, True],   # adapter_0 hides group_b only
            [True, False],   # adapter_1 hides group_a only
        ]

    def test_get_adapter_hiding_policy_matrix_no_policy(self):
        """Empty matrix when no policy configured."""
        cfg = GraniteSwitchConfig(**_valid_kwargs())
        assert cfg.get_adapter_hiding_policy_matrix() == []


# ════════════════════════════════════════════════════════════════════
# 4. Third-party adapter config
# ════════════════════════════════════════════════════════════════════

class TestAdapterThirdParty:

    def test_adapter_third_party_stored(self):
        cfg = GraniteSwitchConfig(**_valid_kwargs(
            adapter_third_party=["adapter_0"],
        ))
        assert cfg.adapter_third_party == ["adapter_0"]

    def test_adapter_third_party_none_by_default(self):
        cfg = GraniteSwitchConfig(**_valid_kwargs())
        assert cfg.adapter_third_party is None
