# SPDX-License-Identifier: Apache-2.0
"""Additional config edge case tests for GraniteSwitchConfig."""

from granite_switch.config import GraniteSwitchConfig


def _valid_kwargs(num_adapters=2, **overrides):
    """Return kwargs for a valid token-exchange config."""
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
        adapter_substitute_token_ids=[1] * num_adapters,
        adapter_names=adapter_names,
        max_lora_rank=8,
        adapter_ranks=[8] * num_adapters,
    )
    base.update(overrides)
    return base


class TestSharedIntermediateSize:
    """GraniteSwitchConfig owns the shared_intermediate_size decision itself,
    independent of the parent-class default.

    This guards the de-hybridization trap: the GraniteMoeShared parent defaults
    shared_intermediate_size to 0, which is ALSO the "no shared MLP" sentinel for
    pure sparse-MoE bases. The config must therefore resolve the value from the
    presence of experts rather than inherit a magic default. See
    docs/DEHYBRIDIZATION_MOESHARED_IMPLEMENTATION_PLAN.md Section 4.A.
    """

    _DIMS = dict(
        vocab_size=300,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
    )

    def test_bare_dense_resolves_to_intermediate_size(self):
        # (a) Dense (no experts) and no explicit value -> shared MLP sized to
        # intermediate_size. NOT the parent default (which would be 0 or 1024).
        cfg = GraniteSwitchConfig(num_adapters=0, **self._DIMS)
        assert cfg.shared_intermediate_size == cfg.intermediate_size == 128

    def test_bare_pure_moe_keeps_zero_sentinel(self):
        # (b) Pure sparse MoE and no explicit value -> 0 ("no shared MLP").
        cfg = GraniteSwitchConfig(
            num_adapters=0,
            num_local_experts=4,
            num_experts_per_tok=2,
            **self._DIMS,
        )
        assert cfg.shared_intermediate_size == 0

    def test_explicit_zero_is_honored(self):
        # (c) An explicit 0 must survive verbatim (the sentinel), even on a
        # dense config that would otherwise resolve to intermediate_size.
        cfg = GraniteSwitchConfig(
            num_adapters=0, shared_intermediate_size=0, **self._DIMS
        )
        assert cfg.shared_intermediate_size == 0

    def test_explicit_positive_is_honored(self):
        # (d) An explicit positive value is honored verbatim.
        cfg = GraniteSwitchConfig(
            num_adapters=0, shared_intermediate_size=256, **self._DIMS
        )
        assert cfg.shared_intermediate_size == 256

    def test_round_trip_preserves_resolved_value(self):
        # (e) The resolved value is frozen at construction and survives
        # save/reload (to_dict -> from_dict), matching the compose-time-frozen,
        # inference-time-consumed lifecycle.
        cfg = GraniteSwitchConfig(num_adapters=0, **self._DIMS)
        reloaded = GraniteSwitchConfig.from_dict(cfg.to_dict())
        assert reloaded.shared_intermediate_size == 128


class TestLayerTypesDefault:
    """layer_types defaults to all-attention with length == num_hidden_layers."""

    def test_default_layer_types_when_omitted(self):
        cfg = GraniteSwitchConfig(num_adapters=0, num_hidden_layers=4)
        assert cfg.layer_types == ["attention"] * 4

    def test_explicit_layer_types_preserved(self):
        cfg = GraniteSwitchConfig(
            num_adapters=0,
            num_hidden_layers=3,
            layer_types=["attention", "attention", "attention"],
        )
        assert cfg.layer_types == ["attention", "attention", "attention"]


class TestLoraTargetModulesDefault:
    """lora_target_modules defaults to qkv_proj/o_proj + shared_mlp pair
    when num_adapters > 0; empty when num_adapters == 0."""

    def test_no_adapters_empty_target_modules(self):
        cfg = GraniteSwitchConfig(num_adapters=0)
        assert cfg.lora_target_modules == []

    def test_adapters_populate_target_modules(self):
        cfg = GraniteSwitchConfig(**_valid_kwargs())
        assert "qkv_proj" in cfg.lora_target_modules
        assert "o_proj" in cfg.lora_target_modules
        assert "shared_input_linear" in cfg.lora_target_modules
        assert "shared_output_linear" in cfg.lora_target_modules

    def test_explicit_target_modules_preserved(self):
        cfg = GraniteSwitchConfig(**_valid_kwargs(lora_target_modules=["qkv_proj"]))
        assert cfg.lora_target_modules == ["qkv_proj"]
