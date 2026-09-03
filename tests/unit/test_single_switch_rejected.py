# SPDX-License-Identifier: Apache-2.0
"""``GraniteSwitchConfig.from_dict`` rejects SingleSwitch / legacy checkpoints.

SingleSwitch has been removed and MultiSwitch is the only engine. A checkpoint
built for SingleSwitch cannot load as MultiSwitch (it was sized for a different
decoder-layer count), so ``from_dict`` refuses it at the one seam that still sees
the raw on-disk config, with an actionable re-compose message.

Two shapes identify a SingleSwitch checkpoint (adapters > 0):
  * an explicit ``switch_type`` that is not ``"multi"``; or
  * no ``switch_type`` key AND no ``ms_code_m`` key (a legacy preview composed
    before the coded engine existed).

A real MultiSwitch checkpoint carries ``ms_code_m`` and is accepted; a stale
``switch_type`` key on it is stripped so the removed constructor parameter never
sees it.
"""

import pytest

from granite_switch.config import GraniteSwitchConfig


def _valid_multi_dict(**overrides):
    """A minimal, constructible MultiSwitch config dict (carries ms_code_m)."""
    d = dict(
        model_type="granite_switch",
        vocab_size=300,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_adapters=2,
        adapter_token_ids=[250, 251],
        adapter_substitute_token_ids=[1, 1],
        adapter_names=["adapter_0", "adapter_1"],
        max_lora_rank=8,
        adapter_ranks=[8, 8],
        ms_code_m=6,
    )
    d.update(overrides)
    return d


class TestSingleSwitchCheckpointRejected:
    def test_explicit_non_multi_switch_type_rejected(self):
        with pytest.raises(ValueError, match="SingleSwitch"):
            GraniteSwitchConfig.from_dict({"num_adapters": 2, "switch_type": "single"})

    def test_multi_coded_alias_rejected(self):
        """The old ``multi_coded`` alias is not ``"multi"``, so it is refused."""
        with pytest.raises(ValueError, match="SingleSwitch"):
            GraniteSwitchConfig.from_dict(
                {"num_adapters": 2, "switch_type": "multi_coded"}
            )

    def test_legacy_preview_without_ms_code_m_rejected(self):
        """No switch_type AND no ms_code_m -> predates the coded engine."""
        with pytest.raises(ValueError, match="SingleSwitch"):
            GraniteSwitchConfig.from_dict({"num_adapters": 2})


class TestMultiSwitchCheckpointAccepted:
    def test_real_multi_checkpoint_loads(self):
        cfg = GraniteSwitchConfig.from_dict(_valid_multi_dict())
        assert cfg.num_adapters == 2
        assert cfg.ms_code_m == 6

    def test_stale_switch_type_is_stripped(self):
        """A stale switch_type on a real MultiSwitch checkpoint is dropped, not stored."""
        cfg = GraniteSwitchConfig.from_dict(_valid_multi_dict(switch_type="multi"))
        assert not hasattr(cfg, "switch_type")

    def test_zero_adapter_checkpoint_never_rejected(self):
        """The reject only applies when adapters are present."""
        cfg = GraniteSwitchConfig.from_dict(
            {"model_type": "granite_switch", "num_adapters": 0, "switch_type": "single"}
        )
        assert cfg.num_adapters == 0
