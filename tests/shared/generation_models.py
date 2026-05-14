# SPDX-License-Identifier: Apache-2.0
"""Shared model configs and builder for generation tests.

Used by both HF (tests/hf/test_generation.py) and vLLM
(tests/vllm/_noneager_generation_tests.py) generation tests.
"""

import gc
import os

import torch

# ── Base model config (dense, attention-only) ─────────────────────

DENSE_CFG = dict(
    vocab_size=256,
    hidden_size=96,
    num_hidden_layers=3,
    num_attention_heads=3,
    num_key_value_heads=1,
    intermediate_size=192,
    shared_intermediate_size=192,
    layer_types=["attention", "attention", "attention"],
    max_position_embeddings=2048,
    attention_bias=False,
    mlp_bias=False,
    embedding_multiplier=12.0,
    residual_multiplier=0.22,
    attention_multiplier=0.0078125,
    logits_scaling=6.0,
    # Set valid mamba params to pass parent GraniteMoeHybridConfig validation
    # mamba_intermediate = mamba_expand(2) * hidden_size(96) = 192
    # mamba_n_heads must divide 192 evenly
    mamba_n_heads=1,
)

# Backward compatibility alias
HYBRID_CFG = DENSE_CFG

# ── Adapter constants ─────────────────────────────────────────────

NUM_ADAPTERS = 2
ADAPTER_RANK = 8

# ── Switch override dicts ─────────────────────────────────────────
# Merged with a base config (HYBRID_CFG or DENSE_CFG) via
# {**base, **overrides}.  Each override includes layer_types and
# num_hidden_layers that prepend the switch layer(s) to the base.


def single_overrides(base_cfg):
    """SingleSwitch overrides for the given base config (token exchange)."""
    base_layers = base_cfg["layer_types"]
    return {
        "num_adapters": NUM_ADAPTERS,
        "adapter_ranks": [ADAPTER_RANK] * NUM_ADAPTERS,
        "adapter_token_ids": [250, 251],
        "adapter_substitute_token_ids": [1, 1],
        "adapter_names": ["adapter_0", "adapter_1"],
        "num_hidden_layers": len(base_layers) + 1,
        "layer_types": ["attention"] + base_layers,
    }


# Backward compatibility alias
basic_overrides = single_overrides


# ── Model builder ─────────────────────────────────────────────────

def save_switch_model(base_cfg, cfg_overrides, tmpdir):
    """Build a GraniteSwitch model from config and save to disk.

    Returns the path to the saved model directory.
    """
    from granite_switch.config import GraniteSwitchConfig
    from granite_switch.hf import GraniteSwitchForCausalLM as HFSwitch

    cfg_dict = {**base_cfg, **cfg_overrides}

    switch_cfg = GraniteSwitchConfig(**cfg_dict)
    torch.manual_seed(0)
    switch = HFSwitch(switch_cfg).eval()

    save_dir = os.path.join(str(tmpdir), "model")
    switch.save_pretrained(save_dir)
    del switch
    gc.collect()
    return save_dir


def make_switch_model(base_cfg, cfg_overrides, seed=0):
    """Build a GraniteSwitch model in-memory (no save to disk).

    Returns (model, config).
    """
    from granite_switch.config import GraniteSwitchConfig
    from granite_switch.hf import GraniteSwitchForCausalLM as HFSwitch

    cfg_dict = {**base_cfg, **cfg_overrides}

    switch_cfg = GraniteSwitchConfig(**cfg_dict)
    torch.manual_seed(seed)
    model = HFSwitch(switch_cfg).eval()
    return model, switch_cfg
