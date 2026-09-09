# SPDX-License-Identifier: Apache-2.0
"""vLLM backend for Granite Switch model."""

__version__ = "0.1.0"

# Export main classes
from granite_switch.config import GraniteSwitchConfig

# Export core components (for advanced use)
from .core import SwitchedLoRALinear
from .decoder import GraniteLoRAEmbeddedAttention, GraniteSwitchDecoderLayer
from .granite_switch_model import GraniteSwitchForCausalLM, GraniteSwitchModel
from .switch import SingleSwitch

__all__ = [
    "GraniteLoRAEmbeddedAttention",
    # Main API
    "GraniteSwitchConfig",
    "GraniteSwitchDecoderLayer",
    "GraniteSwitchForCausalLM",
    "GraniteSwitchModel",
    "SingleSwitch",
    # Core components (advanced)
    "SwitchedLoRALinear",
    "register",
]

# Register config with transformers AutoConfig
try:
    from transformers import AutoConfig

    AutoConfig.register("granite_switch", GraniteSwitchConfig)
except Exception:
    # Registration may fail if already registered or transformers not available
    pass


def register():
    """Register the GraniteSwitch model with vLLM.

    This function is called by vLLM's plugin system on startup.
    It must be re-entrant (can be called multiple times safely).
    """
    from vllm import ModelRegistry

    # vLLM <=0.25 keys its Granite-hybrid layer table on the pre-5.16 spelling of
    # a layer type, so a config written by transformers >=5.16 -- which renamed
    # "attention" to "full_attention" and rewrites it inside
    # PreTrainedConfig.__init__ -- raises KeyError at engine init:
    #
    #     ALL_DECODER_LAYER_TYPES[config.layer_types[layer_idx]]
    #     KeyError: 'full_attention'
    #
    # Both names denote the same layer class, so aliasing is not a behaviour
    # change: it teaches vLLM to accept the only spelling transformers can now
    # produce. This must happen here rather than in a test fixture because the
    # lookup runs in vLLM's spawned engine-core process, and this plugin hook is
    # the one thing we own that executes there (v1/engine/core.py calls
    # load_general_plugins() during init). setdefault makes it a no-op on 0.26+,
    # where upstream added the key themselves, so the block can be deleted
    # whenever we move off the 0.19/0.20 line. See issue #122.
    try:
        from vllm.model_executor.models import granitemoehybrid as _gmh

        _gmh.ALL_DECODER_LAYER_TYPES.setdefault(
            "full_attention", _gmh.ALL_DECODER_LAYER_TYPES["attention"]
        )
    except Exception:  # pragma: no cover - must never block registration
        pass

    # Register config with transformers AutoConfig
    try:
        from transformers import AutoConfig

        AutoConfig.register("granite_switch", GraniteSwitchConfig)
    except Exception:
        pass

    # Register custom ModelArchConfigConvertor so vLLM sees:
    #   1. The correct decoder layer count (excluding the switch's KV-cache
    #      placeholder slot).
    #   2. The native KV cache head size (projection_head_dim). Token
    #      exchange does not expand the head dim, so this is just the base
    #      model's head_dim.
    try:
        from vllm.transformers_utils.model_arch_config_convertor import (
            MODEL_ARCH_CONFIG_CONVERTORS,
            ModelArchConfigConvertorBase,
        )

        class _GraniteSwitchArchConfigConvertor(ModelArchConfigConvertorBase):
            def get_num_hidden_layers(self) -> int:
                cfg = self.hf_text_config
                num_layers = super().get_num_hidden_layers()
                if getattr(cfg, "num_adapters", 0) > 0:
                    # GraniteSwitch configs include one SingleSwitch KV-cache
                    # placeholder before the decoder layers.  vLLM discovers the
                    # switch Attention module separately for KV allocation, but
                    # PP layer slicing must only count physical decoder layers.
                    return max(0, num_layers - 1)
                return num_layers

            def get_head_size(self) -> int:
                cfg = self.hf_text_config
                return getattr(cfg, "projection_head_dim", super().get_head_size())

        MODEL_ARCH_CONFIG_CONVERTORS["granite_switch"] = (
            _GraniteSwitchArchConfigConvertor
        )
    except ImportError:
        pass

    # LoRA/aLoRA and Shadow-Residual are two adaptations of ONE host model, so
    # all arch strings point at the SAME shared GraniteSwitchForCausalLM class.
    # vLLM selects the class from config.architectures; the class then picks its
    # adaptation from config.cross_stream_rank (None -> LoRA, int -> SR).
    #   - GraniteSwitchForCausalLM  : LoRA/aLoRA composed checkpoints.
    #   - SRSwitchForCausalLM       : what the composer writes into a composed SR
    #     checkpoint's config.architectures (mirrors the HF SR arch name), so an
    #     SR model auto-dispatches with no hf_overrides.
    #   - ShadowResidualForCausalLM : explicit alias for callers that force the arch.
    target = "granite_switch.vllm.granite_switch_model:GraniteSwitchForCausalLM"
    supported = ModelRegistry.get_supported_archs()
    for arch in (
        "GraniteSwitchForCausalLM",
        "SRSwitchForCausalLM",
        "ShadowResidualForCausalLM",
    ):
        if arch not in supported:
            ModelRegistry.register_model(arch, target)
            print(f"✓ {arch} registered with vLLM")
