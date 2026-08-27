# SPDX-License-Identifier: Apache-2.0
"""Decoder interfaces for the shared Granite Switch vLLM model.

Granite Switch is a *host* (base model + embedded adapters + a switch that fires
them). LoRA and Shadow-Residual (SR) are two *adaptations* hosted on it, each
exposed to the shared model through a ``DecoderInterface``. The single shared
classes ``GraniteSwitchModel`` / ``GraniteSwitchForCausalLM`` hold one of these
interface objects as ``self.decoder_interface`` and call its hooks instead of
inlining adaptation-specific logic, so the shared ``forward`` / ``__init__``
never branch on which adaptation is in use.

The split between LoRA and SR is confined to the **decoder tier**: the decoder
layer type, its kernel-metadata layout, its ctx-wire types, its weight-fuse
rules, and — for SR — the stream doubling (``[M,H] -> [2M,H]``) and terminal
per-token merge.
Everything above the decoder (vocab sizing, PP handoff, ``compute_logits``,
``sample``, the four host interfaces) is shared, adaptation-agnostic host code.

Selection is keyed on ``config.cross_stream_rank`` (``None`` for LoRA, an int for
SR); see :func:`select_decoder_interface`.

NOTE: this module must NOT use ``from __future__ import annotations`` — the
shared model file re-applies ``@support_torch_compile``, whose dynamic-dim
inference breaks on stringized annotations, and mirroring that constraint here
keeps the strategy types plain.
"""

import abc
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import is_pp_missing_parameter

from ..core import (
    FusedLoRAKernelMeta,
    LoRAContext,
    SwitchedLoRALinear,
)
from .lora import (
    GraniteLoRAEmbeddedAttention,
    GraniteSwitchDecoderLayer,
    rms_norm_select,
)
from .shadow_residual.decoder import ShadowResidualDecoderLayer
from .shadow_residual.kernel_meta import SRFusedLoRAKernelMeta, SRLoRAContext
from .shadow_residual.wcross_shunt import WCrossShunt

logger = init_logger(__name__)


# --------------------------------------------------------------------------- #
# SR load-time skips.
#
# The composer PRE-FUSES q/k/v -> qkv_proj and gate/up -> shared_mlp.input_linear
# (docs/SR_ARCHITECTURE.md; GraniteSwitchConfig rejects unfused_qkv=True), so a
# composed SR checkpoint's parameter names now match the vLLM SR model 1:1 and
# load directly by name — no fuse-at-load shard mapping is needed anymore (the
# old _SR_BASE_STACKED / _SR_LORA_SLICED / _SR_DIRECT_RENAME tables keyed off the
# obsolete unfused q_proj/k_proj/gate_proj/down_proj names and were removed).
#
# These names are still skipped on load:
#   * .cross_stream.base_layer. — HF writes a zeros base for the cross_stream
#     SwitchedLoRALinear, but the vLLM WCrossShunt is W-less (no base_layer).
#   * control_to_substitute_lut / adapter_token_ids — regenerated from config.
# --------------------------------------------------------------------------- #
_SR_SKIP_SUBSTRINGS = (
    "adapter_token_ids",
    "control_to_substitute_lut",
    ".cross_stream.base_layer.",
)


# --------------------------------------------------------------------------- #
# Per-adaptation stack state carried through the decoder loop
# --------------------------------------------------------------------------- #
@dataclass
class LoRAStackState:
    """Threaded through the LoRA decoder loop: the fused/separate residual pair."""

    hidden_states: torch.Tensor
    residual: torch.Tensor | None


@dataclass
class SRStackState:
    """Threaded through the SR decoder loop: the stacked ``[2M, H]`` stream."""

    hs: torch.Tensor


# --------------------------------------------------------------------------- #
# Decoder interface
# --------------------------------------------------------------------------- #
class DecoderInterface(abc.ABC):
    """Hooks that let the shared model build + run either adaptation.

    Build-time hooks are called from ``GraniteSwitchModel.__init__`` /
    ``GraniteSwitchForCausalLM``; forward-time hooks from
    ``GraniteSwitchModel.forward``. All are pure factories or pure tensor
    transforms — none mutate the model.
    """

    # ---- build-time ----
    @abc.abstractmethod
    def make_kernel_meta(self, device) -> tuple[FusedLoRAKernelMeta, LoRAContext]:
        """Build the (kernel-meta, ctx) pair for this adaptation."""

    @abc.abstractmethod
    def make_decoder_layer(self, vllm_config: VllmConfig, prefix: str) -> nn.Module:
        """Build one decoder layer for this adaptation."""

    @abc.abstractmethod
    def ctx_wire_types(self) -> tuple[type, ...]:
        """Module types onto which the shared LoRA ctx is wired."""

    @abc.abstractmethod
    def prepare_kernel_meta(self, lora_meta, adapter_indices, lora_ctx) -> None:
        """Populate the ctx kernel metadata from per-token ``adapter_indices``."""

    @abc.abstractmethod
    def load_weights(self, model, weights: Iterable[tuple[str, torch.Tensor]]) -> set:
        """Apply checkpoint weights into ``model`` and finalize fused state.

        Owns the whole per-adaptation weight-application decision (not just a
        1->1 name rename): LoRA fans HF stacked-MoE tensors into per-expert
        FusedMoE loads; SR fans one checkpoint tensor into sharded fused loads
        and marks the intentionally-absent shared-KV LoRA slices loaded. Both
        end by calling :meth:`finalize_modules`. ``model`` is the ``*ForCausalLM``
        (so ``named_parameters`` / ``is_pp_missing_parameter`` are reachable).
        Returns the set of loaded parameter names (vLLM ignores the return).
        """

    @abc.abstractmethod
    def finalize_modules(self, model, config) -> None:
        """Post-load: finalize fused kernel state + register remap tables.

        ``model`` is the ``*ForCausalLM`` (so both ``.modules()`` and
        ``.model.lora_meta`` are reachable)."""

    # ---- forward-time ----
    @abc.abstractmethod
    def enter_decoder_stack(self, hidden_states, residual):
        """Turn the embedded/handed-off tensors into this adaptation's state."""

    @abc.abstractmethod
    def run_layer(self, layer, positions, state):
        """Run one decoder layer, threading and returning the state."""

    @abc.abstractmethod
    def to_intermediate(self, state) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Collapse state into two token-leading tensors for the PP wire.

        Returns ``(hidden_states, residual)`` — both token-leading ``[M, H]`` (or
        ``residual=None``) so vLLM's per-token IntermediateTensors slice is safe.
        """

    @abc.abstractmethod
    def exit_decoder_stack(self, state, adapter_indices, norm, config) -> torch.Tensor:
        """Last-rank finalize: produce the final ``[M, H]`` hidden states."""


# --------------------------------------------------------------------------- #
# LoRA (the host default)
# --------------------------------------------------------------------------- #
class LoRADecoderInterface(DecoderInterface):
    """Single-stream LoRA/aLoRA. Base defaults == today's LoRA forward."""

    def make_kernel_meta(self, device):
        return FusedLoRAKernelMeta(device=device), LoRAContext()

    def make_decoder_layer(self, vllm_config, prefix):
        return GraniteSwitchDecoderLayer(vllm_config=vllm_config, prefix=prefix)

    def ctx_wire_types(self):
        return (
            SwitchedLoRALinear,
            GraniteLoRAEmbeddedAttention,
            GraniteSwitchDecoderLayer,
        )

    def prepare_kernel_meta(self, lora_meta, adapter_indices, lora_ctx) -> None:
        lora_meta.prepare_and_store(adapter_indices, lora_ctx)

    def load_weights(self, model, weights):
        """Load model weights from checkpoint.

        Handles two checkpoint formats:

        1. **Composed checkpoints** (from compose_granite_switch.py): parameter
           names match the vLLM model exactly — loaded directly.
        2. **HuggingFace checkpoints** (from save_pretrained): MoE expert weights
           use a stacked format that must be split into per-expert tensors for
           vLLM's FusedMoE layer.

           HF format → vLLM format:
           - block_sparse_moe.input_linear.weight [E, 2*I, H]
             → experts.w13_weight via weight_loader(shard_id="w1"/"w3", expert_id=e)
           - block_sparse_moe.output_linear.weight [E, H, I]
             → experts.w2_weight via weight_loader(shard_id="w2", expert_id=e)
           - block_sparse_moe.router.layer.weight [E, H]
             → block_sparse_moe.gate.weight (direct rename)
        """
        params_dict = dict(model.named_parameters())
        loaded_params: set = set()

        def _load_direct(name, loaded_weight):
            """Load a weight directly by name."""
            if name.endswith(".bias") and name not in params_dict:
                return
            if is_pp_missing_parameter(name, model):
                return
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(
                    param,
                    "weight_loader",
                    default_weight_loader,
                )
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        def _load_expert(param_name, loaded_weight, weight_name, shard_id, expert_id):
            """Load a per-expert weight into a FusedMoE packed parameter."""
            if is_pp_missing_parameter(param_name, model):
                return
            if param_name not in params_dict:
                return
            param = params_dict[param_name]
            weight_loader = param.weight_loader
            weight_loader(
                param,
                loaded_weight,
                weight_name,
                shard_id=shard_id,
                expert_id=expert_id,
            )
            loaded_params.add(param_name)

        for name, loaded_weight in weights:
            # ── HF stacked MoE: input_linear → per-expert w1/w3 ──
            if name.endswith(".block_sparse_moe.input_linear.weight"):
                for e in range(loaded_weight.size(0)):
                    w1_name = name.replace(
                        ".block_sparse_moe.input_linear.weight",
                        f".block_sparse_moe.experts.{e}.w1.weight",
                    )
                    w3_name = name.replace(
                        ".block_sparse_moe.input_linear.weight",
                        f".block_sparse_moe.experts.{e}.w3.weight",
                    )
                    w1_param, w3_param = loaded_weight[e].chunk(2, dim=0)
                    _load_expert(
                        name.replace(".input_linear.", ".experts.w13_"),
                        w1_param,
                        w1_name,
                        shard_id="w1",
                        expert_id=e,
                    )
                    _load_expert(
                        name.replace(".input_linear.", ".experts.w13_"),
                        w3_param,
                        w3_name,
                        shard_id="w3",
                        expert_id=e,
                    )
                continue

            # ── HF stacked MoE: output_linear → per-expert w2 ──
            if name.endswith(".block_sparse_moe.output_linear.weight"):
                for e in range(loaded_weight.size(0)):
                    w2_name = name.replace(
                        ".block_sparse_moe.output_linear.weight",
                        f".block_sparse_moe.experts.{e}.w2.weight",
                    )
                    _load_expert(
                        name.replace(".output_linear.", ".experts.w2_"),
                        loaded_weight[e],
                        w2_name,
                        shard_id="w2",
                        expert_id=e,
                    )
                continue

            # ── HF MoE router → gate ──
            if name.endswith(".block_sparse_moe.router.layer.weight"):
                gate_name = name.replace(
                    ".block_sparse_moe.router.layer.weight",
                    ".block_sparse_moe.gate.weight",
                )
                _load_direct(gate_name, loaded_weight)
                continue

            # ── Direct load (built checkpoints + all non-MoE weights) ──
            _load_direct(name, loaded_weight)

        # Report unloaded parameters
        unloaded_params = [n for n in params_dict if n not in loaded_params]
        if unloaded_params:
            shown = unloaded_params[:10]
            suffix = (
                f"\n  ... and {len(unloaded_params) - 10} more"
                if len(unloaded_params) > 10
                else ""
            )
            logger.warning(
                "%d parameters were not loaded from checkpoint:\n%s%s",
                len(unloaded_params),
                "\n".join(f"  - {n}" for n in shown),
                suffix,
            )

        # Finalize fused LoRA weights (build w_ext from loaded lora_A + base weights)
        self.finalize_modules(model, model.config)
        return loaded_params

    def finalize_modules(self, model, config) -> None:
        if not hasattr(config, "adapter_ranks") or config.adapter_ranks is None:
            return
        adapter_ranks = config.adapter_ranks
        for module in model.modules():
            if isinstance(module, SwitchedLoRALinear):
                module.finalize_weights(adapter_ranks)

        # Assign sequential indices to all SwitchedLoRALinear modules and register
        # their remap tables so lora_meta can compute exact per-module bitmasks at
        # forward time.
        if model.model.lora_meta is not None:
            lora_modules = [
                m for m in model.modules() if isinstance(m, SwitchedLoRALinear)
            ]
            for idx, m in enumerate(lora_modules):
                m._module_idx = idx
            all_remap_tables = torch.stack([m.remap_table for m in lora_modules], dim=0)
            module_cfg_keys = [m._block_cfg_key for m in lora_modules]
            model.model.lora_meta.register_remap_tables(
                all_remap_tables, module_cfg_keys
            )

    def enter_decoder_stack(self, hidden_states, residual):
        return LoRAStackState(hidden_states=hidden_states, residual=residual)

    def run_layer(self, layer, positions, state):
        hidden_states, residual = layer(
            positions=positions,
            hidden_states=state.hidden_states,
            residual=state.residual,
        )
        return LoRAStackState(hidden_states=hidden_states, residual=residual)

    def to_intermediate(self, state):
        return state.hidden_states, state.residual

    def exit_decoder_stack(self, state, adapter_indices, norm, config):
        # Fold in the last residual via rms_norm_select so the same fused/separate
        # convention is used throughout (bit-exact with the original vLLM class).
        hidden_states, _ = rms_norm_select(
            norm,
            state.hidden_states,
            state.residual,
            config.fused_add_norm,
        )
        return hidden_states


# --------------------------------------------------------------------------- #
# Shadow-Residual
# --------------------------------------------------------------------------- #
class SRDecoderInterface(DecoderInterface):
    """Dual-stream SR: base ++ adapter stacked ``[2M, H]``, merged per token.

    The doubling is per *stream*, not per adapter: the adapter half carries each
    token's own real adapter id, and K/V always comes from the base half, so
    tokens on different adapters never reach each other through attention.

    The ``[2M, H]`` stack lives strictly intra-rank. At a PP boundary only two
    token-leading ``[M, H]`` halves cross (via :meth:`to_intermediate`); each
    rank re-stacks them at :meth:`enter_decoder_stack`. Nothing ``2M`` is ever
    sent, so vLLM's per-token IntermediateTensors slice stays correct.
    """

    def make_kernel_meta(self, device):
        return SRFusedLoRAKernelMeta(device=device), SRLoRAContext()

    def make_decoder_layer(self, vllm_config, prefix):
        return ShadowResidualDecoderLayer(vllm_config=vllm_config, prefix=prefix)

    def ctx_wire_types(self):
        # SwitchedLoRALinear projections read the 2M metadata; WCrossShunt reads
        # the M-length real-id metadata. Both index the same tables by _module_idx.
        return (SwitchedLoRALinear, WCrossShunt)

    def prepare_kernel_meta(self, lora_meta, adapter_indices, lora_ctx) -> None:
        # Prepares BOTH layouts (2M for projections, M-real for the shunt) from
        # the token-leading [M] real ids — so every PP rank reconstructs the
        # kernel metadata locally.
        lora_meta.prepare_and_store_sr(adapter_indices, lora_ctx)

    def load_weights(self, model, weights):
        """Direct-by-name load of a PRE-FUSED composed SR checkpoint.

        The composer pre-fuses q/k/v -> ``qkv_proj`` and gate/up ->
        ``shared_mlp.input_linear`` (see docs/SR_ARCHITECTURE.md; the config
        rejects ``unfused_qkv=True``), so a composed SR checkpoint's parameter
        names match the vLLM SR model 1:1. Every tensor loads directly by name
        through its own vLLM ``weight_loader`` — the fused base tensor's shape
        already matches ``*.base_layer.weight``, exactly like
        :meth:`LoRADecoderInterface.load_weights` loads a composed LoRA
        checkpoint. No fuse-at-load shard logic.

        SR-specific handling layered on the plain direct load:
          * skip ``_SR_SKIP_SUBSTRINGS`` (the WCrossShunt has no ``base_layer``;
            config-regenerated buffers);
          * mark the fused-qkv K/V LoRA slices loaded — they carry no delta
            (``lora_B`` is zero), so vLLM's strict init check must accept them
            whether or not the checkpoint ships them.
        """
        params = dict(model.named_parameters())
        loaded: set = set()

        def _load(pname: str, w) -> bool:
            if pname not in params:
                return False
            if is_pp_missing_parameter(pname, model):
                loaded.add(pname)
                return True
            param = params[pname]
            wl = getattr(param, "weight_loader", default_weight_loader)
            wl(param, w)
            loaded.add(pname)
            return True

        for name, w in weights:
            if any(s in name for s in _SR_SKIP_SUBSTRINGS):
                continue
            if name.endswith(".bias") and name not in params:
                continue
            _load(name, w)

        # Shared-KV: the K/V slices of the fused qkv LoRA carry no delta
        # (lora_B is zero), so mark them loaded — vLLM's strict init check must
        # accept them whether or not a checkpoint ships them.
        for pname in params:
            if (
                ".self_attn.qkv_proj.lora_A_slices.1" in pname
                or ".self_attn.qkv_proj.lora_A_slices.2" in pname
                or ".self_attn.qkv_proj.lora_B_slices.1" in pname
                or ".self_attn.qkv_proj.lora_B_slices.2" in pname
            ):
                loaded.add(pname)

        unloaded = [n for n in params if n not in loaded]
        if unloaded:
            logger.warning(
                "%d SR params not loaded from checkpoint:\n%s",
                len(unloaded),
                "\n".join(f"  - {n}" for n in unloaded[:15]),
            )

        self.finalize_modules(model, model.config)
        return loaded

    def finalize_modules(self, model, config) -> None:
        adapter_ranks = getattr(config, "adapter_ranks", None)
        if adapter_ranks is None:
            return
        cross_rank = int(getattr(config, "cross_stream_rank", 0) or 0)
        cross_ranks = [cross_rank] * config.num_adapters

        sll = [m for m in model.modules() if isinstance(m, SwitchedLoRALinear)]
        for m in sll:
            m.finalize_weights(adapter_ranks)
        shunts = [m for m in model.modules() if isinstance(m, WCrossShunt)]
        for m in shunts:
            m.finalize_weights(cross_ranks)

        # Shared _module_idx across BOTH the projections and the shunts, so the SR
        # kernel meta can index them by _module_idx in both the 2M and M-real layouts.
        if model.model.lora_meta is not None:
            all_modules = sll + shunts
            for idx, m in enumerate(all_modules):
                m._module_idx = idx
            all_remap_tables = torch.stack([m.remap_table for m in all_modules], dim=0)
            module_cfg_keys = [m._block_cfg_key for m in all_modules]
            model.model.lora_meta.register_remap_tables(
                all_remap_tables, module_cfg_keys
            )

    def enter_decoder_stack(self, hidden_states, residual):
        # First rank: a single embedding stream -> double it (base ++ adapter,
        # equal pre-invocation). Later ranks: hidden_states=base half,
        # residual=adapter half (see to_intermediate) -> re-stack.
        if residual is None:
            hs = torch.cat([hidden_states, hidden_states], dim=0)  # [2M, H]
        else:
            hs = torch.cat([hidden_states, residual], dim=0)  # [2M, H]
        return SRStackState(hs=hs)

    def run_layer(self, layer, positions, state):
        return SRStackState(hs=layer(positions, state.hs))

    def to_intermediate(self, state):
        # Split the [2M, H] stack into two token-leading [M, H] halves. The base
        # and adapter halves diverge after layer 0 (adapter accumulates deltas +
        # shunt), so BOTH must cross; residual carries the adapter half.
        m = state.hs.shape[0] // 2
        return state.hs[:m].contiguous(), state.hs[m:].contiguous()

    def exit_decoder_stack(self, state, adapter_indices, norm, config):
        hs = state.hs
        m = adapter_indices.shape[0]
        # Per-token merge: adapter-active tokens read the adapter stream, the rest
        # read the (base-equivalent) base stream. SR uses plain norm, not
        # rms_norm_select (its residual is materialized inside each layer).
        select = (adapter_indices > 0).unsqueeze(1)  # [M, 1]
        merged = torch.where(select, hs[m:], hs[:m])
        return norm(merged)


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def is_shadow_residual(config) -> bool:
    """SR checkpoints set ``cross_stream_rank`` (an int); LoRA leaves it ``None``."""
    return getattr(config, "cross_stream_rank", None) is not None


def select_decoder_interface(config) -> DecoderInterface:
    """Pick the decoder interface for ``config``.

    Keyed on ``config.cross_stream_rank`` rather than ``config.architectures``:
    the SR interface needs the ``cross_stream_rank`` value to build ``WCrossShunt``
    anyway, so the discriminant and the data are the same field.
    """
    return (
        SRDecoderInterface() if is_shadow_residual(config) else LoRADecoderInterface()
    )


__all__ = [
    "DecoderInterface",
    "LoRADecoderInterface",
    "SRDecoderInterface",
    "is_shadow_residual",
    "select_decoder_interface",
]
