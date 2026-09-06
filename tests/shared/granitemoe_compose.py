# SPDX-License-Identifier: Apache-2.0
"""Build a tiny composed ``granitemoe`` checkpoint on disk, on CPU, no download.

``granitemoe`` is a *pure sparse* MoE: every layer has an expert bank and no
dense ``shared_mlp``, which upstream encodes as ``shared_intermediate_size == 0``.

These builders exist as shared utilities rather than test-local helpers because
two suites need the same checkpoint for opposite reasons:

* ``tests/composer/test_granitemoe_compose_e2e.py`` — that the shared MLP and its
  LoRA target groups disappear together while the frozen expert tensors transfer;
* ``tests/vllm/test_tp_integration.py`` — that the HF-stacked -> ``FusedMoE``
  remap still lands the right bytes when ``FusedMoE`` shards the expert bank
  across tensor-parallel ranks.

The LoRA weights here are drawn from ``torch.randn``, **including ``lora_B``**.
That is load-bearing for the TP caller: ``SwitchedLoRALinear`` zero-initializes
``lora_B`` (``hf/core/lora.py``), so a synthetic switch model built by
``tests.shared.generation_models.save_switch_model`` has an identically-zero
adapter delta, and any comparison over it silently degenerates into comparing two
base-only runs.  Composing a real PEFT adapter is what makes the adapter live.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import torch
from safetensors.torch import save_file

# Defaults reproduce the geometry the composer e2e suite has always used.
LORA_RANK = 4
CROSS_STREAM_RANK = 8
NUM_LAYERS = 2
HIDDEN = 64
INTERMEDIATE = 32
NUM_EXPERTS = 4
TOP_K = 2
VOCAB_SIZE = 300
CONTROL_TOKEN_ID = 250
HEAD_DIM = 16


@dataclass(frozen=True)
class MoeGeometry:
    """Shape of the synthetic base.

    ``num_attention_heads * head_dim == hidden`` is required: the projections
    below are written square, as every real Granite config has them.
    """

    hidden: int = HIDDEN
    head_dim: int = HEAD_DIM
    num_attention_heads: int = 4
    num_key_value_heads: int = 4
    num_layers: int = NUM_LAYERS
    intermediate: int = INTERMEDIATE
    num_experts: int = NUM_EXPERTS
    top_k: int = TOP_K
    vocab_size: int = VOCAB_SIZE
    max_position_embeddings: int = 128
    lora_rank: int = LORA_RANK
    cross_stream_rank: int = CROSS_STREAM_RANK

    def __post_init__(self):
        if self.num_attention_heads * self.head_dim != self.hidden:
            raise ValueError(
                f"num_attention_heads * head_dim ({self.num_attention_heads} * "
                f"{self.head_dim}) must equal hidden ({self.hidden})"
            )


DEFAULT_GEOMETRY = MoeGeometry()

# head_dim 16 is below what vLLM's attention backends accept, so a GPU caller
# needs a wider head.  Keeping 4 attention heads means TP=2 gives each rank 2
# heads, which a 1-head-per-rank shape would not distinguish from replication.
GPU_GEOMETRY = MoeGeometry(hidden=128, head_dim=32)


def create_base_model(path: Path, geometry: MoeGeometry = DEFAULT_GEOMETRY):
    """Create a tiny ``granitemoe`` base checkpoint on disk.

    ``shared_intermediate_size`` is deliberately absent from the config, exactly
    as in the real ``granitemoe`` checkpoints: the arch descriptor is what pins
    it to 0, not the base config.
    """
    g = geometry
    path.mkdir(parents=True, exist_ok=True)
    base_cfg = {
        "model_type": "granitemoe",
        "architectures": ["GraniteMoeForCausalLM"],
        "vocab_size": g.vocab_size,
        "hidden_size": g.hidden,
        "intermediate_size": g.intermediate,
        "num_hidden_layers": g.num_layers,
        "num_attention_heads": g.num_attention_heads,
        "num_key_value_heads": g.num_key_value_heads,
        "num_local_experts": g.num_experts,
        "num_experts_per_tok": g.top_k,
        "max_position_embeddings": g.max_position_embeddings,
        "rms_norm_eps": 1e-5,
        "attention_multiplier": 1.0,
        "logits_scaling": 1.0,
        "residual_multiplier": 1.0,
        "embedding_multiplier": 1.0,
        "torch_dtype": "float32",
        "head_dim": g.head_dim,
    }
    (path / "config.json").write_text(json.dumps(base_cfg))

    torch.manual_seed(0)
    state_dict = {"model.embed_tokens.weight": torch.randn(g.vocab_size, g.hidden)}

    for i in range(g.num_layers):
        prefix = f"model.layers.{i}"
        for mod in ("q_proj", "k_proj", "v_proj", "o_proj"):
            state_dict[f"{prefix}.self_attn.{mod}.weight"] = torch.randn(
                g.hidden, g.hidden
            )
        # Sparse expert bank: (E, 2*intermediate, hidden) / (E, hidden, intermediate)
        # / (E, hidden), named identically in the switch model.
        moe = f"{prefix}.block_sparse_moe"
        state_dict[f"{moe}.input_linear.weight"] = torch.randn(
            g.num_experts, 2 * g.intermediate, g.hidden
        )
        state_dict[f"{moe}.output_linear.weight"] = torch.randn(
            g.num_experts, g.hidden, g.intermediate
        )
        state_dict[f"{moe}.router.layer.weight"] = torch.randn(g.num_experts, g.hidden)
        state_dict[f"{prefix}.input_layernorm.weight"] = torch.ones(g.hidden)
        state_dict[f"{prefix}.post_attention_layernorm.weight"] = torch.ones(g.hidden)

    state_dict["model.norm.weight"] = torch.ones(g.hidden)
    state_dict["lm_head.weight"] = torch.randn(g.vocab_size, g.hidden)

    save_file(state_dict, str(path / "model.safetensors"))


def create_lora_adapter(path: Path, geometry: MoeGeometry = DEFAULT_GEOMETRY):
    """Create an attention-only (q/k/v/o) LoRA adapter.

    Mirrors the validated answerability adapter on a pure sparse base: no MLP
    targets at all, since LoRA on fused 3D expert parameters is not supported.
    """
    g = geometry
    path.mkdir(parents=True, exist_ok=True)
    config = {
        "r": g.lora_rank,
        "lora_alpha": g.lora_rank,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "peft_type": "LORA",
    }
    (path / "adapter_config.json").write_text(json.dumps(config))

    torch.manual_seed(42)
    state_dict = {}
    for layer_idx in range(g.num_layers):
        lp = f"base_model.model.model.layers.{layer_idx}.self_attn"
        for mod in ("q_proj", "k_proj", "v_proj", "o_proj"):
            state_dict[f"{lp}.{mod}.lora_A.weight"] = torch.randn(g.lora_rank, g.hidden)
            state_dict[f"{lp}.{mod}.lora_B.weight"] = torch.randn(g.hidden, g.lora_rank)

    save_file(state_dict, str(path / "adapter_model.safetensors"))


def create_sr_adapter(path: Path, geometry: MoeGeometry = DEFAULT_GEOMETRY):
    """Create a Shadow Residual adapter for a ``granitemoe`` base.

    Targets ``cross_stream`` plus q/o only: SR reads K/V from the base stream,
    and there is no shared MLP to target.
    """
    g = geometry
    path.mkdir(parents=True, exist_ok=True)
    config = {
        "r": g.lora_rank,
        "lora_alpha": g.lora_rank,
        "target_modules": ["q_proj", "o_proj", "cross_stream"],
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "peft_type": "LORA",
        "rank_pattern": {"cross_stream": g.cross_stream_rank},
        "alpha_pattern": {"cross_stream": g.cross_stream_rank},
    }
    (path / "adapter_config.json").write_text(json.dumps(config))

    torch.manual_seed(7)
    state_dict = {}
    for layer_idx in range(g.num_layers):
        lp = f"base_model.model.model.layers.{layer_idx}"
        for mod in ("q_proj", "o_proj"):
            state_dict[f"{lp}.self_attn.{mod}.lora_A.weight"] = torch.randn(
                g.lora_rank, g.hidden
            )
            state_dict[f"{lp}.self_attn.{mod}.lora_B.weight"] = torch.randn(
                g.hidden, g.lora_rank
            )
        state_dict[f"{lp}.cross_stream.lora_A.weight"] = torch.randn(
            g.cross_stream_rank, g.hidden
        )
        state_dict[f"{lp}.cross_stream.lora_B.weight"] = torch.randn(
            g.hidden, g.cross_stream_rank
        )

    save_file(state_dict, str(path / "adapter_model.safetensors"))


ADAPTER_BUILDERS: dict[str, Callable[..., None]] = {
    "lora": create_lora_adapter,
    "sr": create_sr_adapter,
}


class Build(NamedTuple):
    output_dir: Path
    model: object
    base_path: Path


def compose_granitemoe(
    *,
    name: str,
    base_path: Path,
    adapter_path: Path,
    output_dir: Path,
    make_adapter: Callable[..., None],
    geometry: MoeGeometry = DEFAULT_GEOMETRY,
) -> Build:
    """Compose a tiny granitemoe model and save it to disk."""
    from granite_switch.composer import GraniteSwitchComposer

    create_base_model(base_path, geometry)
    make_adapter(adapter_path, geometry)

    model = GraniteSwitchComposer.from_base_and_adapters(
        base_model_name_or_path=str(base_path),
        adapter_paths=[str(adapter_path)],
        adapter_token_ids=[CONTROL_TOKEN_ID],
        adapter_substitute_token_ids=[1],
        adapter_names=[name],
    )
    model.save_pretrained(str(output_dir))
    return Build(output_dir, model, base_path)
