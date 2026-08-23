# Shadow Residual (SR) Dual-Stream Architecture

## Overview

Shadow Residual (SR) is a dual-stream inference architecture for Granite Switch. It runs a
**frozen base stream** and an **adapter stream** through the same layer, connected by a
low-rank `cross_stream` injection:

```
Input
  |
  +--- Base Stream (frozen) -----> Attn(Q, K, V from base) --> MLP --> h_base
  |                                                                      |
  |                                    cross_stream = lora_B @ lora_A @ h_base
  |                                                                      |
  +--- Adapter Stream (LoRA) ----> Attn(Q from adapter, K/V from base) -> MLP --> h_adapt
                                                                             + cross_stream
```

The base stream is the unmodified Granite model — the *same* modules, called with
`adapter_indices=None`, so the two streams share every parameter. The adapter stream applies
LoRA deltas to selected projections. `cross_stream` transfers information from base to
adapter at each layer; it is a `SwitchedLoRALinear` whose **base weight is zero**, so it
contributes nothing when no adapter is active.

K and V always come from the base stream (shared-base-KV). The adapter stream computes only
Q with LoRA, and a single KV cache is written from the base stream.

## One Model Class, Two Decoder Layers

There is **one** `GraniteSwitchModel` / `GraniteSwitchForCausalLM` pair, in
`src/granite_switch/hf/modeling_granite_switch.py`. It covers both stream modes and picks
its decoder layer class once, in `__init__`, from `config.dual_stream`:

| `config.dual_stream` | Decoder layer | Adapters |
|---|---|---|
| `False` (default) | `GraniteSwitchAttentionDecoderLayer` | standard LoRA / aLoRA |
| `True` | `SRSwitchDecoderLayer` | Shadow Residual |

`SRSwitchDecoderLayer` **subclasses** `GraniteSwitchAttentionDecoderLayer`. Because SR uses
the same fused projections as the single-stream layer (see below), its parameters are
identical to the parent's; the subclass adds only the layer-level `cross_stream` site in
`__init__` and a dual-stream `forward`. Subclassing rather than duplicating is deliberate:
an earlier standalone SR layer had its own `__init__` that never built `block_sparse_moe`,
silently dropping every expert on a MoE base. A subclass has no second `__init__` to forget
things in.

The checkpoint's `architectures` field is `["GraniteSwitchForCausalLM"]` in both modes —
transformers derives it from the class name, so one class is what makes one
`AutoModelForCausalLM` registration possible.

There are no `SRSwitch*` model classes and no back-compat aliases for them. Nothing needs
them: HF resolves a model class from the **config class** registered for `model_type:
granite_switch`, never from the `architectures` string, and any checkpoint old enough to
carry `architectures: ["SRSwitchForCausalLM"]` also carries `unfused_qkv` and is rejected
(see below). Note that vLLM *does* dispatch on `architectures`, so an SR-on-vLLM backend
must register `GraniteSwitchForCausalLM`.

## SR Uses Fused Projections

SR uses the **same fused projections** as the rest of the HF backend — `qkv_proj` and
`shared_mlp.input_linear` — symmetric with the vLLM backend:

- `qkv_proj`: one `[hidden_size, q_size + 2*kv_size]` matmul, then split
- `shared_mlp.input_linear`: one `[hidden_size, 2*intermediate_size]` matmul (gate + up)

Only the **base** weight is concatenated. LoRA is never fused: each PEFT module keeps its
own `(lora_A, lora_B)` pair in a slice, so a q-only SR adapter populates
`qkv_proj.lora_A_slices.0` and leaves slices 1 and 2 zero.

Earlier SR builds used unfused Q/K/V and gate/up/down projections to mirror the
shadow-residual training code's reduction order. That was dropped, for three reasons:

1. **MoE correctness** — the unfused path needed a separate SR layer class, which is where
   the dropped-experts bug came from.
2. **A vLLM SR path becomes possible** with no new weight layout, since vLLM is fused.
3. **It measured better.** On the full answerability eval (3565 rows, granite-4.1-3b),
   fused SR scored accuracy 0.8496 / weighted-F1 0.8313 against 0.844 unfused.

Consequence: **SR checkpoints composed before the fusion change must be re-composed.**
Their configs carry `unfused_qkv: True`, and since transformers silently keeps unknown
config keys, such a checkpoint would otherwise load into a fused model and quietly mismatch
keys. `GraniteSwitchConfig` therefore rejects `unfused_qkv=True` with a re-compose error.

## Adapter Weight Structure

An SR adapter is a PEFT adapter with an extra layer-level `cross_stream` module:

```
target_modules: ["q_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "cross_stream"]

base_model.model.model.layers.{i}.self_attn.q_proj.lora_A.weight   # [32, 2560]
base_model.model.model.layers.{i}.self_attn.q_proj.lora_B.weight   # [2560, 32]
base_model.model.model.layers.{i}.self_attn.o_proj.lora_A.weight   # [32, 2560]
base_model.model.model.layers.{i}.self_attn.o_proj.lora_B.weight   # [2560, 32]
base_model.model.model.layers.{i}.mlp.gate_proj.lora_A.weight      # [32, 2560]
base_model.model.model.layers.{i}.mlp.gate_proj.lora_B.weight      # [6912, 32]
base_model.model.model.layers.{i}.mlp.up_proj.lora_A.weight        # [32, 2560]
base_model.model.model.layers.{i}.mlp.up_proj.lora_B.weight        # [6912, 32]
base_model.model.model.layers.{i}.mlp.down_proj.lora_A.weight      # [32, 6912]
base_model.model.model.layers.{i}.mlp.down_proj.lora_B.weight      # [2560, 32]
base_model.model.model.layers.{i}.cross_stream.lora_A.weight       # [64, 2560]
base_model.model.model.layers.{i}.cross_stream.lora_B.weight       # [2560, 64]
```

The composer remaps these onto the fused sites:

| PEFT module | Composed slice |
|---|---|
| `q_proj` | `self_attn.qkv_proj.lora_A_slices.0` / `lora_B_slices.0` |
| `o_proj` | `self_attn.o_proj.lora_A` |
| `gate_proj` | `shared_mlp.input_linear.lora_A_slices.0` |
| `up_proj` | `shared_mlp.input_linear.lora_A_slices.1` |
| `down_proj` | `shared_mlp.output_linear.lora_A` |
| `cross_stream` | `cross_stream.lora_A` (layer-level, no parent) |

An SR adapter carrying `k_proj` or `v_proj` weights is **rejected**: K/V are read from the
base stream, so those weights could never be applied and silently ignoring them would be
worse than failing.

## One Kind of Adapter Per Checkpoint

A checkpoint holds **either** SR adapters **or** standard LoRA/aLoRA adapters, never both.
The decoder runs in one stream mode for the whole checkpoint, so there is no layout that
could serve both. `GraniteSwitchComposer.from_base_and_adapters` classifies every adapter
and raises `ValueError("Cannot mix Shadow Residual ...")` on a mixed set.

## Composer Auto-Detection

No manual configuration is needed — pass SR adapter paths to the composer and it will:

1. **Scan adapter weights** for a `cross_stream` module to classify each adapter.
2. **Reject a mixed set** of SR and non-SR adapters.
3. Set `dual_stream = True` when the adapters are SR (and leave both SR config fields unset
   otherwise, so a LoRA/aLoRA build allocates no `cross_stream` and has exactly the
   pre-SR parameter count).
4. **Detect `cross_stream_rank`** as the `max` across adapters from `adapter_config.json`'s
   `rank_pattern` — adapters may have been trained with different cross-stream ranks, and
   the allocated site must fit the largest.
5. **Resolve the SR arch** via `resolve_arch(..., dual_stream=True)`, which is the ordinary
   fused arch plus `_cross_stream_groups()` — no group replacement.
6. **Instantiate `GraniteSwitchForCausalLM`**, which selects `SRSwitchDecoderLayer`.

`validate_cross_stream_population` then asserts every adapter in a `dual_stream` checkpoint
has non-zero `cross_stream` weights, catching a silent load failure.

## Config Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dual_stream` | bool | `False` | Whole-checkpoint decoder mode; selects the decoder layer class |
| `cross_stream_rank` | int \| None | `None` | Rank of the `cross_stream` injection (from `rank_pattern`) |

The two are validated together: `dual_stream=True` requires `cross_stream_rank`, and
`dual_stream=False` requires it to be `None`. A config carrying the removed `unfused_qkv`
key raises a re-compose error.

## Key Files

| File | Role |
|------|------|
| `src/granite_switch/hf/modeling_granite_switch.py` | `GraniteSwitchForCausalLM`, `GraniteSwitchModel`, and both decoder layer classes |
| `src/granite_switch/hf/core/lora.py` | `SwitchedLoRALinear`, `forward_dual_stream`, `return_kv` |
| `src/granite_switch/composer/compose_utils.py` | SR detection, mixing rejection, `cross_stream_rank` |
| `src/granite_switch/composer/arch.py` | `_cross_stream_groups()`, `granite_dense_sr_arch`, `granite_moe_hybrid_sr_arch` |
| `src/granite_switch/composer/validator.py` | `validate_cross_stream_population` |
| `src/granite_switch/config.py` | `dual_stream` / `cross_stream_rank` and their validation |
