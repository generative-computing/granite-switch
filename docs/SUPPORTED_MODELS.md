# Supported Models

Granite Switch supports Granite models. The architecture is detected
automatically from the HuggingFace `config.model_type` field.

## Feature Support

| Model Family | `model_type` | Support | KV Cache Hiding |
|---|---|---|:---:|
| Granite 4.x Dense | `granite` | **Full** | Yes |

- **Full**: Primary development target with comprehensive test coverage.

### Example Models

Any Granite model whose HuggingFace config has `model_type: granite` can be used
as a base model. The table below lists representative examples.

#### Granite 4.x (`granite`)

| Model Tag | Size | Variant |
|---|---|---|
| `ibm-granite/granite-4.1-3b` | 3B | Dense, instruct |
| `ibm-granite/granite-4.1-8b` | 8B | Dense, instruct |
| `ibm-granite/granite-4.1-30b` | 30B | Dense, instruct |

Base variants (`granite-4.1-3b-base`, `granite-4.1-8b-base`) are also supported.

## Target Layers

### Attention Layers

| Base Model PEFT Modules | Granite-Switch Layer Name | Full Parameter Path |
|---|---|---|
| `q_proj`, `k_proj`, `v_proj` (fused) | `qkv_proj` | `model.layers.{i}.self_attn.qkv_proj.lora_{A,B}_slices.{0,1,2}` |
| `o_proj` | `o_proj` | `model.layers.{i}.self_attn.o_proj.lora_{A,B}` |

### MLP Layers

Granite models use `mlp.gate_proj` / `mlp.up_proj` / `mlp.down_proj` in
the base model. These are remapped to the `shared_mlp` namespace:

| Base Model PEFT Modules | Granite-Switch Layer Name | Full Parameter Path |
|---|---|---|
| `gate_proj` + `up_proj` (fused) | `shared_input_linear` | `model.layers.{i}.shared_mlp.input_linear.lora_{A,B}_slices.{0,1}` |
| `down_proj` | `shared_output_linear` | `model.layers.{i}.shared_mlp.output_linear.lora_{A,B}` |

## Summary Matrix

| Architecture | `qkv_proj` | `o_proj` | `shared_input_linear` | `shared_output_linear` |
|---|:---:|:---:|:---:|:---:|
| Granite 4.x Dense | Y | Y | Y | Y |
