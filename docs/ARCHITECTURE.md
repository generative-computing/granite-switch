# Architecture

## Granite Switch Model

The Granite Switch extends the base Granite model with:

### 1. Embedded LoRA Adapters (frozen during inference)

Multiple task/domain-specific adapters are embedded in the same checkpoint. Each adapter has
LoRA weights (`lora_A`, `lora_B`) stacked in tensors and is activated via special control tokens
or router-selected indices.

### 2. Control Tokens

Each adapter has a control token `<|adapter|>` that fires the switch. KV hiding uses
group-based control dimensions (`K=finfo.min`, `Q=per-adapter policy`). Control tokens are
KV-hidden to prevent cross-request interference.

### 3. Chat Template Integration

The tokenizer chat template maps adapter names to control tokens and places them automatically
based on adapter type:

- **ALORA adapters**: token placed either in the user message (by matching the invocation
  sequence) or right before the generation prompt
- **LORA adapters**: token placed at sequence beginning

### 4. Optional Trainable Router (SingleSwitch)

SingleSwitch is a single attention head that uses a one-hot dim-0 pattern to compute per-token
adapter indices via attention-based cumsum. It has no decoder layers and no projection head —
only a vocab-size lookup table, so parameter cost is negligible relative to the full model.

---

## Two Backends

Both backends share the same checkpoint format (`save_pretrained` / `from_pretrained`).

### HuggingFace Backend (`granite_switch.hf`)

Full `transformers` integration (`PreTrainedModel`, `GenerationMixin`). Used for training and
debugging. Uses fused QKV and gate-up projections, which changes floating-point reduction order
relative to the upstream `GraniteMoeHybridForCausalLM` (see Common Gotchas #9 in `CLAUDE.md`).

### vLLM Backend (`granite_switch.vllm`)

Production inference backend (10-20x speedup). Uses Punica kernels for optimized LoRA
computation, PagedAttention for efficient KV cache, and supports continuous batching and
tensor/pipeline parallelism. Registered as a vLLM plugin via the `granite_switch.vllm` entry point.

---

## Key Configuration Fields

These fields are specific to Granite Switch and not present in base Granite:

| Field | Description |
|---|---|
| `num_adapters` | Number of embedded LoRA adapters |
| `adapter_token_ids` | Token IDs for each adapter's control token |
| `adapter_names` | Human-readable names for each adapter |
| `hiding_groups` | Named groups of adapters for KV hiding |
| `hiding_policy` | Per-adapter KV hiding rules |
| `lora_rank` | LoRA rank (same for all adapters) |
| `lora_alpha` | LoRA alpha scaling factor |
| `control_dims` | Number of KV dimensions reserved for control |

### Granite-Specific Parameters (inherited from base model)

- **`attention_multiplier`**: Attention score scaling (replaces `1/sqrt(head_dim)`)
- **`logits_scaling`**: Applied to final logits
- **`residual_multiplier`**: Applied to residual connections
- **`embedding_multiplier`**: Applied to input embeddings

Always load these from config — never hardcode.
