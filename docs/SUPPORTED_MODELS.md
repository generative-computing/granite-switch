# Supported Models

Granite Switch supports Granite models. The architecture is detected
automatically from the HuggingFace `config.model_type` field.

## Feature Support

| Model Family | `model_type` | Support | KV Cache Hiding |
|---|---|---|:---:|
| Granite 4.0 / 4.1 Dense | `granite` | **Full** | Yes |
| Granite 4.2 Dense (ChatML) | `granite` | **Full** | Yes |

- **Full**: Primary development target with comprehensive test coverage.

### Chat-template formats

Granite ships two chat-template families, both fully supported. The format is
auto-detected from the base tokenizer's template at compose time (see
`detect_template_format` in `composer/tokenizer_setup.py`):

| Format | Models | Role markers | Word embeddings |
|---|---|---|---|
| `granite_format` | 4.0 / 4.1 | `<\|start_of_role\|>ROLE<\|end_of_role\|>` … `<\|end_of_text\|>` | tied |
| `chatml` | 4.2 | `<\|im_start\|>ROLE\n` … `<\|im_end\|>`, plus a `<think>` block | untied |

Adapter control-token injection (LoRA prefix, ALoRA user-message invocation,
ALoRA assistant-boundary fallback) works identically for both formats. Granite
4.2 additionally sets `tie_word_embeddings: false`; the composer preserves the
distinct LM head.

For both formats the composer initializes every new control-token output row
from a reserved `<|unused_N|>` row, so a control token is as unlikely to be
emitted as a token the base model was trained never to emit. This runs on the
tied path (4.0/4.1) as well as the untied one: a control token's *input* row is
never read, because the switch rewrites the control-token id to its
token-exchange substitute before the embedding lookup, so writing the shared
matrix affects only the output side. See `initialize_control_token_output_rows`
in `composer/compose_granite_switch.py`.

Audio input (`--enable-audio`) is also supported on both formats — the `<|audio|>`
marker injection is format-aware. The two templates need different treatment
because ChatML has no content-part loop at all; see
[AUDIO.md](AUDIO.md#openai-compatible-server--chat-api).

#### Multi-turn KV policy on 4.2

`KVHistoryPolicy.RE_PREFILL` works on both formats with no extra arguments.
`KVHistoryPolicy.PRESERVE_MIXED_HISTORY` reuses the exact ids it already sent, so
it needs a template whose render only ever grows. On `chatml` that means passing
both flags to `build_prompt()` on **every** turn:

```python
conv.build_prompt(adapter="uncertainty",
                  enable_thinking=False,
                  truncate_history_thinking=False)
```

| flag | default | why `PRESERVE_MIXED_HISTORY` needs it off |
|---|---|---|
| `enable_thinking` | `True` | The generation prompt ends inside an open `<think>`, while a completed turn is past `</think>`, so the assistant turn terminator cannot be derived. Turning it off costs the model's reasoning. |
| `truncate_history_thinking` | `True` | Strips reasoning from assistant turns older than the newest user message, rewriting bytes that have already been sent. |

Both are refused rather than mis-served, but the errors are generic -- one reports
that the turn terminator cannot be derived, the other that the template is not
append-only, and both suggest `RE_PREFILL`. On `chatml` the remedy is the flags
above. `granite_format` needs neither flag -- it has no `<think>` block.

### Example Models

Any Granite model whose HuggingFace config has `model_type: granite` can be used
as a base model. The table below lists representative examples.

**Note:** Granite Switch currently supports single-GPU inference only. Models
that do not fit in a single GPU's memory are not yet supported.

#### Granite 4.x (`granite`)

| Model Tag | Size | Variant |
|---|---|---|
| `ibm-granite/granite-4.1-3b` | 3B | Dense, instruct (role-marker template) |
| `ibm-granite/granite-4.1-8b` | 8B | Dense, instruct (role-marker template) |
| `ibm-granite/granite-4.0-micro` | 3B | Dense, instruct (role-marker template) |
| Granite 4.2 3B | 3B | Dense, instruct (ChatML template, untied embeddings) |

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
