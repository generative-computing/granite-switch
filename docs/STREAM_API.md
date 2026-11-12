# The Stream API: Head Streams and How to Write a Head Type

Every slot in a Granite Switch checkpoint is a **stream**: its control token turns it on, and its
**stream type** decides what runs at its tokens. A slot is either an adapter or a head:

| | Adapter stream (`lora`) | Head stream (`probe`, or a registered type) |
|---|---|---|
| What runs | LoRA, aLoRA or Shadow Residual weights inside every decoder layer | A small module at one layer, at the slot's marker token |
| Where its code lives | The core's fused LoRA kernels (built in) | The type's own HF and vLLM modules |
| What it emits | Text, through the LM head | One token: the head's scores go into the marker's logits row |
| Where its control token goes | Where its adapter was trained (start, before the invocation, generation prompt) | Last in the prompt (the chat template puts it there) |

A head's control token is its **marker**. The head reads the hidden state at the marker, and the
model rewrites that one row of logits. A head can read the base model's state, or the state one
named adapter computes (a *pair*, see [Heads on adapters](#heads-on-adapters)).

The probe ([activation probe slots](../tutorials/guides/activation_probe_slots.md)) is the first
head type. A new type needs three things and no change to Granite Switch:

1. **a config entry**, written by compose from the manifest;
2. **forward code**: a `HeadType` subclass that builds the module, for HF and vLLM;
3. **weights**: a safetensors file in each slot's folder.

## The stream entry

`config.json` holds one entry per slot under `streams`, in the order of the control tokens:

```json
"streams": [
  {"type": "lora"},
  {"type": "probe", "read_layer": 20, "head_row": 0,
   "output": {"mode": "select", "token_ids": [9891, 2201]}},
  {"type": "probe", "read_layer": 20, "head_row": 1, "adapter": "answerability",
   "output": {"mode": "select", "token_ids": [9891, 2201]}},
  {"type": "mlp_head", "read_layer": null, "head_row": 0,
   "output": {"mode": "merge", "token_ids": [9891, 2201], "weight": 0.5},
   "settings": {"width": 32}}
]
```

| Field | Meaning |
|---|---|
| `type` | `"lora"` for an adapter slot, or a registered head type's name |
| `read_layer` | The decoder layer the head reads, from 0; `null` reads after the final norm |
| `head_row` | The slot's row in its type's weight bank. Each type numbers its own bank; compose assigns them |
| `adapter` | Optional: an adapter slot (by name) that runs at the marker before the head reads. Omitted means the base model |
| `output.mode` | `select` or `merge` (below) |
| `output.token_ids` | The tokens the scores land on, one per score |
| `output.weight` | The scale for `merge` (default 1.0) |
| `settings` | Anything type-specific; the type's `check` validates it |

An adapter slot's entry is just `{"type": "lora"}`: its control token, substitute token and
template placement stay in the `adapter_*` config fields.

**Output modes.** A head's scores always go into its marker's logits row:

- `select`: every logit is set to −∞, then the scores go onto `token_ids`. The request can only
  emit one of those tokens (a classifier verdict, as the probe does).
- `merge`: the scores, times `weight`, are added to the LM head's logits on `token_ids`. Every
  other logit is unchanged (a head that nudges the model).

**Refusals.** A version refuses, by name, what it doesn't know: an unregistered type, an output
mode it doesn't have, or a field in an entry (or in its `output`) that it doesn't have. A
`read_layer` outside the model's layers is refused too. So an entry written by a newer version
fails loudly instead of running as something else.

**Older checkpoints.** A checkpoint without `streams` loads as before: its entries are rebuilt from
the older fields (`adapter_kinds`, `probe_label_token_ids`, `probe_read_layers`), in the layout it
was written in. A streams checkpoint also keeps those fields, derived from the entries, for older
versions to read; the entries are what this version uses.

## Writing a head type

A head type is a `HeadType` subclass, registered by name:

```python
from granite_switch.streams import HeadType, register_head_type


@register_head_type
class MLPHeadType(HeadType):
    name = "mlp_head"                    # the entry's "type"
    weights_file = "mlp_head.safetensors"  # the file each slot folder ships

    def check(self, spec, config):       # optional: validate spec.settings
        ...

    def hf_module(self, specs, config):  # one module holding every slot of this type
        ...

    def load_weights(self, module, row, state, spec):  # copy one slot's file into its row
        ...
```

| Member | Required | What it does |
|---|---|---|
| `name` | yes | The registry key, and the `type` of the entries it serves |
| `hf_module(specs, config)` | yes | Builds the module for every slot of this type (`specs`, one `StreamSpec` each) |
| `load_weights(module, row, state, spec)` | yes | Copies one slot's trained weights (`state`, read from `weights_file`) into row `row` of the module. Compose calls it |
| `weights_file` | no | The file a slot folder ships (default `head.safetensors`) |
| `module_name` | no | The model attribute, and so the checkpoint key prefix, of the module (default `stream_<name>`) |
| `check(spec, config)` | no | Validates one slot's `settings`; raise `ValueError` |
| `vllm_module(specs, config)` | no | The vLLM module. Defaults to the HF module, which must then follow the vLLM rules below |

A complete example, about 60 lines, is
[`tests/shared/stream_plugins/mlp_head.py`](../tests/shared/stream_plugins/mlp_head.py): a small
two-layer MLP per slot, in `merge` mode.

### The module's one call

```python
scores = module(hidden, rows)
```

| Argument | Shape | What it holds |
|---|---|---|
| `hidden` | `[n, hidden_size]` | The read layer's hidden state, one row per token passed in |
| `rows` | `[n]` | Which slot's weights each token uses: the slot's `head_row` + 1. A 0 means "no slot of this type here" |
| returns | `[n, width]` | One score per output token of the slot, in the order of its `token_ids`; `width` is at least the slot's number of outputs. Values on rows with 0 are ignored |

Granite Switch decides where the module runs and on what: the layer, the tokens and the rows. The
module only does the math. It never sees control tokens, positions or logits.

- **HF** passes only the marker tokens.
- **vLLM** passes every token of the step, because the forward pass runs inside a CUDA graph,
  which can't pick "only the markers" by looking at the data. A 0 in `rows` marks the tokens that
  don't count.

### Rules for the module on vLLM

vLLM compiles the forward pass and replays it as a CUDA graph. A graph repeats fixed GPU work: it
can't hand a value back to Python, or change a shape, from one step to the next. The module runs
inside it.

- **Nothing read back to Python:** no `.item()`, `.tolist()`, `.cpu()`, and no `if` on a tensor's
  value.
- **No shapes that depend on the data:** no boolean-mask indexing and no `nonzero()`. Pick each
  token's weights with `rows`, as the example does.
- **Input:** the hidden state in the model's dtype (bf16 when serving), with the full hidden size.
- **Output:** any float dtype. The scores travel to the logits in the model's dtype, so expect
  bf16 rounding.
- **The same parameters on both backends:** the HF and vLLM modules load the same checkpoint, by
  name, so their parameter names and shapes must match.
- **More than one GPU:** with tensor parallelism every GPU holds the whole module and runs it on
  the whole hidden state, so don't build it from vLLM's parallel layers. With pipeline parallelism
  it runs on the stage that holds its read layer, and the core passes the scores on.

### What a head costs

On HF the module runs on the markers only. On vLLM it runs on every token of every step, prefill
and decode, once per read layer its slots use, so its cost grows with the step's tokens (up to the
prefill budget), not with the number of markers. The probe is one matmul over all of them, which is
small next to one decoder layer. Keep a type's work per token far below a decoder layer's.

### Registering a plugin

A built-in type registers when `granite_switch.streams.types` is imported. A plugin package
registers through the `granite_switch.streams` entry-point group, and Granite Switch loads it the
first time a checkpoint names the type:

```toml
# pyproject.toml of the plugin's package
[project.entry-points."granite_switch.streams"]
mlp_head = "my_streams.mlp_head:MLPHeadType"
```

The entry point is what makes the type visible to every process: compose, HF, and vLLM's engine and
worker processes, which import nothing from the caller.

- **Name the entry point after the type:** its name must be the class's `name`.
- **Only the named entry point is imported,** so a plugin that fails to import affects only the
  checkpoints that use it.
- **A name belongs to one class:** registering a second class under a taken name is refused, so a
  plugin can't replace the built-in `probe`.

## Composing a head

A head is an entry in the compose manifest, next to the adapters:

```yaml
answerability:
  path: ./rag/answerability/lora
  type: lora
tox:                        # a probe on the base model
  path: ./tox               # holds probe_head.safetensors
  kind: probe               # the stream type
  labels: [yes, no]         # the output words; each must be a single token
  layer: 20                 # the read layer; omit for the final norm
tox_rag:                    # the same kind of head, on the answerability adapter
  path: ./tox_rag
  kind: probe
  labels: [yes, no]
  layer: 20
  adapter: answerability
nudge:                      # a plugin type, in merge mode
  path: ./nudge             # holds mlp_head.safetensors
  kind: mlp_head
  labels: [yes, no]
  output: {mode: merge, weight: 0.5}
  settings: {width: 32}
```

| Key | Meaning |
|---|---|
| `kind` | The stream type. Omitted, or `lora`, means an adapter |
| `labels` | The output words, turned into `output.token_ids` |
| `layer` | The read layer (`read_layer`) |
| `adapter` | An adapter slot of the same checkpoint that runs at the marker |
| `output` | `mode`, `weight`, or explicit `token_ids` |
| `settings` | Passed to the type, which checks it |

A head entry's `type` is not used: a head's marker always goes last. For each head slot, compose
looks up the type, turns the labels into token ids, gives the slot the next row of its type's bank,
asks the type to check the entry, loads the folder's `weights_file` through `load_weights`, and puts
the marker last in the chat template. It prints one line per head:

```
Head streams:
  tox (probe) → select over token ids [9891, 2201] @ layer 20
  tox_rag (probe on answerability) → select over token ids [9891, 2201] @ layer 20
  nudge (mlp_head) → merge over token ids [9891, 2201] @ final norm
```

**Weights layout.** All slots of a type share one module, one row per slot: the probe's bank is
`model.probe_head.*`, another type's is `model.stream_<type>.*`. Heads take no LoRA rows: the LoRA
tensors hold rows for adapter slots only, and slots may come in any order.

## Heads on adapters

MultiSwitch gives every token one slot, and the latest control token wins. A head's marker is a
control token too, so at the marker no adapter is active and a head reads the base model. A head
whose entry names an `adapter` (a *pair*) runs that adapter's LoRA at its marker instead. Each pair
is its own slot with its own control token, fixed at compose time; a request still names one slot.

For the head to read exactly the state the adapter computes on its own, the chat template renders
a pair like this:

| The adapter | What the template places | Why |
|---|---|---|
| Shadow Residual | The marker alone | The adapter writes no K/V, so running it at one token gives the same state as running it on every token |
| LoRA | The adapter's own token where it goes alone, then the marker last | A LoRA changes the K/V of every token it runs on, so it must run from its own token |
| aLoRA invoked inside the user message | The same as a LoRA | The same reason |
| aLoRA invoked at the assistant header | The marker alone | The adapter would start where the marker sits, on the same token: both stand for the role-open token |

Measured on granite-4.1-3b with real adapters (HF, fp32): the pair's head reads exactly the state
its adapter computes alone, for the answerability LoRA, the requirement-check aLoRA (invoked in the
message) and the answerability aLoRA (invoked at the header).

Each pair costs a slot, a control token and a bank row, even when the same head is reused on
several adapters.

## Multi-turn conversations

`Conversation` treats a head turn as a question about the conversation, not a turn in it:

- **The head reads only this turn.** Under `PRESERVE_MIXED_HISTORY`, when the kept history holds
  control tokens, a head turn's prompt is rendered from text, so no earlier adapter's tokens change
  what the head reads. The kept history is not touched.
- **Its verdict joins the transcript as text.** Recording a head turn's answer adds it as an
  assistant message, and never keeps the marker in the ids sent later.
- **One token.** `completion_payload` asks for one token for a head turn, unless the caller says
  otherwise.

## Limits

- **A marker must be the prompt's last token.** The chat template and `Conversation` always place
  it there. A prompt built by hand can do otherwise: HF raises; vLLM doesn't check, so the verdict
  is lost and the request goes on as normal text.
- **One head per request.** A head reads at its marker, the prompt's last token, so two verdicts
  on one prompt take two requests.
- **Speculative decoding and microbatching** are refused with head streams on vLLM.
