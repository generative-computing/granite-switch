# Activation Probe Slots

This guide explains how to compose a Granite Switch model with an **activation probe slot** — a
lightweight linear head over a hidden state, fired by the same control token as a LoRA adapter —
and how to serve it so the model emits a label *word* (`safe`, `unsafe`, …) for a detection
request.

## Overview

An **activation probe slot** (probe for short) holds a small linear head instead of a LoRA. It
uses the same control-token machinery (one `<|name|>` token, one adapter index) and reads the
configured layer's activation to produce per-label logits. Probe and generation-adapter slots can
coexist in one checkpoint; a request selects one slot. A probe reads the base model's state, or,
with `adapter:`, the state one named adapter computes ([A probe on an adapter](#a-probe-on-an-adapter)).

The probe is the first **head stream** type of Granite Switch's stream API. Other head types plug
in the same way, with no change to Granite Switch; see [The Stream API](../../docs/STREAM_API.md).

At inference the verdict does not come back as a raw score vector. Each label word is resolved to a
single token id at compose time, and the model rewrites the request's final logit row so that the
winning label's token id wins the argmax. The endpoint therefore emits the literal label word — the
same shape as any other generation, so it works through the standard vLLM OpenAI server with no
custom client.

```
1. Train a head  ──>  2. Manifest (kind/labels)  ──>  3. Compose  ──>  4. Serve + detect
   (weight/bias)         (io.yaml or YAML list)      (granite-switch)     (vLLM OpenAI API)
```

Multiple probe slots may coexist, and **each slot may have its own number of labels** (e.g. a
2-label `safe`/`unsafe` slot next to a 3-label `good`/`bad`/`neutral` slot).

## Prerequisites

- Granite base model (e.g., `ibm-granite/granite-4.1-3b`)
- A trained probe head (see Step 1)
- `granite-switch[vllm,compose]` installed

See [PREREQUISITES.md](../PREREQUISITES.md) for detailed setup.

## Step 1: Train (or stage) a probe head

A probe slot is **not** a LoRA, so its directory holds a different artifact. There is no
`adapter_model.safetensors`; instead the directory must contain a **`probe_head.safetensors`**
with exactly two tensors:

| Key | Shape | Meaning |
|-----|-------|---------|
| `weight` | `[num_labels, hidden_size]` | the linear head |
| `bias`   | `[num_labels]`             | per-label bias |

`num_labels` is *this slot's own* label count, and `hidden_size` must match the base model's hidden
size. That is the whole artifact — a single linear layer that maps a base-model hidden state to one
logit per label.

Train the head however you like (a frozen base model plus a trainable linear layer over the last
hidden state is the usual recipe). At serving time the head reads the hidden state of the control
token, which the model embeds as the base's role-open marker: `<|start_of_role|>` on Granite
4.0/4.1, `<|im_start|>` on 4.2. So train on the chat-rendered prompt with no generation prompt,
with that marker appended, and read the hidden state there. A probe on an adapter reads the state
that adapter computes, so train it on that: the prompt the chat template renders for the pair (see
[A probe on an adapter](#a-probe-on-an-adapter)), with the adapter loaded.

By default that is the **final post-norm** hidden state (HF's `last_hidden_state`). A slot can
instead read an intermediate decoder layer (see `layer` in Step 2); train it on that layer's output.
With HF's `output_hidden_states=True` that is `hidden_states[layer + 1]`, except for the last
decoder layer, whose entry HF replaces with the normalized output; read that one with a forward hook
on `model.model.layers[layer]`. The head must be trained on the same representation it will read at
serving time.

Here is a helper that writes a **synthetic** head.

```python
from pathlib import Path
import torch
from safetensors.torch import save_file

HIDDEN = 2560  # granite-4.1-3b hidden size

def make_head(name, labels):
    """Write a synthetic probe head for `name` under the library layout.

    A real head would carry trained weights; this one is zero-weight with a bias
    that forces row 0 to win, so the emitted label is deterministic.
    """
    head_dir = Path(f"./{name}-head/{name}/granite-4.1-3b/lora")
    head_dir.mkdir(parents=True, exist_ok=True)
    weight = torch.zeros(len(labels), HIDDEN)
    bias = torch.full((len(labels),), -10.0)
    bias[0] = 10.0  # row 0 (labels[0]) always wins
    save_file({"weight": weight, "bias": bias}, str(head_dir / "probe_head.safetensors"))
    # Every composed slot (LoRA or probe) also needs an io.yaml — compose copies
    # it verbatim into the checkpoint's io_configs/. A two-line stub is enough here.
    (head_dir / "io.yaml").write_text(f"name: {name}\nmodel: ~\n")
    return str(head_dir)
```

> The directory layout `<name>/<name>/<target_model>/<technology>/` mirrors the library layout the
> composer expects for a local adapter. For a probe the trailing segment is not used: a probe's
> control token always goes last.
>
> The `io.yaml` is required for **every** slot: compose copies each slot's `io.yaml` into the
> output's `io_configs/<name>/`, so a slot without one fails the copy step. Real adapters ship a
> full `io.yaml` (instruction, response schema, sampling parameters); a synthetic slot only needs
> the two-line stub above.

### Label words must be single-token

Each label word is resolved to a **single token id** at compose time — the verdict is written at one
token id per label. If a label word does not encode to a single token in the base model's tokenizer,
compose fails loud and lists the offenders, e.g.:

```
1 label word(s) do not encode to a single token, ...
  'Ambiguous' → 3 tokens: ['Amb', 'igu', 'ous']
```

Pick a single-token synonym for each label. Short lowercase words (`safe`, `unsafe`, `yes`, `no`,
`good`, `bad`, `neutral`) are single-token in Granite BPE; capitalized or rarer words often are not.
If no natural single-token word fits, the tokenizer's reserved `<|unused_N|>` tokens are a fallback.

## Step 2: Declare the slot in a manifest

Probe slots are declared with two extra keys — `kind: probe` and `labels:` — next to the slot's
path. The simplest way to pass one or more of them to compose is a **YAML manifest** that maps each
slot name to its path and metadata:

```yaml
# ./adapters.yaml (one probe slot)
safety:
  path: ./safety-head/safety/granite-4.1-3b/lora
  kind: probe                # the stream type: a probe head (default "lora" when omitted)
  labels: [safe, unsafe]     # label words, resolved to single token ids at compose
```

`kind` is the slot's stream type. An adapter entry's `type` (its LoRA technology, `lora` or
`alora`) decides where its control token goes; a probe's control token always goes last, so a probe
entry needs no `type`. A plain adapter entry simply omits `kind`/`labels`.

A probe entry may also set an optional `layer`, the zero-based decoder layer whose output the
head reads (the residual stream after that layer, before the final norm). Omit it to read the final
post-norm hidden state:

```yaml
safety_mid:
  path: ./safety-mid-head/safety_mid/granite-4.1-3b/lora
  kind: probe
  labels: [safe, unsafe]
  layer: 20                  # 0..num_decoder_layers - 1; omit for final
```

A layer the model doesn't have is refused when the model is built (at compose, too). Decoder layer
numbers exclude the switch's KV-cache slots. Pipeline parallelism preserves the same numbering
across stages.

You can mix LoRA and probe entries in one manifest, and give each probe its own `labels`
list of any length:

```yaml
# ./adapters.yaml — one real LoRA + two synthetic probe slots (2 and 3 labels)
answerability:
  path: /path/to/answerability/granite-4.1-3b/lora   # a real LoRA (see below)
  type: lora
safety:
  path: ./safety-head/safety/granite-4.1-3b/lora
  kind: probe
  labels: [safe, unsafe]
faithfulness:
  path: ./faithfulness-head/faithfulness/granite-4.1-3b/lora
  kind: probe
  labels: [good, bad, neutral]      # single-token words — see the label rule above
```

> **Slot order.** Compose lists adapter slots first and heads after them, then derives every
> downstream artifact (control tokens, chat template, adapter indices) from that order. You do not
> need to order the manifest yourself. Heads take no LoRA rows, and the probe head bank holds one
> row per probe slot, padded to the widest slot's label count.

The script below download a rag lora and constructed two synthetic probes for testing.

```python
from pathlib import Path
import yaml
from huggingface_hub import snapshot_download
# make_head is the helper defined in Step 1.

# 1. A real LoRA slot: download just the answerability adapter for granite-4.1-3b.
snap = snapshot_download(
    repo_id="ibm-granite/granite-lib-rag-r1.0",
    allow_patterns=["answerability/granite-4.1-3b/lora/**"],
)
lora_dir = str(Path(snap) / "answerability" / "granite-4.1-3b" / "lora")

# 2. Two synthetic probe slots with different label counts (make_head from Step 1).
safety_dir = make_head("safety", ["safe", "unsafe"])
faith_dir = make_head("faithfulness", ["good", "bad", "neutral"])

# 3. Emit the manifest pairing the real LoRA with the two synthetic probes.
manifest = {
    "answerability": {"path": lora_dir, "type": "lora"},
    "safety": {"path": safety_dir, "kind": "probe", "labels": ["safe", "unsafe"]},
    "faithfulness": {"path": faith_dir, "kind": "probe",
                     "labels": ["good", "bad", "neutral"]},
}
Path("./adapters.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
print("wrote ./adapters.yaml")
```

### A probe on an adapter

A probe entry may name an adapter slot of the same checkpoint. The probe then reads the state that
adapter computes, not the base model's:

```yaml
answerability:
  path: /path/to/answerability/granite-4.1-3b/lora
  type: lora
safety_rag:                  # the safety probe, on the answerability adapter
  path: ./safety-rag-head/safety_rag/granite-4.1-3b/lora
  kind: probe
  labels: [safe, unsafe]
  adapter: answerability
```

Each pair of a probe and an adapter is its own slot, with its own control token, fixed at compose
time; a request still names one slot (`adapter_name="safety_rag"`). The chat template places what
that adapter needs for the probe to read exactly the state it computes alone:

| The adapter | What the template places |
|---|---|
| Shadow Residual | The probe's control token alone (the adapter writes no K/V) |
| LoRA | The adapter's own token where it goes alone (the start), then the probe's token last |
| aLoRA invoked inside the user message | The adapter's own token there, then the probe's token last |
| aLoRA invoked at the assistant header | The probe's token alone: the adapter would start right there |

Each pair costs a slot, a control token and a bank row, even when the same head is reused on
several adapters.

## Step 3: Compose

Pass the manifest to the compose CLI exactly like any other adapter source — a `.yaml`/`.yml` path is
auto-detected as a manifest:

```bash
python -m granite_switch.composer.compose_granite_switch \
  --base-model ibm-granite/granite-4.1-3b \
  --adapters ./adapters.yaml \
  --output ./granite-switch-detect
```

Compose will:

1. Load the base model and discover the slots from the manifest.
2. Resolve each probe's label words to single token ids (failing loud on multi-token words).
3. Give each probe the next row of the probe bank, and read its `probe_head.safetensors` into it.
4. Stack everything into one checkpoint under `--output`.


```
Head streams:
  safety (probe) → select over token ids [19193, 39257] @ final norm
  faithfulness (probe) → select over token ids [19045, 14176, 60668] @ final norm
```

A slot with `layer: 20` prints `@ layer 20` instead, and a probe on an adapter prints
`(probe on answerability)`.

The resulting `config.json` holds one **stream entry** per slot, under `streams`, as plain
lists/ints, so it round-trips through `save_pretrained`/`from_pretrained` with no custom code, and
both the HF and vLLM backends read it identically. A probe's entry:

```json
{"type": "probe", "read_layer": null, "head_row": 0,
 "output": {"mode": "select", "token_ids": [19193, 39257]}}
```

| Field | Meaning |
|-------|---------|
| `type` | `"probe"`; an adapter slot's entry is just `{"type": "lora"}` |
| `read_layer` | the zero-based decoder layer the probe reads; `null` for the final post-norm state |
| `head_row` | the slot's row in the probe bank |
| `adapter` | only on a probe on an adapter: that adapter's slot name |
| `output.token_ids` | the label token ids; `output.mode` is `select` for a probe |

`config.json` also keeps the older probe fields (`adapter_kinds`, `probe_label_token_ids`,
`probe_read_layers`), derived from the entries, so older versions of Granite Switch can read it. A
checkpoint composed before stream entries has only those fields and still loads: its entries are
rebuilt from them. See [The Stream API](../../docs/STREAM_API.md) for every field.

Note that the label *words* are not stored — only their token ids. Compose resolves and prints the
words, but `config.json` keeps ids, so recovering the words means decoding them with the
checkpoint's tokenizer.

## Step 4: Serve and send a detection request

Serve the composed checkpoint with the standard vLLM OpenAI-compatible server:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model ./granite-switch-detect \
  --port 8000
```

Fire a probe by placing its control token at the very end of the prompt, after the last turn's
close — one token per slot, `<|safety|>` for the `safety` slot, `<|faithfulness|>` for the other. The
verdict is read at the control token, so leave out the generation prompt. The completion is the
emitted label word. (A probe on a LoRA or aLoRA adapter also needs that adapter's own token in the
prompt; let the chat template place both, below.)

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "./granite-switch-detect",
    "prompt": "<|start_of_role|>user<|end_of_role|>Is this text safe? The weather is nice today.<|end_of_text|>\n<|safety|>",
    "max_tokens": 1
  }'
```

```json
{"choices": [{"text": "safe", ...}], ...}
```

Only the probe request's final logit row is rewritten; every other (non-probe) request runs
untouched with its full vocabulary. If you request logprobs, you will see the label token ids
carrying the verdict scores and the rest of the vocabulary pinned at the blanking floor — exactly the
two labels for a 2-label slot, three for a 3-label slot.

### Firing via the chat template (optional)

Instead of placing `<|safety|>` by hand, you can let the composed model's chat template place it for
you by passing `adapter_name` — a probe slot is activated exactly like any other adapter. The
template appends the control token after the last turn, and for a probe on an adapter, also places
that adapter's token:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "./granite-switch-detect",
    "messages": [{"role": "user", "content": "Is this text safe? The weather is nice today."}],
    "max_tokens": 1,
    "chat_template_kwargs": {"adapter_name": "safety"}
  }'
```

### Reading logprobs (optional)

To see the raw verdict rather than just the winning word, ask for a couple of logprobs:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "./granite-switch-detect",
    "prompt": "<|start_of_role|>user<|end_of_role|>Is this text safe? ...<|end_of_text|>\n<|safety|>",
    "max_tokens": 1,
    "logprobs": 5
  }'
```

Only the label ids (`safe`, `unsafe`) will show finite logprobs; all other tokens sit at the blanking
floor. That is the probe head's per-label verdict surfaced through the standard logprobs
channel.

## How it works (brief)

- **Routing.** A probe control token selects its slot like any adapter; its position is routed
  out of the LoRA stream (or, for a probe on an adapter, through that adapter's LoRA) and the head
  reads the activation there.
- **Padded head bank.** All probes share one tensor, one row per probe slot, padded to the widest
  slot's label count, keeping the vLLM head one compile-safe matmul.
- **Scoring.** Intermediate-layer slots are scored as that layer runs, final-layer slots after the
  final norm. Scores and slot id travel with the hidden states to the logits step, so this also
  works under CUDA-graph capture. Under pipeline parallelism the first stage finds the markers and
  each stage passes scores and marker indices on.
- **Verdict exit.** The marker's logit row is blanked and the head's scores are written onto that
  slot's label token ids, so the sampler emits the label word.

## Related

- [The Stream API](../../docs/STREAM_API.md) — stream entries, heads on adapters, and how to write
  a new head type.
- [Bring Your Own Adapter](build_your_own_adapter.md) — train and compose a LoRA adapter.
- [Composer README](../../src/granite_switch/composer/README.md) — full compose CLI reference.
