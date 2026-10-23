# Activation Probe Slots

This guide explains how to compose a Granite Switch model with an **activation probe slot** — a
lightweight linear head over a hidden state, fired by the same control token as a LoRA adapter —
and how to serve it so the model emits a label *word* (`safe`, `unsafe`, …) for a detection
request.

## Overview

An **activation probe slot** (probe for short) holds a small linear head instead of a LoRA. It
uses the same control-token machinery (one `<|name|>` token, one adapter index) and reads the
configured layer's activation to produce per-label logits. Probe and generation-adapter slots can
coexist in one checkpoint; a request selects one slot.

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
with that marker appended, and read the hidden state there.

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
> composer expects for a local adapter. The trailing `lora`/`alora` segment selects **token
> placement** (LoRA vs activated-LoRA), independent of the fact that this slot is a probe.
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
1 probe label word(s) do not encode to a single token, ...
  'Ambiguous' → 3 tokens: ['Amb', 'igu', 'ous']
```

Pick a single-token synonym for each label. Short lowercase words (`safe`, `unsafe`, `yes`, `no`,
`good`, `bad`, `neutral`) are single-token in Granite BPE; capitalized or rarer words often are not.
If no natural single-token word fits, the tokenizer's reserved `<|unused_N|>` tokens are a fallback.

## Step 2: Declare the slot in a manifest

Probe slots are declared with two extra keys — `kind: probe` and `labels:` — on top of the
normal adapter fields. The simplest way to pass one or more of them to compose is a **YAML manifest**
that maps each slot name to its path and metadata:

```yaml
# ./adapters.yaml (one probe slot)
safety:
  path: ./safety-head/safety/granite-4.1-3b/lora
  type: lora                 # token placement (lora vs alora); independent of `kind`
  kind: probe                # marks this slot a probe head (default "lora" when omitted)
  labels: [safe, unsafe]     # label words, resolved to single token ids at compose
```

`type` and `kind` are independent axes: `type` is the LoRA *technology* (drives control-token
placement), while `kind` is whether the slot is a real LoRA or a probe head. A plain LoRA entry
simply omits `kind`/`labels`.

A probe entry may also set an optional `layer`, the zero-based decoder layer whose output the
head reads (the residual stream after that layer, before the final norm). Omit it to read the final
post-norm hidden state:

```yaml
safety_mid:
  path: ./safety-mid-head/safety_mid/granite-4.1-3b/lora
  type: lora
  kind: probe
  labels: [safe, unsafe]
  layer: 20                  # 0..num_decoder_layers - 1; omit for final
```

Probe slots can also be composed alongside Shadow Residual (SR) adapters; a probe
request reads that layer's base stream. Decoder layer numbers exclude the switch's KV-cache slots.
Pipeline parallelism preserves the same numbering across stages.

You can mix LoRA and probe entries in one manifest, and give each probe its own `labels`
list of any length:

```yaml
# ./adapters.yaml — one real LoRA + two synthetic probe slots (2 and 3 labels)
answerability:
  path: /path/to/answerability/granite-4.1-3b/lora   # a real LoRA (see below)
  type: lora
safety:
  path: ./safety-head/safety/granite-4.1-3b/lora
  type: lora
  kind: probe
  labels: [safe, unsafe]
faithfulness:
  path: ./faithfulness-head/faithfulness/granite-4.1-3b/lora
  type: lora
  kind: probe
  labels: [good, bad, neutral]      # single-token words — see the label rule above
```

> **Slot ordering is automatic.** Compose stably reorders slots so all LoRA slots come first and
> probe slots last, then derives every downstream artifact (control tokens, chat template,
> adapter indices) from that order. You do not need to order the manifest yourself; probes
> ending up as the contiguous suffix is what lets the LoRA weight transfer stay positional and the
> probe head bank pad uniformly.

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
    "safety": {"path": safety_dir, "type": "lora", "kind": "probe",
               "labels": ["safe", "unsafe"]},
    "faithfulness": {"path": faith_dir, "type": "lora", "kind": "probe",
                     "labels": ["good", "bad", "neutral"]},
}
Path("./adapters.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
print("wrote ./adapters.yaml")
```


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
3. Read each probe's `probe_head.safetensors` into the head bank at its slot.
4. Stack everything into one checkpoint under `--output`.


```
Probe slots:
  safety → labels ['safe', 'unsafe'] → token ids [19193, 39257] @ final norm
  faithfulness → labels ['good', 'bad', 'neutral'] → token ids [19045, 14176, 60668] @ final norm
```

A slot with `layer: 20` prints `@ layer 20` instead.

The resulting `config.json` carries the per-slot probe metadata as plain lists/ints, so it
round-trips through `save_pretrained`/`from_pretrained` with no custom code, and both the HF and
vLLM backends read it identically. Three fields are the stored contract:

| Field | Meaning |
|-------|---------|
| `adapter_kinds` | per-slot `"lora"` / `"probe"`, one entry per adapter — what marks a slot a probe |
| `probe_label_token_ids` | per-slot list of label token ids, `null` on LoRA slots |
| `probe_read_layers` | per-slot zero-based decoder layer a probe reads; `null` for the final post-norm state and on LoRA slots |

The rest are **derived** from those in `GraniteSwitchConfig.__init__`, so they appear on the config
object but are not independent inputs:

| Derived field | From |
|---------------|------|
| `probe_num_labels_per_slot` | `len()` of each slot's label token ids (`0` on LoRA slots) |
| `max_probe_labels` | `max()` of the above — the padded width the head bank is built with |
| `probe_control_token_ids` | the control-token ids of the probe slots; both backends match these in `input_ids` to locate a verdict's read point |

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
emitted label word:

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
template appends the control token after the last turn:

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
  out of the LoRA stream and the head reads the activation there.
- **Padded head bank.** All heads share one tensor padded to `max_probe_labels` (like the LoRA
  bank pads to `max_lora_rank`), keeping the vLLM head one compile-safe operation.
- **Scoring.** Intermediate-layer slots are scored as that layer runs, final-layer slots after the
  final norm. Scores and slot id travel with the hidden states to the logits step, so this also
  works under CUDA-graph capture. Under pipeline parallelism the first stage finds the markers and
  each stage passes scores and marker indices on.
- **Verdict exit.** The marker's logit row is blanked and the head's scores are written onto that
  slot's label token ids, so the sampler emits the label word.

## Related

- [Bring Your Own Adapter](build_your_own_adapter.md) — train and compose a LoRA adapter.
- [Composer README](../../src/granite_switch/composer/README.md) — full compose CLI reference.
