# Classifier Slots

This guide explains how to compose a Granite Switch model with a **classifier slot** — a
lightweight classification head fired by the same control token as a LoRA adapter — and how to
serve it so the model emits a label *word* (`safe`, `unsafe`, …) for a detection request.

## Overview

A Granite Switch adapter slot is normally a LoRA adapter. A **classifier slot** is an alternative
occupant of that same slot: instead of a LoRA, it holds a small linear classification head. It is
triggered by the *same* control-token machinery — one `<|name|>` token, one adapter index — but
where a LoRA would modify the hidden states, a classifier reads the clean base-model hidden state
and produces per-label logits.

At inference the verdict does not come back as a raw score vector. Each label word is resolved to a
single token id at compose time, and the model rewrites the request's final logit row so that the
winning label's token id wins the argmax. The endpoint therefore emits the literal label word — the
same shape as any other generation, so it works through the standard vLLM OpenAI server with no
custom client.

```
1. Train a head  ──>  2. Manifest (kind/labels)  ──>  3. Compose  ──>  4. Serve + detect
   (weight/bias)         (io.yaml or YAML list)      (granite-switch)     (vLLM OpenAI API)
```

Multiple classifier slots may coexist, and **each slot may have its own number of labels** (e.g. a
2-label `safe`/`unsafe` slot next to a 3-label `good`/`bad`/`neutral` slot).

## Prerequisites

- Granite base model (e.g., `ibm-granite/granite-4.1-3b`)
- A trained classifier head (see Step 1)
- `granite-switch[vllm,compose]` installed

See [PREREQUISITES.md](../PREREQUISITES.md) for detailed setup.

## Step 1: Train (or stage) a classifier head

A classifier slot is **not** a LoRA, so its directory holds a different artifact. There is no
`adapter_model.safetensors`; instead the directory must contain a **`classifier_head.safetensors`**
with exactly two tensors:

| Key | Shape | Meaning |
|-----|-------|---------|
| `weight` | `[num_labels, hidden_size]` | the linear head |
| `bias`   | `[num_labels]`             | per-label bias |

`num_labels` is *this slot's own* label count, and `hidden_size` must match the base model's hidden
size. That is the whole artifact — a single linear layer that maps a base-model hidden state to one
logit per label.

Train the head however you like (a frozen base model plus a trainable linear layer over the last
hidden state is the usual recipe). Here is a helper that writes a **synthetic** head.

```python
from pathlib import Path
import torch
from safetensors.torch import save_file

HIDDEN = 2560  # granite-4.1-3b hidden size

def make_head(name, labels):
    """Write a synthetic classifier head for `name` under the library layout.

    A real head would carry trained weights; this one is zero-weight with a bias
    that forces row 0 to win, so the emitted label is deterministic.
    """
    head_dir = Path(f"./{name}-head/{name}/granite-4.1-3b/lora")
    head_dir.mkdir(parents=True, exist_ok=True)
    weight = torch.zeros(len(labels), HIDDEN)
    bias = torch.full((len(labels),), -10.0)
    bias[0] = 10.0  # row 0 (labels[0]) always wins
    save_file({"weight": weight, "bias": bias}, str(head_dir / "classifier_head.safetensors"))
    # Every composed slot (LoRA or classifier) also needs an io.yaml — compose copies
    # it verbatim into the checkpoint's io_configs/. A two-line stub is enough here.
    (head_dir / "io.yaml").write_text(f"name: {name}\nmodel: ~\n")
    return str(head_dir)
```

> The directory layout `<name>/<name>/<target_model>/<technology>/` mirrors the library layout the
> composer expects for a local adapter. The trailing `lora`/`alora` segment selects **token
> placement** (LoRA vs activated-LoRA), independent of the fact that this slot is a classifier.
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
1 classifier label word(s) do not encode to a single token, ...
  'Ambiguous' → 3 tokens: ['Amb', 'igu', 'ous']
```

Pick a single-token synonym for each label. Short lowercase words (`safe`, `unsafe`, `yes`, `no`,
`good`, `bad`, `neutral`) are single-token in Granite BPE; capitalized or rarer words often are not.
If no natural single-token word fits, the tokenizer's reserved `<|unused_N|>` tokens are a fallback.

## Step 2: Declare the slot in a manifest

Classifier slots are declared with two extra keys — `kind: classifier` and `labels:` — on top of the
normal adapter fields. The simplest way to pass one or more of them to compose is a **YAML manifest**
that maps each slot name to its path and metadata:

```yaml
# ./adapters.yaml (one classifier slot)
safety:
  path: ./safety-head/safety/granite-4.1-3b/lora
  type: lora                 # token placement (lora vs alora); independent of `kind`
  kind: classifier           # marks this slot a classifier head (default "lora" when omitted)
  labels: [safe, unsafe]     # label words, resolved to single token ids at compose
```

`type` and `kind` are independent axes: `type` is the LoRA *technology* (drives control-token
placement), while `kind` is whether the slot is a real LoRA or a classifier head. A plain LoRA entry
simply omits `kind`/`labels`.

You can mix LoRA and classifier entries in one manifest, and give each classifier its own `labels`
list of any length:

```yaml
# ./adapters.yaml — one real LoRA + two synthetic classifier slots (2 and 3 labels)
answerability:
  path: /path/to/answerability/granite-4.1-3b/lora   # a real LoRA (see below)
  type: lora
safety:
  path: ./safety-head/safety/granite-4.1-3b/lora
  type: lora
  kind: classifier
  labels: [safe, unsafe]
faithfulness:
  path: ./faithfulness-head/faithfulness/granite-4.1-3b/lora
  type: lora
  kind: classifier
  labels: [good, bad, neutral]      # single-token words — see the label rule above
```

> **Slot ordering is automatic.** Compose stably reorders slots so all LoRA slots come first and
> classifier slots last, then derives every downstream artifact (control tokens, chat template,
> adapter indices) from that order. You do not need to order the manifest yourself; classifiers
> ending up as the contiguous suffix is what lets the LoRA weight transfer stay positional and the
> classifier head bank pad uniformly.

The script below download a rag lora and constructed two synthetic classifiers for testing.

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

# 2. Two synthetic classifier slots with different label counts (make_head from Step 1).
safety_dir = make_head("safety", ["safe", "unsafe"])
faith_dir = make_head("faithfulness", ["good", "bad", "neutral"])

# 3. Emit the manifest pairing the real LoRA with the two synthetic classifiers.
manifest = {
    "answerability": {"path": lora_dir, "type": "lora"},
    "safety": {"path": safety_dir, "type": "lora", "kind": "classifier",
               "labels": ["safe", "unsafe"]},
    "faithfulness": {"path": faith_dir, "type": "lora", "kind": "classifier",
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
2. Resolve each classifier's label words to single token ids (failing loud on multi-token words).
3. Read each classifier's `classifier_head.safetensors` into the head bank at its slot.
4. Stack everything into one checkpoint under `--output`.


```
Classifier slots:
  safety → labels ['safe', 'unsafe'] → token ids [19193, 39257]
  faithfulness → labels ['good', 'bad', 'neutral'] → token ids [19045, 14176, 60668]
```

The resulting `config.json` carries the per-slot classifier metadata
(`classifier_num_labels_per_slot`, `max_classifier_labels`, `classifier_label_names`,
`classifier_label_token_ids`, `classifier_slot_indices`) — all plain lists/ints, so it round-trips
through `save_pretrained`/`from_pretrained` with no custom code, and both the HF and vLLM backends
read it identically.

## Step 4: Serve and send a detection request

Serve the composed checkpoint with the standard vLLM OpenAI-compatible server:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model ./granite-switch-detect \
  --port 8000
```

Fire a classifier by placing its control token in the prompt — one token per slot, `<|safety|>` for
the `safety` slot, `<|faithfulness|>` for the other. The completion is the emitted label word:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "./granite-switch-detect",
    "prompt": "<|safety|>Is this text safe? The weather is nice today.",
    "max_tokens": 1
  }'
```

```json
{"choices": [{"text": "safe", ...}], ...}
```

Only the classifier request's final logit row is rewritten; every other (non-classifier) request runs
untouched with its full vocabulary. If you request logprobs, you will see the label token ids
carrying the verdict scores and the rest of the vocabulary pinned at the blanking floor — exactly the
two labels for a 2-label slot, three for a 3-label slot.

### Firing via the chat template (optional)

Instead of placing `<|safety|>` by hand, you can let the composed model's chat template place it for
you by passing `adapter_name` — a classifier slot is activated exactly like any other adapter:

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
    "prompt": "<|safety|>Is this text safe? ...",
    "max_tokens": 1,
    "logprobs": 5
  }'
```

Only the label ids (`safe`, `unsafe`) will show finite logprobs; all other tokens sit at the blanking
floor. That is the classifier head's per-label verdict surfaced through the standard logprobs
channel.

## How it works (brief)

- **Same trigger as a LoRA.** One `<|name|>` control token per slot, one adapter index. The switch
  tags that index as a classifier and routes the token out of the LoRA stream, so the LoRA path
  no-ops there and the classifier head runs on the clean base hidden state.
- **Padded head bank.** All classifier heads live in one stacked tensor padded to
  `max_classifier_labels`; a slot with fewer labels leaves its extra rows zero. This keeps the vLLM
  head a single fused, compile-safe operation regardless of how many labels each slot has — the same
  way the LoRA bank pads every adapter to `max_lora_rank`.
- **Label-word verdict exit.** The model collapses each classifier request to its final token,
  computes the head's per-label logits, blanks the vocabulary, and scatters the verdict onto that
  slot's own label token ids (sliced to the slot's real label count, so padded columns are never
  emitted). The sampler then emits the label word.

## Related

- [Bring Your Own Adapter](build_your_own_adapter.md) — train and compose a LoRA adapter.
- [Composer README](../../src/granite_switch/composer/README.md) — full compose CLI reference.
