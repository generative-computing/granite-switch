# Bring Your Own Adapter (BYOA)

This guide explains how to train your own adapter (aLoRA or LoRA) and compose it into a Granite Switch model.

## Overview

Granite Switch supports embedding custom adapters alongside (or instead of) the official IBM Granite adapter libraries. This enables:

- Custom task-specific adapters
- Domain-specific suite of adapters
- Research and experimentation

## Prerequisites

- Granite base model (e.g., `ibm-granite/granite-4.1-3b`)
- Training data for your task(s)
- `granite-switch[hf,compose]` installed

See [PREREQUISITES.md](../PREREQUISITES.md) for detailed setup.

## Workflow Overview

```
1. Train (a)LoRA  ──>  2. Compose  ──>  3. Use Model
     (HF PEFT)        (granite-switch)    (HF or vLLM)
```

## Step 1: Train Your (a)LoRA Adapter

Today, IBM Granite Switch models come with IBM-trained adapters from the [Granite Libraries](https://huggingface.co/collections/ibm-granite/granite-libraries). However, Granite Switch allows you to compose your own modular model with custom adapters.

For training your own adapter, use the HuggingFace PEFT library:

- **[LoRA Training Script](https://huggingface.co/docs/diffusers/training/lora)** — Reference implementation for training LoRA adapters
- **[aLoRA Training Script](https://github.com/huggingface/peft/tree/main/examples/alora_finetuning)** — Reference implementation for training Activated LoRA adapters, for technical details on the aLoRA architecture see [Activated LoRA Paper](https://arxiv.org/abs/2504.12397)

Save the trained adapter under the library-style layout the composer expects for a local adapter — `<adapter_name>/<target_model>/<technology>/` — for example, `./uncertainty-adapter/uncertainty/granite-4.1-3b/alora/` for an aLoRA adapter, or `.../lora/` for a LoRA adapter.

### Adapter Compatibility Requirements

When composing multiple adapters together, all adapters must be trained on the same base model.
Composed model size is determined by the base model size and the maximal rank of the adapters.

Rank (`r`) and scaling (`lora_alpha`) may vary across adapters: the composer zero-pads smaller-rank adapters up to the build's max rank, and stores per-adapter scaling in the composed config. The IBM Granite adapter libraries ship a mix of ranks (typically `r=16` or `r=32`).

## Step 2: Create Adapter Configuration

The compose system auto-discovers adapters using an `io.yaml` file alongside the adapter weights. At minimum, `io.yaml` declares the adapter's name:

```yaml
# ./uncertainty-adapter/uncertainty/granite-4.1-3b/alora/io.yaml
name: uncertainty
```

The adapter's technology (aLoRA — activated LoRA — or LoRA) is **not** a field in `io.yaml`; it is inferred from the directory name (`alora/` or `lora/`) at the bottom of the library-style layout.

Production adapters typically carry additional fields consumed by higher-level runtimes such as Mellea: `response_format` (a JSON schema for structured output), `transformations` (post-processing rules), `instruction`, `parameters` (e.g., `max_completion_tokens`, `temperature`), and `sentence_boundaries`. See Step 3's running example for a complete, production-grade `io.yaml`.

## Step 3: Compose into Granite Switch

Use the compose CLI to embed your adapter into a Granite Switch checkpoint, on its own or alongside official adapter libraries. The compose process:

1. Loads the base Granite model
2. Discovers adapters from each source (using `io.yaml`)
3. Validates compatibility (rank, alpha)
4. Stacks adapter weights into a single checkpoint
5. Generates control tokens and chat template
6. Produces a compose report (`compose_report.json`) and a model card (`BUILD.md`)

For the full CLI reference, adapter source options, filtering, and more examples, see the [Composer README](../../src/granite_switch/composer/README.md).

### Running example

In practice you will train your own adapter with Step 1's recipe. To make the rest of this guide runnable end to end, we stand in for a locally-trained adapter by staging an existing IBM adapter — `uncertainty` from `ibm-granite/granitelib-core-r1.0` — in the same library-style layout Step 1 writes to, then compose it alongside the full `ibm-granite/granitelib-rag-r1.0` library. The composed model ends up with 7 adapters: 6 from the RAG library plus `uncertainty`.

**Stage the adapter as if you had just trained it:**

```python
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download

# Download just the uncertainty/granite-4.1-3b/alora subtree from core.
core_lib = snapshot_download(
    "ibm-granite/granitelib-core-r1.0",
    allow_patterns=["uncertainty/granite-4.1-3b/alora/*"],
)

# Stage under a library-style layout (<adapter_name>/<target_model>/<technology>/).
# This is what the composer expects when you point --adapters at a locally-trained
# adapter, and it is the layout a real PEFT training run would produce (plus an
# io.yaml you author by hand).
local_adapter = Path("./uncertainty-adapter/uncertainty/granite-4.1-3b/alora")
if local_adapter.exists():
    shutil.rmtree("./uncertainty-adapter")
shutil.copytree(
    f"{core_lib}/uncertainty/granite-4.1-3b/alora",
    local_adapter,
)

# The IBM-shipped adapter includes a Sigstore signature (model.sig) that
# authenticates the upstream weights. A locally-trained adapter would not
# have one, so drop it to match the layout PEFT actually produces.
(local_adapter / "model.sig").unlink(missing_ok=True)

print(sorted(p.name for p in local_adapter.iterdir()))
# ['adapter_config.json', 'adapter_model.safetensors', 'io.yaml']
```

**Inspect the `io.yaml` that ships with the adapter** — this is what you would write by hand for an adapter you actually trained. The schema is richer than the minimal example in Step 2; `response_format` declares the shape of the adapter's output, and the optional `transformations` and `parameters` fields are consumed by higher-level runtimes such as Mellea:

```yaml
name: uncertainty
# Model name string, or null to use whatever is provided in the chat completion request
model: ~
# JSON schema of the model's output
response_format: |
  {
    "type": "object",
    "properties": {
      "score": {
        "type": "string",
        "enum": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
      }
    },
    "required": ["score"],
    "additionalProperties": false
  }
# Output transformation rules to apply
transformations:
  - type: likelihood
    categories_to_values:
      "0": 0.05
      "1": 0.15
      "2": 0.25
      "3": 0.35
      "4": 0.45
      "5": 0.55
      "6": 0.65
      "7": 0.75
      "8": 0.85
      "9": 0.95
    input_path: ["score"]
  - type: project
    input_path: []
    retained_fields:
      score: "certainty"
instruction: ~
parameters:
  max_completion_tokens: 15
  temperature: 0.0
sentence_boundaries: ~
```

**Run the composer:**

```bash
python -m granite_switch.composer.compose_granite_switch \
    --base-model ibm-granite/granite-4.1-3b \
    --adapters \
        ibm-granite/granitelib-rag-r1.0 \
        ./uncertainty-adapter/uncertainty/granite-4.1-3b/alora \
    --output ./composed-model
```

The first `--adapters` entry is the RAG library — the composer auto-discovers all 6 adapters inside. The second entry is the local path to the "built-by-us" adapter; the composer walks up from `alora/` and uses the grandparent-directory name (`uncertainty`) as the adapter name.

### Compose output

```
./composed-model/
├── model-00001-of-00002.safetensors   # Full model with embedded adapters
├── model-00002-of-00002.safetensors
├── model.safetensors.index.json
├── config.json                        # Model configuration (includes adapter_names)
├── tokenizer.json                     # Tokenizer with control tokens
├── tokenizer_config.json
├── special_tokens_map.json
├── chat_template.jinja
├── adapter_index.json                 # Adapter name, control token, io.yaml pointer
├── compose_report.json                # Detailed compose report
├── BUILD.md                           # Compose-specific model card
└── io_configs/                        # Original io.yaml files, one per adapter
    ├── answerability/io.yaml
    ├── citations/io.yaml
    ├── context_relevance/io.yaml
    ├── hallucination_detection/io.yaml
    ├── query_clarification/io.yaml
    ├── query_rewrite/io.yaml
    └── uncertainty/io.yaml
```

The base model's tokenizer and generation assets (`generation_config.json`, `merges.txt`, `vocab.json`) are copied through unchanged, so the composed directory is self-contained and deployment-ready.

## Step 4: Use the Composed Model

> **Note:** Custom (BYOA) adapters are not supported by [Mellea](https://github.com/generative-computing/mellea). Mellea only supports the official IBM Granite Library adapters. To invoke your custom adapters, use the chat template directly as shown below.

### With HuggingFace

The composed model ships with a chat template that places the adapter's control token correctly for its technology (aLoRA vs LoRA). Invoke any adapter by passing `adapter_name=` to `apply_chat_template`:

```python
import granite_switch.hf  # Register HF backend

from transformers import AutoModelForCausalLM, AutoTokenizer

# Load composed model
model = AutoModelForCausalLM.from_pretrained("./composed-model")
tokenizer = AutoTokenizer.from_pretrained("./composed-model")
model.eval()
model.to("cuda")

# Activate your adapter via the chat template
messages = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris is the capital of France."},
]

prompt = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    adapter_name="uncertainty",  # Your adapter name from io.yaml
    tokenize=False,
)

outputs = model.generate(**tokenizer(prompt, return_tensors="pt").to("cuda"), max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### With vLLM

For production deployment:

```bash
# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model ./composed-model \
    --port 8000
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

response = client.chat.completions.create(
    model="./composed-model",
    messages=[
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris is the capital of France."},
    ],
    extra_body={
        "chat_template_kwargs": {"adapter_name": "uncertainty"},
    },
    max_completion_tokens=100,
)
print(response.choices[0].message.content)
```

## Next Steps

- **[Hello Mellea](../quickstart/hello_mellea.ipynb)** - run your composed checkpoint through Mellea's intrinsic wrappers
- **[Compose Granite Switch](../notebooks/03_compose_granite_switch.ipynb)** - compose a full model from the IBM adapter libraries
- **[Government RAG Pipeline](../notebooks/02_govt_rag_pipeline.ipynb)** - wire your adapter into an end-to-end RAG loop
- **[Using Mellea with Granite Switch](mellea_with_granite_switch.md)** - deeper Mellea integration details
