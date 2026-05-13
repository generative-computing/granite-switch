# Granite Switch — Fine-tuning, finally composable

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

| [**Browse Adapters**](https://huggingface.co/collections/ibm-granite/granite-libraries) | [Models on HF](https://huggingface.co/ibm-granite/granite-switch-4.1-8b-preview) | [Tutorials](tutorials/README.md) |

Task-specific fine-tuning delivers large accuracy gains on small models — but shipping a separate model per task is operationally painful. Granite Switch gives you the accuracy of many models with the footprint of one: compose a single checkpoint from our adapter library in minutes, then swap or upgrade individual capabilities as your needs change.

Browse the full set of ready-to-use adapters in the [Granite Libraries collection](https://huggingface.co/collections/ibm-granite/granite-libraries) on Hugging Face.

## Key Features

- **Composable** — Combine independently trained adapters into one checkpoint, whether IBM's or yours. Swap, upgrade, or customize without retraining.
- **Fast** — Built on IBM's Activated LoRA technology for efficient KV cache reuse, low latency, and [high inference throughput](https://github.com/lastras/granite-switch/tree/alora-vs-lora-race/tutorials/alora_vs_lora_race).
- **Accurate** — Task-specific adapters can match and even surpass the accuracy of significantly larger generalist models, while requiring only a fraction of the serving cost. See the [adapter catalog](https://generative-computing.github.io/granite-switch/adapter_catalog.html#hallucination-detection) for benchmark comparisons across all 12 adapters.
- **Inference-ready** — Support for Hugging Face and vLLM.

## Quick Start

### Install

```bash
python -m venv venv && source venv/bin/activate

# Granite-Switch installation is based on your usecase:
pip install "granite-switch[compose]"   # Compose modular models
pip install "granite-switch[hf]"        # HuggingFace inference
pip install "granite-switch[vllm]"      # vLLM production inference (0.19.x)
pip install "granite-switch[vllm20]"    # vLLM 0.20+ (requires CUDA 13+)
pip install "granite-switch[dev]"       # Everything (uses vLLM 0.19.x by default)
pip install "granite-switch[dev-vllm20]" # Dev environment with vLLM 0.20+
```

Requires Python 3.9+ and PyTorch 2.0+.

> **vLLM version note:** This project currently defaults to vLLM 0.19.1 due to vLLM 0.20's
> dependency on CUDA 13.0+ (via PyTorch 2.11), which is incompatible with many existing
> environments running CUDA 12.x drivers. Use `.[vllm20]` if your environment supports CUDA 13+.

### Compose a Model

Combine a base Granite model with adapters into a single deployable checkpoint:

```bash
python -m granite_switch.composer.compose_granite_switch \
  --base-model ibm-granite/granite-4.1-3b \
  --adapters ibm-granite/granitelib-core-r1.0 ibm-granite/granitelib-rag-r1.0  ibm-granite/granitelib-guardian-r1.0 \
  --output ./my-model
```

Use the **[Adapter Composer](https://generative-computing.github.io/granite-switch/adapter_catalog.html)** to browse available adapters, compare benchmarks, and generate a ready-to-run compose command.

This downloads the base model, embeds compatible LoRA adapters (with a preference towards activated LoRA), adds control tokens and a chat template, and produces a model directory that works with both HuggingFace and vLLM.

For convenience, you can find already composed Granite Switch models for the Granite 4.1 model family here:

- [ibm-granite/granite-switch-4.1-3b-preview](https://huggingface.co/ibm-granite/granite-switch-4.1-3b-preview)
- [ibm-granite/granite-switch-4.1-8b-preview](https://huggingface.co/ibm-granite/granite-switch-4.1-8b-preview)
- [ibm-granite/granite-switch-4.1-30b-preview](https://huggingface.co/ibm-granite/granite-switch-4.1-30b-preview)


### Run Inference

**vLLM + Mellea (recommended):**

```bash
pip install mellea
# Example with the 3B model 
python -m vllm.entrypoints.openai.api_server --model ibm-granite/granite-switch-4.1-3b-preview --port 8000
```

```python
from mellea.backends.openai import OpenAIBackend
from mellea.stdlib.components.intrinsic import rag
from mellea.stdlib.context import ChatContext

backend = OpenAIBackend(
    model_id="ibm-granite/granite-switch-4.1-3b-preview",
    base_url="http://localhost:8000/v1",
    api_key="unused",
)
backend.register_embedded_adapter_model("ibm-granite/granite-switch-4.1-3b-preview")

query = "I want to ask you something. what is...mmmm the the main city(capital you call it,right?) of France?"
ctx = ChatContext()

rewritten = rag.rewrite_question(query, ctx, backend)
print(f"original:  {query}")
print(f"rewritten: {rewritten}")
# => "What is the capital of France?"
```

**HuggingFace:**

```python
import granite_switch.hf  # Register HF backend

from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("ibm-granite/granite-switch-4.1-3b-preview", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-switch-4.1-3b-preview")

messages = [{"role": "user", "content": "What is the capital of France?"}]
documents = [{"doc_id": "1", "text": "Paris is the capital of France."}]

prompt = tokenizer.apply_chat_template(
    messages,
    documents=documents,
    adapter_name="answerability",  # activates the answerability adapter
    add_generation_prompt=True,
    tokenize=False,
)
outputs = model.generate(**tokenizer(prompt, return_tensors="pt").to(model.device))
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# => "answerable"
```

## How It Works

Granite Switch uses a **switch layer**—a small attention-based mechanism that reads control tokens from the input and determines which adapter's LoRA weights to apply at each position.

**What makes composition work:**

- **KV cache normalization** — each adapter sees only the base model's KV cache, never another adapter's internal state
- **No joint training required** — Adapters can be developed, tested, and published independently
- **Standard inference** — The entire model loads in vLLM with zero code changes

## Documentation

For detailed tutorials and many working examples, see the [Tutorials](tutorials/README.md) section.

## Citation

```bibtex
@software{granite_switch,
  title  = {Granite Switch: Coarse-Grained Expert Switching for LLMs},
  author = {IBM Research},
  year   = {2025},
  url    = {https://github.com/ibm-granite/granite-switch}
}
```

## IBM ❤️ Open Source AI

Granite Switch was started by IBM Research.

## License

Granite Switch has an Apache-2.0 license, as found in the [LICENSE](LICENSE) file.
