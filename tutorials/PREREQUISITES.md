# Prerequisites

Setup requirements for running Granite Switch tutorials.

## Hardware Requirements

| Tutorial Type | GPU Required |
|---------------|--------------|
| Quickstart (micro model) | Yes (CUDA) |
| HuggingFace Scripts (3B) | Yes (CUDA) |
| vLLM/Mellea (8B) | Yes (CUDA) |
| Large Models (30B) | Yes (CUDA) |

## Software Requirements

### Python Version

Python 3.10+ is required.

### Base Installation

```bash
pip install granite-switch
```

### HuggingFace Backend

For direct model inference with HuggingFace Transformers:

```bash
pip install "granite-switch[hf,compose]"
```

This includes:
- `transformers` for model loading and generation
- `torch` with CUDA support
- `peft` for LoRA operations
- Compose tools for model building

### vLLM Backend

For production inference with vLLM:

```bash
pip install "granite-switch[vllm]"
```

This includes:
- `vllm>=0.19.1` for high-performance inference
- Tensor parallelism support for multi-GPU setups

### Mellea Integration

Mellea provides high-level intrinsic functions for adapter invocation:

```bash
pip install mellea
```

### Notebook Dependencies

For running Jupyter notebooks:

```bash
pip install jupyter chromadb tqdm httpx python-dotenv
```

## Model Access

### Base Models

Available on [HuggingFace Hub](https://huggingface.co/ibm-granite):

| Model | Size | Use Case |
|-------|------|----------|
| `ibm-granite/granite-4.0-micro` | 3B | Quick demos, testing |
| `ibm-granite/granite-4.1-3b` | 3B | Development, single GPU |
| `ibm-granite/granite-4.1-8b` | 8B | Production, single GPU |
| `ibm-granite/granite-4.1-30b` | 30B | Production, multi-GPU |

### Adapter Libraries

Official IBM Granite adapter libraries (r1.0):

| Library | Adapters | Purpose |
|---------|----------|---------|
| [ibm-granite/granitelib-rag-r1.0](https://huggingface.co/ibm-granite/granitelib-rag-r1.0) | 5 | RAG intrinsics (rewrite, answerability, citations, etc.) |
| [ibm-granite/granitelib-core-r1.0](https://huggingface.co/ibm-granite/granitelib-core-r1.0) | 3 | Core intrinsics (certainty, requirements, attributions) |
| [ibm-granite/granitelib-guardian-r1.0](https://huggingface.co/ibm-granite/granitelib-guardian-r1.0) | 4 | Guardian intrinsics (harm check, policy, factuality, etc.) |

### HuggingFace Authentication

For accessing gated models:

```bash
# Interactive login
huggingface-cli login

# Or set token directly
export HF_TOKEN=your_token_here
```

## Starting a vLLM Server

For tutorials using Mellea or the vLLM backend:

```bash
# Single GPU
python -m vllm.entrypoints.openai.api_server \
    --model <path-or-hf-repo> \
    --port 8000 \
    --host 0.0.0.0

# Multi-GPU (tensor parallelism)
python -m vllm.entrypoints.openai.api_server \
    --model <path-or-hf-repo> \
    --port 8000 \
    --host 0.0.0.0 \
    --tensor-parallel-size 2
```

Verify the server is running:

```bash
curl http://localhost:8000/v1/models
```

## External Resources

| Resource | URL | Description |
|----------|-----|-------------|
| Mellea | https://github.com/generative-computing/mellea | Intrinsics framework for adapter invocation |
| Granite Models | https://huggingface.co/ibm-granite | Official IBM Granite models |
| Granite Libraries | https://huggingface.co/collections/ibm-granite/granite-libraries | Adapter libraries collection |
| vLLM Docs | https://docs.vllm.ai/ | vLLM documentation |

## Next Steps

Once prerequisites are installed, proceed to:

1. **[Hello Adapter](quickstart/hello_adapter.ipynb)** - Minimal HuggingFace example
2. **[Hello Mellea](quickstart/hello_mellea.ipynb)** - Mellea intrinsics with vLLM
3. **[Learning Paths](README.md#learning-paths)** - Choose your path based on use case
