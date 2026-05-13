# Granite Switch Tutorials

## Notebooks

Interactive Jupyter tutorials in [`notebooks/`](notebooks/):

| Notebook | Topics | Duration |
|----------|--------|----------|
| [00_hello_adapter.ipynb](notebooks/00_hello_adapter.ipynb) | Minimal adapter invocation with HuggingFace | 5 min |
| [01_hello_mellea.ipynb](notebooks/01_hello_mellea.ipynb) | Mellea intrinsics intro with vLLM | 5 min |
| [02_granite_switch_with_hf.ipynb](notebooks/02_granite_switch_with_hf.ipynb) | Compose + HuggingFace backend, `adapter_name=` invocation, Core + Guardian adapters in a multi-turn conversation | 20 min |
| [03_govt_rag_pipeline.ipynb](notebooks/03_govt_rag_pipeline.ipynb) | Full RAG pipeline, ChromaDB, Guardian | 30 min |
| [04_compose_granite_switch.ipynb](notebooks/04_compose_granite_switch.ipynb) | Compose a checkpoint from adapter libraries | 15 min |

## How-To Guides

| Guide | Description |
|-------|-------------|
| [Using Mellea with Granite Switch](how-to/mellea_with_granite_switch.md) | Connect Mellea to a Granite Switch model |
| [Bring Your Own Adapter](how-to/bring_your_own_adapter.md) | Train, compose, and use custom adapters |

## Learning Paths

### Path 1: Low-Level Understanding (HuggingFace)

Best for: Understanding how Granite Switch works at the control-token level

The HuggingFace examples show how adapters are activated via control tokens. This is useful
for understanding the underlying mechanics, but **for actual inference, use Mellea** (Path 2),
which provides constrained decoding, prompt formatting, and proper input/output processing.

1. [Prerequisites](PREREQUISITES.md#huggingface-backend)
2. [Hello Adapter](notebooks/00_hello_adapter.ipynb) — see control tokens in action
3. [Granite Switch with HuggingFace](notebooks/02_granite_switch_with_hf.ipynb) — detailed walkthrough

### Path 2: Inference with Mellea (Recommended)

Best for: All inference use cases — development through production

Mellea is the correct way to invoke Granite Switch capabilities. It handles constrained decoding,
prompt rewriting, and input/output processing automatically. Currently supports vLLM; HuggingFace
support coming soon.

1. [Prerequisites](PREREQUISITES.md#vllm-backend)
2. [Hello Mellea](notebooks/01_hello_mellea.ipynb)
3. [RAG Pipeline](notebooks/03_govt_rag_pipeline.ipynb) — full RAG with ChromaDB

### Composing Models

Before running inference, you need a composed Granite Switch model. Options:

1. **Use pre-composed models** from [HuggingFace](https://huggingface.co/ibm-granite/granite-switch-4.1-3b-preview) (recommended for getting started)
2. **Compose your own** — see [Compose Your Checkpoint](notebooks/04_compose_granite_switch.ipynb)

### Path 3: Bring Your Own Adapter

Best for: Custom adapter development

1. [Bring Your Own Adapter Guide](how-to/bring_your_own_adapter.md)

## Notebooks

Interactive Jupyter tutorials in [`notebooks/`](notebooks/):

| Notebook | Topics | Duration |
|----------|--------|----------|
| [01_granite_switch_with_hf.ipynb](notebooks/01_granite_switch_with_hf.ipynb) | Compose + HuggingFace backend, `adapter_name=` invocation, Core + Guardian adapters in a multi-turn conversation | 20 min |
| [02_govt_rag_pipeline.ipynb](notebooks/02_govt_rag_pipeline.ipynb) | Full RAG pipeline, ChromaDB, Guardian | 30 min |
| [03_compose_granite_switch.ipynb](notebooks/03_compose_granite_switch.ipynb) | Compose a checkpoint from adapter libraries | 15 min |

## Adapter Libraries

Granite Switch checkpoints embed adapters drawn from IBM's granitelib libraries. The three libraries below are featured throughout these tutorials:

| Adapter | Purpose | Where used in tutorials | HF repo |
|---------|---------|-------------------------|---------|
| Core | Foundational post-generation intrinsics: certainty scoring, requirement checking, and response attribution. | [01](notebooks/01_granite_switch_with_hf.ipynb), [03](notebooks/03_compose_granite_switch.ipynb) | [ibm-granite/granitelib-core-r1.0](https://huggingface.co/ibm-granite/granitelib-core-r1.0) |
| RAG | Retrieval-augmented generation intrinsics: query rewrite, answerability, hallucination detection, and citation generation. | [hello_mellea](quickstart/hello_mellea.ipynb), [02](notebooks/02_govt_rag_pipeline.ipynb), [03](notebooks/03_compose_granite_switch.ipynb) | [ibm-granite/granitelib-rag-r1.0](https://huggingface.co/ibm-granite/granitelib-rag-r1.0) |
| Guardian | Safety and risk detection: harm, social bias, jailbreaking, factuality, and policy compliance checks. | [hello_adapter](quickstart/hello_adapter.ipynb), [hello_mellea](quickstart/hello_mellea.ipynb), [01](notebooks/01_granite_switch_with_hf.ipynb), [02](notebooks/02_govt_rag_pipeline.ipynb), [03](notebooks/03_compose_granite_switch.ipynb) | [ibm-granite/granitelib-guardian-r1.0](https://huggingface.co/ibm-granite/granitelib-guardian-r1.0) |

## External Resources

| Resource | Description |
|----------|-------------|
| [Mellea](https://github.com/generative-computing/mellea) | IBM's library for writing Generative Programs |
| [Granite aLoRA Adapters](https://huggingface.co/collections/ibm-granite/granite-libraries) | Official adapter libraries on HuggingFace |
| [vLLM Documentation](https://docs.vllm.ai/) | High-performance inference |
| [Granite Models](https://huggingface.co/ibm-granite) | Base Granite models |

## Reference Documentation

For technical details, see [`docs/`](../docs/):

- [Supported Models](../docs/SUPPORTED_MODELS.md) — Model compatibility
- [Git Workflow](../docs/GIT_WORKFLOW.md) — Contribution guidelines
