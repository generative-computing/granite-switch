# Granite Switch Tutorials

Granite Switch facilitates a modular architecture by consolidating multiple LoRA adapters into a single, unified checkpoint. The following tutorials explore the underlying mechanics and usability, detailing adapter invocation, multi-step pipelines with guardrails, and checkpoint composition.

## Notebooks

Step-by-step walkthroughs covering adapter invocation, pipeline construction, and model composition.

| Notebook | Topics | Duration | Colab |
|----------|--------|----------|-------|
| [00_hello_adapter.ipynb](notebooks/00_hello_adapter.ipynb) | Minimal adapter invocation with HuggingFace | 5 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/generative-computing/granite-switch/blob/main/tutorials/notebooks/00_hello_adapter.ipynb) |
| [01_hello_mellea.ipynb](notebooks/01_hello_mellea.ipynb) | Mellea intrinsics intro with vLLM | 5 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/generative-computing/granite-switch/blob/main/tutorials/notebooks/01_hello_mellea.ipynb) |
| [02_granite_switch_with_hf.ipynb](notebooks/02_granite_switch_with_hf.ipynb) | Compose + HuggingFace backend, `adapter_name=` invocation, Core + Guardian adapters in a multi-turn conversation | 10 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/generative-computing/granite-switch/blob/main/tutorials/notebooks/02_granite_switch_with_hf.ipynb) |
| [03_01_govt_rag_pipeline_simple.ipynb](notebooks/03_01_govt_rag_pipeline_simple.ipynb) | Simple RAG pipeline without guardians (rewrite, answerability, citations) | 30 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/generative-computing/granite-switch/blob/main/tutorials/notebooks/03_01_govt_rag_pipeline_simple.ipynb) |
| [03_02_govt_rag_pipeline_sequential.ipynb](notebooks/03_02_govt_rag_pipeline_sequential.ipynb) | Full RAG pipeline with guardian checks (harm + scope) | 30 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/generative-computing/granite-switch/blob/main/tutorials/notebooks/03_02_govt_rag_pipeline_sequential.ipynb) |
| [04_compose_granite_switch.ipynb](notebooks/04_compose_granite_switch.ipynb) | Compose a checkpoint from adapter libraries | 15 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/generative-computing/granite-switch/blob/main/tutorials/notebooks/04_compose_granite_switch.ipynb) |
| [05_alora_vs_lora_race.ipynb](notebooks/05_alora_vs_lora_race.ipynb) | ALORA vs LoRA race: side-by-side throughput comparison on a multi-step RAG pipeline | 20 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/generative-computing/granite-switch/blob/main/tutorials/notebooks/05_alora_vs_lora_race.ipynb) |

## Guides

| Guide | Description |
|-------|-------------|
| [Using Mellea with Granite Switch](guides/mellea_with_granite_switch.md) | Connect Mellea to a Granite Switch model |
| [Bring Your Own Adapter](guides/bring_your_own_adapter.md) | Train, compose, and use custom adapters |
| [Compare Inference Throughput](guides/compare_inference_throughput.md) | Compare LoRA vs aLoRA based models in an inference race setup |

## Learning Paths
### Composing Models

for the following learning path we will use pre built models [HuggingFace](https://huggingface.co/ibm-granite/granite-switch-4.1-3b-preview) you can compose your on model see Path 3 Compose your own model



### Path 1: Inference with Mellea (Recommended)

Best for: All inference use cases — development through production

Mellea is the correct way to invoke Granite Switch capabilities. It handles constrained decoding,
prompt rewriting, and input/output processing automatically. Currently supports vLLM; HuggingFace
support coming soon.

1. [Prerequisites](PREREQUISITES.md#vllm-backend)
2. [Hello Mellea](notebooks/01_hello_mellea.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/generative-computing/granite-switch/blob/main/tutorials/notebooks/01_hello_mellea.ipynb)





### Path 2: Real-World Pipelines (Usability)

Best for: Seeing how adapters compose into multi-step applications

1. [Simple RAG Pipeline](notebooks/03_01_govt_rag_pipeline_simple.ipynb) — rewrite, answerability, citations [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/generative-computing/granite-switch/blob/main/tutorials/notebooks/03_01_govt_rag_pipeline_simple.ipynb)
2. [Sequential RAG with Guardians](notebooks/03_02_govt_rag_pipeline_sequential.ipynb) — harm + scope checks [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/generative-computing/granite-switch/blob/main/tutorials/notebooks/03_02_govt_rag_pipeline_sequential.ipynb)





### Path 3: Bring Your Own Adapter

Best for: Custom adapter development

1. [Bring Your Own Adapter Guide](guides/bring_your_own_adapter.md)

### Composing Models
**Compose your own** — see [Compose Your Checkpoint](notebooks/04_compose_granite_switch.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/generative-computing/granite-switch/blob/main/tutorials/notebooks/04_compose_granite_switch.ipynb)


### Low-Level Understanding (HuggingFace)

Best for: Understanding how Granite Switch works at the control-token level

HuggingFace inference examples demonstrate how adapters are activated via control tokens, providing insight into the underlying mechanics. For most applications, we recommend running inference with Mellea (Part 2).




1. [Prerequisites](PREREQUISITES.md#huggingface-backend)
2. [Hello Adapter](notebooks/00_hello_adapter.ipynb) — see control tokens in action [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/generative-computing/granite-switch/blob/main/tutorials/notebooks/00_hello_adapter.ipynb)
3. [Granite Switch with HuggingFace](notebooks/02_granite_switch_with_hf.ipynb) — detailed walkthrough [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/generative-computing/granite-switch/blob/main/tutorials/notebooks/02_granite_switch_with_hf.ipynb)



### Reference Scripts

Runnable scripts in [`scripts/`](scripts/) for common tasks:

| Script | Description |
|--------|-------------|
| [run_adapter_generation_direct.py](scripts/reference/run_adapter_generation_direct.py) | Direct adapter invocation via control tokens |
| [run_adapter_generation_mellea.py](scripts/reference/run_adapter_generation_mellea.py) | Adapter invocation through Mellea |


## Adapter Libraries

Granite Switch checkpoints embed adapters drawn from IBM's granitelib libraries. The three libraries below are featured throughout these tutorials:

| Adapter | Purpose | Where used in tutorials | HF repo |
|---------|---------|-------------------------|---------|
| Core | Foundational post-generation intrinsics: certainty scoring, requirement checking, and response attribution. | [02](notebooks/02_granite_switch_with_hf.ipynb), [04](notebooks/04_compose_granite_switch.ipynb) | [ibm-granite/granitelib-core-r1.0](https://huggingface.co/ibm-granite/granitelib-core-r1.0) |
| RAG | Retrieval-augmented generation intrinsics: query rewrite, answerability, hallucination detection, and citation generation. | [01](notebooks/01_hello_mellea.ipynb), [03_01](notebooks/03_01_govt_rag_pipeline_simple.ipynb), [03_02](notebooks/03_02_govt_rag_pipeline_sequential.ipynb), [04](notebooks/04_compose_granite_switch.ipynb) | [ibm-granite/granitelib-rag-r1.0](https://huggingface.co/ibm-granite/granitelib-rag-r1.0) |
| Guardian | Safety and risk detection: harm, social bias, jailbreaking, factuality, and policy compliance checks. | [00](notebooks/00_hello_adapter.ipynb), [01](notebooks/01_hello_mellea.ipynb), [02](notebooks/02_granite_switch_with_hf.ipynb), [03_02](notebooks/03_02_govt_rag_pipeline_sequential.ipynb), [04](notebooks/04_compose_granite_switch.ipynb) | [ibm-granite/granitelib-guardian-r1.0](https://huggingface.co/ibm-granite/granitelib-guardian-r1.0) |

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
