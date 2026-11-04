# Granite Switch Composer

The composer builds Granite Switch models by embedding LoRA adapters into a base Granite checkpoint. The output is a single model directory that can be loaded directly by the HuggingFace or vLLM backends.

## Available Adapter Libraries

The [Granite Libraries](https://huggingface.co/collections/ibm-granite/granite-libraries) collection provides pre-trained adapter libraries for Granite models:

| Library | Release | Adapters |
|---------|---------|----------|
| **RAG** | `ibm-granite/granitelib-rag-r1.0` | answerability, citations, context_relevance, hallucination_detection, query_clarification, query_rewrite |
| **Core** | `ibm-granite/granitelib-core-r1.0` | context-attribution, requirement-check, uncertainty |
| **Guardian** | `ibm-granite/granitelib-guardian-r1.0` | factuality-correction, factuality-detection, guardian-core, policy-guardrails |

## Quick Start

```bash
# Build with all adapters from a single library (Granite 4.1)
python -m granite_switch.composer.compose_granite_switch \
  --base-model ibm-granite/granite-4.1-3b \
  --adapters ibm-granite/granitelib-rag-r1.0

# Build with all adapters from all libraries (Granite 4.1)
python -m granite_switch.composer.compose_granite_switch \
  --base-model ibm-granite/granite-4.1-3b \
  --adapters ibm-granite/granitelib-rag-r1.0 \
             ibm-granite/granitelib-core-r1.0 \
             ibm-granite/granitelib-guardian-r1.0
```

## Adapter Sources

Adapters can come from four places:

1. **HuggingFace adapter libraries** -- repositories containing multiple adapters in the `adapter_name/model/technology/` layout
2. **Single adapter directories** -- a local path or HF repo pointing directly to one adapter (contains `adapter_config.json`)
3. **YAML manifest files** -- a `.yaml`/`.yml` file listing adapter names, paths, and types
4. **Built-in adapter slots** -- empty LoRA slots created via `--built-in-adapters`

```bash
# HuggingFace libraries (auto-downloaded)
--adapters ibm-granite/granitelib-rag-r1.0 ibm-granite/granitelib-core-r1.0

# Local adapter library
--adapters /path/to/adapter-library

# Single adapter (local)
--adapters /path/to/adapter_name/granite-4.1-3b/alora

# YAML manifest
--adapters /path/to/adapters.yaml

# Mix of sources
--adapters ibm-granite/granitelib-rag-r1.0 /path/to/custom-adapter adapters.yaml

# Built-in empty slots (no external weights)
--built-in-adapters base reasoning
```

### YAML Manifest Format

A YAML manifest lists adapters with explicit paths and technology types:

```yaml
answerability:
  path: "/path/to/answerability/granite-4.1-3b/alora"
  type: "alora"

context_relevance:
  path: "/path/to/context_relevance/granite-4.1-3b/lora"
  type: "lora"
```

Each top-level key is the adapter name. Required fields per adapter:
- **`path`** -- absolute path to the adapter directory (must contain `adapter_model.safetensors` and `adapter_config.json`)
- **`type`** -- adapter technology: `"alora"` or `"lora"`

## Selecting Adapters

When using an adapter library, the composer discovers all adapters matching your target model by default. Use these flags to select a subset:

### Include specific adapters

```bash
# By exact name -- pick adapters from different libraries
--include-adapters answerability context-attribution guardian-core

# By glob pattern (fnmatch syntax: *, ?, [seq])
--include-adapters 'query_*' 'factuality-*'

# Mix of exact names and patterns
--include-adapters uncertainty 'query_*'
```

### Exclude adapters

```bash
# Exclude by name
--exclude-adapters hallucination_detection

# Exclude by pattern
--exclude-adapters 'factuality-*'
```

Include is applied first, then exclude. If both are used, only adapters passing both filters are included.

### Filter by technology

Adapters come in two technology types: **alora** (preferred) and **lora**. By default, when both exist for the same adapter, alora is used.

```bash
# Only use lora adapters
--technology-filter lora

# Only use alora adapters
--technology-filter alora
```

Note: `--technology-filter` *filters* which adapters are discovered. This is different from `--technology`, which provides a *fallback* technology for adapters whose type cannot be auto-detected from the directory name (e.g., adapters in non-standard directory layouts).

### List available adapters

Preview what's in a library before building:

```bash
python -m granite_switch.composer.compose_granite_switch \
  --adapters ibm-granite/granitelib-rag-r1.0 \
             ibm-granite/granitelib-core-r1.0 \
             ibm-granite/granitelib-guardian-r1.0 \
  --list-adapters
```

## Full CLI Reference

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--adapters` | `str ...` | `[]` | HuggingFace repo IDs, local paths, or YAML manifest files |
| `--base-model` | `str` | `ibm-granite/granite-4.1-3b` | Base Granite model (HF repo or local path) |
| `--target-model` | `str` | derived from `--base-model` | Target model name for adapter discovery |
| `--output` | `str` | `./granite-with-all-aloras` | Output directory for the composed model |
| `--include-adapters` | `str ...` | all | Only include adapters matching these names/patterns |
| `--exclude-adapters` | `str ...` | none | Exclude adapters matching these names/patterns |
| `--technology-filter` | `alora\|lora` | both | Only include adapters of this technology type |
| `--technology` | `alora\|lora` | auto-detect | Fallback technology when auto-detection fails |
| `--list-adapters` | flag | off | List available adapters and exit (no build) |
| `--built-in-adapters` | `str ...` | `[]` | Names for built-in (empty LoRA) adapter slots |
| `--lora-rank` | `int` | `8` | LoRA rank for built-in adapters |
| `--lora-alpha` | `float` | same as rank | LoRA alpha for built-in adapters |
| `--switch-head-dim` | `int` | auto | Dimension of Q/K/V vectors in switch attention |
| `--control-dims` | `int` | auto | Extra dims for K/V to mask control tokens |

## Examples

```bash
# Build with all adapters from all libraries (Granite 4.1)
python -m granite_switch.composer.compose_granite_switch \
  --base-model ibm-granite/granite-4.1-3b \
  --adapters ibm-granite/granitelib-rag-r1.0 \
             ibm-granite/granitelib-core-r1.0 \
             ibm-granite/granitelib-guardian-r1.0

# Pick specific adapters across libraries
python -m granite_switch.composer.compose_granite_switch \
  --base-model ibm-granite/granite-4.1-3b \
  --adapters ibm-granite/granitelib-rag-r1.0 \
             ibm-granite/granitelib-guardian-r1.0 \
  --include-adapters answerability citations guardian-core

# Exclude specific adapters
python -m granite_switch.composer.compose_granite_switch \
  --base-model ibm-granite/granite-4.1-3b \
  --adapters ibm-granite/granitelib-rag-r1.0 \
  --exclude-adapters hallucination_detection query_clarification

# Build with only lora-type adapters
python -m granite_switch.composer.compose_granite_switch \
  --base-model ibm-granite/granite-4.1-3b \
  --adapters ibm-granite/granitelib-rag-r1.0 \
  --technology-filter lora

# Combine library adapters with built-in slots
python -m granite_switch.composer.compose_granite_switch \
  --base-model ibm-granite/granite-4.1-3b \
  --adapters ibm-granite/granitelib-core-r1.0 \
  --include-adapters uncertainty requirement-check \
  --built-in-adapters base

# Build from a YAML manifest
python -m granite_switch.composer.compose_granite_switch \
  --adapters /path/to/adapters.yaml

# Mix YAML manifest with HuggingFace libraries
python -m granite_switch.composer.compose_granite_switch \
  --adapters adapters.yaml ibm-granite/granitelib-core-r1.0

# Custom output directory
python -m granite_switch.composer.compose_granite_switch \
  --base-model ibm-granite/granite-4.1-3b \
  --adapters ibm-granite/granitelib-rag-r1.0 \
  --output ./my-custom-model
```

## Output

The composer produces a model directory containing:

- `config.json` -- GraniteSwitchConfig with adapter metadata
- `model*.safetensors` -- Model weights with embedded LoRA adapters
- `tokenizer*` -- Tokenizer with added control tokens
- `adapter_index.json` -- Maps adapter indices to names, tokens, and configs
- `io_configs/` -- Per-adapter io.yaml files (copied from source adapters)
- Upstream auxiliary files (generation_config.json, LICENSE, etc.)

This directory can be loaded directly:

```python
# HuggingFace
from granite_switch.hf import GraniteSwitchForCausalLM

model = GraniteSwitchForCausalLM.from_pretrained("./granite-with-all-aloras")

# vLLM
from vllm import LLM

llm = LLM(model="./granite-with-all-aloras")
```
