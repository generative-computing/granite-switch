# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

**granite-switch** implements **Granite Switch**, a system for building and deploying Granite models with embedded LoRA adapters. The system is a single unified Python package (`granite_switch`) with optional extras for different backends.

1. **Building models with embedded adapters** - Combine a base Granite model with multiple LoRA adapters into a single checkpoint
2. **Automatic adapter control** - Activate adapters via special control tokens or chat templates
3. **Fast inference** - Deploy with vLLM for speedup over standard HuggingFace inference
4. **Optional trainable switching** - Train a router to automatically select adapters per-token

## Project Structure

```
granite-switch/
├── pyproject.toml                       # Single package definition with optional extras
├── src/
│   └── granite_switch/                  # Unified package
│       ├── __init__.py                  # Core exports (GraniteSwitchConfig, __version__)
│       ├── config.py                    # Unified GraniteSwitchConfig
│       │
│       ├── composer/                    # Compose system (requires [compose] extra)
│       │   ├── __init__.py
│       │   ├── adapter_discovery.py     # Adapter discovery and resolution
│       │   ├── adapter_loader.py        # Adapter weight loading
│       │   ├── arch.py                  # Architecture definitions
│       │   ├── compose_granite_switch.py  # Main compose script (CLI entry point)
│       │   ├── compose_utils.py           # GraniteSwitchComposer class
│       │   ├── tokenizer_setup.py       # Tokenizer configuration for control tokens
│       │   ├── validator.py             # Compose validation checks
│       │   ├── weight_remapper.py       # Adapter name remapping (AdapterRemapper)
│       │   ├── weight_transfer.py       # Base model weight transfer
│       │   └── reporting/               # Compose reporting utilities
│       │       ├── __init__.py
│       │       ├── adapter_analysis.py
│       │       ├── compose_report.py
│       │       ├── hiding_constant_report.py
│       │       ├── model_card.py
│       │       └── population_table.py
│       │
│       ├── hf/                          # HuggingFace backend (requires [hf] extra)
│       │   ├── __init__.py              # Registers with transformers AutoConfig/AutoModel
│       │   ├── modeling_granite_switch.py
│       │   ├── core/
│       │   │   ├── __init__.py
│       │   │   └── lora.py              # SwitchedLoRALinear, MergedSwitchedLoRALinear
│       │   └── switch/
│       │       ├── __init__.py
│       │       └── single.py            # SingleSwitch (HF attention backends)
│       │
│       └── vllm/                        # vLLM backend (requires [vllm] extra)
│           ├── __init__.py              # register() for vLLM plugin system
│           ├── granite_switch_model.py
│           ├── core/
│           │   ├── __init__.py
│           │   ├── lora.py              # SwitchedLoRALinear (Punica kernels)
│           │   ├── lora_kernel_meta.py
│           │   └── decoder.py           # Decoder layers
│           └── switch/
│               ├── __init__.py
│               └── single.py            # SingleSwitch (vLLM Attention)
│
├── tests/                               # All tests
│   ├── unit/                            # Unit tests (fastest, CPU)
│   ├── hf/                              # HuggingFace-specific tests
│   ├── vllm/                            # vLLM-specific tests
│   ├── composer/                        # Compose system tests
│   ├── integration/                     # Cross-backend integration tests
│   ├── regression/                      # Regression tests (hf/, vllm/, integration/, shared/, tools/)
│   └── shared/                          # Shared test utilities and parametrized cases
│
├── scratch/                             # Throwaway debug/diagnostic scripts (gitignored)
├── docs/                                # Documentation
├── tutorials/                           # Tutorials and how-to guides
├── CLAUDE.md                            # This file
└── README.md
```

## Installation (local/dev)

```bash
# Core package only (config)
pip install -e .

# With HuggingFace backend
pip install -e ".[hf]"

# With vLLM backend
pip install -e ".[vllm]"

# With compose tools
pip install -e ".[compose]"

# Everything (development)
pip install -e ".[dev]"
```

## Import Paths

```python
# Config (shared by all backends)
from granite_switch import GraniteSwitchConfig
from granite_switch.config import GraniteSwitchConfig  # equivalent

# HuggingFace backend
from granite_switch.hf import GraniteSwitchForCausalLM
from granite_switch.hf.core.lora import SwitchedLoRALinear
from granite_switch.hf.switch.single import SingleSwitch

# vLLM backend (auto-registered via plugin entry point)
from granite_switch.vllm import register

# Compose system
from granite_switch.composer import GraniteSwitchComposer
```

## File Organization Convention

**IMPORTANT:** Keep the repository organized by placing files in their designated directories.

### Documentation Files (Markdown)

**All `.md` documentation files MUST go in a `docs/` directory:**

- **Root-level docs (`docs/`)**: Cross-implementation documentation, guides, and architecture docs
- **Exceptions**: Only `CLAUDE.md` and `README.md` may be at the repository root

### Test Files (Python)

**All `test_*.py` test files MUST go in a `tests/` directory:**

- **`tests/unit/`**: Unit tests (fastest, CPU-only)
- **`tests/hf/`**: HuggingFace implementation tests
- **`tests/vllm/`**: vLLM implementation tests
- **`tests/composer/`**: Compose system tests
- **`tests/integration/`**: Cross-implementation and end-to-end integration tests
- **`tests/regression/`**: Regression tests (hf/, vllm/, integration/, shared/, tools/)
- **`tests/shared/`**: Shared test utilities and parametrized cases

**IMPORTANT: `tests/` is for official regression tests ONLY.** Do NOT place throwaway diagnostic,
debugging, or exploratory scripts in `tests/`. Use `scratch/` instead (it is gitignored). Running
`pytest tests/` should only execute curated, maintained tests — never one-off investigations.

### Naming Conventions

- **Test files**: `test_*.py`
- **Documentation**: `UPPER_CASE.md`
- **Scripts**: `snake_case.py`

## Development Commands

### Composing Models

```bash
# Compose with HuggingFace adapters
python -m granite_switch.composer.compose_granite_switch \
  --adapters ibm-granite/granitelib-rag-r1.0

# Multiple adapters
python -m granite_switch.composer.compose_granite_switch \
  --adapters ibm-granite/granitelib-rag-r1.0 your-org/extra-adapter

# Custom output directory
python -m granite_switch.composer.compose_granite_switch \
  --adapters ibm-granite/granitelib-rag-r1.0 --output ./my-custom-model
```

### Testing

**Always use `-v -s --tb=short`** when running tests. `-v` (verbose) prints each test name as
it starts, giving real-time progress visibility. `-s` disables output capture so `print()`
statements inside tests appear immediately instead of being swallowed. Without these, long-running
test files produce no output until they finish. `-x` (fail fast) stops on the first failure —
no point running 200 more tests after something breaks.

**Check GPU availability first** — the underlying hardware can change between sessions:

```bash
python -c "import torch; print('GPU' if torch.cuda.is_available() else 'CPU only')"
```

This determines which tests can run. vLLM and integration tests require a GPU; unit and HF tests
run on CPU.

**Run tests incrementally by directory**, in order of speed — don't run the full suite as a
single command:

```bash
# 1. Unit tests first (fastest, CPU)
pytest tests/unit/ -v -s --tb=short -x

# 2. HF tests by file (CPU)
pytest tests/hf/test_single_switch.py -v -s --tb=short -x
pytest tests/hf/test_model_forward.py -v -s --tb=short -x

# 3. vLLM tests by file (GPU required)
pytest tests/vllm/test_single_switch.py -v -s --tb=short -x
pytest tests/vllm/test_model_forward.py -v -s --tb=short -x

# 4. Integration tests last (slowest, GPU required)
pytest tests/integration/ -v -s --tb=short -x

# Run a specific test pattern when debugging
pytest tests/ -k "pattern" -v -s --tb=short -x
```

### vLLM Deployment

```bash
# Verify plugin registration
python -c "from vllm.plugins import load_general_plugins; \
           from vllm import ModelRegistry; \
           load_general_plugins(); \
           print('OK' if 'GraniteSwitchForCausalLM' in ModelRegistry.get_supported_archs() else 'FAIL')"

# Start API server
python -m vllm.entrypoints.openai.api_server \
  --model ./granite-with-all-aloras \
  --port 8000
```

## Architecture

### Granite Switch Model

The Granite Switch extends the base Granite model with:

1. **Embedded LoRA Adapters** (frozen during inference)
   - Multiple task/domain-specific adapters embedded in the same checkpoint
   - Each adapter has LoRA weights (lora_A, lora_B) stacked in tensors
   - Controlled via special tokens or router-selected indices

2. **Control Tokens**
   - Each adapter has a control token `<|adapter|>` that fires the switch
   - KV hiding uses group-based control dimensions (K=finfo.min, Q=per-adapter policy)
   - Control tokens are KV-hidden to prevent cross-request interference

3. **Chat Template Integration**
   - Maps adapter names to control tokens
   - Automatic token placement based on adapter type (ALORA vs LORA)

4. **Optional Trainable Router** (SingleSwitch)
   - N transformer layers that compute adapter indices per-token
   - Linear projection head to num_adapters dimensions
   - ~1-2% of total model parameters

### Two Backends

#### HuggingFace Backend (`granite_switch.hf`)

**Purpose**: Model building and optional router training

- Full `transformers` integration (`PreTrainedModel`, `GenerationMixin`)
- Training with `Trainer` API
- Standard PyTorch operations

#### vLLM Backend (`granite_switch.vllm`)

**Purpose**: Fast production inference (10-20x speedup)

- Punica kernels for optimized LoRA computation
- PagedAttention for efficient KV cache
- Continuous batching, tensor/pipeline parallelism
- OpenAI-compatible API server

### Weight Compatibility

Both backends share the same weight format:

```python
# Built/trained with HuggingFace
model_hf.save_pretrained("./checkpoint")

# Loaded directly with vLLM
llm = LLM(model="./checkpoint")
```

## Key Configuration Parameters

### Granite-Specific Parameters

- **`attention_multiplier`**: Attention score scaling (instead of `1/sqrt(head_dim)`)
- **`logits_scaling`**: Applied to final logits (main architectural difference with Llama)
- **`residual_multiplier`**: Applied to residual connections
- **`embedding_multiplier`**: Applied to input embeddings

Always use config values - never hardcode these parameters.

### Switch Configuration

```json
{
  "model_type": "granite_switch",
  "architectures": ["GraniteSwitchForCausalLM"],
  "num_adapters": 4,
  "adapter_token_ids": [100, 101, 102, 103],
  "adapter_names": ["adapter_0", "adapter_1", "adapter_2", "adapter_3"],
  "hiding_groups": {"all_controls": ["adapter_0", "adapter_1", "adapter_2", "adapter_3"]},
  "hiding_policy": {"base": ["all_controls"], "adapter_0": ["all_controls"], "...": "..."},
  "lora_rank": 8,
  "lora_alpha": 8.0,
  "switch_head_dim": 32,
  "control_dims": 32
}
```

## Common Gotchas

### 1. Adapter Index Convention

**Control tokens**: `0` = no adapter, `1+` = adapter indices

**vLLM Punica kernels**: `-1` = no adapter (internal conversion: `adapter_indices - 1`)

### 2. Control Token Generatability

All control tokens are freely generatable — there is no runtime suppression. The
model can produce any control token during generation.

### 3. Chat Template Token Placement

- **ALORA adapters**: Token placed either in user message by matching invocation sequence or right before generation prompt
- **LORA adapters**: Token placed at sequence beginning

### 4. Granite vs Llama Differences

- Granite uses `logits_scaling` (typically 8.0)
- Custom attention scaling via `attention_multiplier`
- Different residual and embedding multipliers

Always load from config, never hardcode.

### 5. End-to-End Tests Must Use Compose Infrastructure

No test should manually assemble `GraniteSwitchConfig` or call `transfer_base_weights`
directly.  All model construction must go through `GraniteSwitchComposer` so that the
compose pipeline itself is what's being tested.  If the composer can't handle a use case
(e.g., zero-adapter skinning), extend the composer — don't work around it in tests.

### 6. HF Attention Backends and Causal Masking

The eager backend does NOT handle `attention_mask=None` as causal — it treats `None` as no mask
(full attention). SDPA and FlashAttention handle `attention_mask=None` correctly via `is_causal`
attribute on the module.

The HF stress tests (`tests/hf/test_single_switch.py`) auto-detect which attention backends work on the
current platform by probing each with a k=-inf GQA call at import time. Unavailable backends are skipped.

### 7. Known Limitation: Hidden Count Offset When Position 0 is in a Hiding Group

When position 0 is a control token in a hiding group (e.g., a LoRA prefix token with
`add_bos_token=False`), `hidden_count` is off by 1, causing a 1-position RoPE offset. This is
acceptable because adapter detection is exact and RoPE is robust to small positional shifts.

### 8. Known Limitation: TP Row-Parallel Bias Doubling

`SwitchedLoRALinear`'s row-parallel bypass path passes bias to all TP ranks instead of
suppressing it for rank > 0. After all-reduce this doubles the bias. Not affected: all Granite
architectures (4.0, 4.1) use `attention_bias=False` and `mlp_bias=False`.

### 9. HF Backend Uses Fused Projections (Not Bit-Exact with Upstream HF)

The GraniteSwitch HF backend uses fused QKV and gate-up projections, symmetric with the vLLM
backend architecture. Upstream HuggingFace `GraniteMoeHybridForCausalLM` uses separate projections.
Fused projections change the floating-point reduction order, so bit-exact skinning equivalence
with the upstream HF model is not achievable. The vLLM skinning equivalence tests are the
authoritative check — both the upstream and skinned models use the same fused-projection
architecture there. The HF skinning tests in `tests/composer/test_skinning_equivalence.py` are
skipped for this reason.

## Documentation

- `docs/GIT_WORKFLOW.md` - Git branching strategy and commit guidelines
- `docs/SUPPORTED_MODELS.md` - Model compatibility

## Git Workflow

**See [docs/GIT_WORKFLOW.md](docs/GIT_WORKFLOW.md) for complete git workflow guidelines.**

**Quick reference:**

- **Branch naming**: `feature/ticket-ID-description` or `bugfix/ticket-ID-description`
- **Workflow**: Branch from `main` → develop → rebase → PR → merge → delete branch
- **Critical**: Always verify comments match code before committing (see GIT_WORKFLOW.md)
- **Commit format**: Clear summary + explanation of WHAT changed and WHY

When committing, **never sign as Claude** (per project instructions)

## License

Apache-2.0 (as indicated by SPDX headers in source files)
