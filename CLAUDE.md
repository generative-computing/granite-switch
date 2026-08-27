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
│       ├── kernels/                     # Backend-agnostic Triton kernels
│       │   ├── __init__.py
│       │   └── switch_lora_kernel.py    # SWITCH fused-LoRA kernel: expand + swiglu + W_cross shrink
│       │
│       └── vllm/                        # vLLM backend (requires [vllm] extra)
│           ├── __init__.py              # register() for vLLM plugin system
│           ├── granite_switch_model.py  # GraniteSwitch{Model,ForCausalLM}: LoRA + SR, TP/PP
│           ├── core/
│           │   ├── __init__.py
│           │   ├── lora.py              # SwitchedLoRALinear (SWITCH kernel backend)
│           │   ├── lora_kernel_meta.py  # FusedLoRAKernelMeta: per-module remap + tile bitmasks
│           │   └── lora_ops.py          # torch custom-op wrappers for the SWITCH launchers
│           ├── decoder/                 # per-adaptation decoder tier (DecoderInterface)
│           │   ├── __init__.py
│           │   ├── interface.py
│           │   ├── lora/decoder.py      # GraniteLoRAEmbeddedAttention (LoRA / aLoRA)
│           │   └── shadow_residual/     # SR dual-stream decoder
│           │       ├── decoder.py       # ShadowResidualAttention (doubled-Q, TP-aware)
│           │       ├── wcross_shunt.py  # WCrossShunt (base->adapter cross-stream)
│           │       ├── kernel_meta.py   # SRFusedLoRAKernelMeta
│           │       └── _sr_ops.py       # Q-head interleave / deinterleave
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
├── .pre-commit/                         # Pre-commit hook scripts (validate_links.py)
├── .pre-commit-config.yaml              # Pre-commit hook configuration
├── scratch/                             # Throwaway debug/diagnostic scripts (gitignored)
├── docs/                                # Documentation
├── tutorials/                           # Tutorials and how-to guides
├── CLAUDE.md                            # This file
└── README.md
```

## Installation (local/dev)

This project uses [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
# Core package only (config)
uv sync

# With HuggingFace backend
uv sync --extra hf

# With vLLM backend
uv sync --extra vllm

# With compose tools
uv sync --extra compose

# Everything (development)
uv sync --extra dev
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

- SWITCH fused-LoRA Triton kernel for optimized adapter computation (replaces Punica)
- PagedAttention for efficient KV cache
- Continuous batching, tensor/pipeline parallelism
- OpenAI-compatible API server

### Shadow Residual (SR) Dual-Stream

SR runs a frozen base stream and a LoRA adapter stream through the same layer, connected by a
`cross_stream` low-rank injection. The two streams share every parameter (the base stream is
the same modules called with `adapter_indices=None`), and K/V always come from the base
stream. SR uses the same fused projections as the rest of the HF backend.

One `GraniteSwitchModel` / `GraniteSwitchForCausalLM` pair serves both modes and picks its
decoder layer class in `__init__` from `config.dual_stream`: `SRSwitchDecoderLayer` (a
subclass) when True, `GraniteSwitchAttentionDecoderLayer` otherwise. A checkpoint holds
**either** SR adapters **or** standard LoRA/aLoRA adapters — the composer rejects a mixed
set. See [docs/SR_ARCHITECTURE.md](docs/SR_ARCHITECTURE.md) for full details.

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

**SWITCH kernel (vLLM)**: per-module *kernel-local* indices — `0` = base (no adapter, or not applicable to this module), `1..` = that module's applicable adapters in ascending rank order. Global adapter ids are remapped per module centrally (see `vllm/core/lora_kernel_meta.py`).

### 2. Control Token Generatability

All control tokens are freely generatable — there is no runtime suppression. The
model can produce any control token during generation.

### 3. Chat Template Token Placement

- **ALORA adapters**: Token placed either in user message by matching invocation sequence or right before generation prompt
- **LORA adapters**: Token placed at sequence beginning

### 3a. Dual Chat-Template Formats

Two Granite chat-template families are supported, auto-detected by
`detect_template_format` from the base tokenizer's template: the
`<|start_of_role|>…<|end_of_role|>` role-marker format (4.0/4.1) and the ChatML
`<|im_start|>…<|im_end|>` format (4.2). All adapter control-token injection is
driven by the detected `TemplateFormat` rather than hardcoded per format.

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

### 9. Backends Use Fused Projections (Not Bit-Exact with Upstream)

The GraniteSwitch HF backend uses fused QKV and gate-up projections, symmetric with the vLLM
backend architecture. Upstream HuggingFace `GraniteMoeHybridForCausalLM` uses separate projections.
Fused projections change the floating-point reduction order, so bit-exact skinning equivalence
with the upstream HF model is not achievable. The HF skinning tests in
`tests/composer/test_skinning_equivalence.py` are skipped for this reason.

The vLLM check (`tests/vllm/test_generation_equivalence.py`) is the authoritative equivalence
test, but it is **not** token-exact either. The skinned model runs its projections through the
fused SWITCH Triton kernel (`kernels/switch_lora_kernel.py`), whose float-reduction order differs
from vLLM's native linear even with a zero adapter, so the two paths are numerically close but not
bit-identical. That sub-ULP difference can flip a near-tie greedy argmax — observed under vLLM 0.20
but not 0.19, because each version's numerics land the tie on a different side. The test therefore
gates **distribution equivalence** (per-position top-k JSD + Jaccard), not token-for-token match:
robust to benign ties while still catching a real logit/weight regression. See
`tests/vllm/_generation_equivalence_worker.py` for the metric and thresholds.

### 10. Known Limitation: bf16 Caps the vLLM MultiSwitch Counting Head at 188 Control Tokens

The coded engine recovers a control token's write address from a `1/(1+n)` attention signal.
The HF backend forces that signal to fp32 and is exact past 4095; the vLLM backend's heads are
paged-KV `vllm.Attention` modules, so the signal takes the KV-cache dtype and bf16 inverts
exactly only up to `n = 188` (189 aliases). This is a different limit from the codebook's
`capacity == 2048`, which bounds the memory head, not the counting head. Handled only in
`Conversation`, and by fallback rather than refusal: above `MAX_RETAINED_CONTROL_TOKENS = 188` it
re-prefills the turn, resetting the count to one and losing history's adapter attribution (logged;
counted in `Conversation.reprefills`). A raw `/v1/completions` request still bypasses this entirely.
`--kv-cache-dtype fp8` is also unguarded and would saturate `_NEG_INF` and `code(n) * 28`.
See section 8 of [docs/MULTISWITCH_EXPLAINED.md](docs/MULTISWITCH_EXPLAINED.md).

### 11. `MultiSwitch`'s Debug Attributes Exist Only Under `enforce_eager`

`_debug_write_addresses` and `_debug_counting_signal` are written behind
`if not torch.compiler.is_compiling():` (`src/granite_switch/vllm/switch/multi.py`), and the switch
runs inside the `@support_torch_compile` region (`src/granite_switch/vllm/granite_switch_model.py`).
`is_compiling()` is True during tracing, so the body is absent from the compiled graph — a
default-configured server never sets them. Any test reading them must pass `enforce_eager=True` and
must assert they are present, not skip on `None`. The same reason is why a host-side validation check
cannot live in `MultiSwitch.forward`.
### 12. Pre-Fusion SR Checkpoints Must Be Re-Composed

SR used to build unfused Q/K/V and gate/up/down projections (`unfused_qkv=True`); it now uses
the same fused projections as everything else. Since `transformers` silently keeps unknown
config keys, an old unfused checkpoint would otherwise load into a fused model and quietly
mismatch keys, so `GraniteSwitchConfig` rejects `unfused_qkv=True` with a re-compose error.
See [docs/SR_ARCHITECTURE.md](docs/SR_ARCHITECTURE.md) for details.

## Pre-commit

**See [docs/CICD.md](docs/CICD.md) for the full CI/CD setup — pre-commit hook list, setup steps, and what runs on every commit vs. in CI.**

This repo uses [pre-commit](https://pre-commit.com/) (ruff, hygiene hooks, SPDX-header and DCO-signoff checks, and a local link validator) to enforce quality before a commit lands. Hooks run automatically on `git commit` and most are auto-fixing, so re-stage and commit again if a hook modifies files. Do NOT use `--no-verify` to bypass — fix the underlying issue instead.

## Documentation

- `docs/GIT_WORKFLOW.md` - Git branching strategy and commit guidelines
- `docs/SUPPORTED_MODELS.md` - Model compatibility
- `docs/SR_ARCHITECTURE.md` - Shadow Residual dual-stream architecture

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
