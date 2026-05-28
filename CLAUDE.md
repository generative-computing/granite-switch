# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

**granite-switch** is a single Python package (`granite_switch`) for building and deploying Granite models with embedded LoRA adapters. Two backends share the same weight format: `granite_switch.hf` (HuggingFace, training) and `granite_switch.vllm` (production inference, 10-20x speedup via Punica kernels + PagedAttention).

## Project Structure

Key layout rules — full tree via `find src/` or `find tests/`:

- `src/granite_switch/` — unified package; `composer/`, `hf/`, `vllm/` match the optional extras
- `tests/` — official test suite only; subdirs: `unit/`, `hf/`, `vllm/`, `composer/`, `integration/`, `regression/`, `shared/`
- `tutorials/` — notebooks and guides; see `tutorials/CLAUDE.md` for conventions

## Installation (local/dev)

```bash
pip install -e ".[dev]"         # everything (recommended for development)
pip install -e ".[hf,compose]"  # HF + composer only (no vLLM)
```

## File Organization Convention

**IMPORTANT:** Keep the repository organized by placing files in their designated directories.

### Documentation Files (Markdown)

**All `.md` documentation files MUST go in a `docs/` directory:**

- **Root-level docs (`docs/`)**: Cross-implementation documentation, guides, and architecture docs
- **Exceptions**: Only `CLAUDE.md` and `README.md` may be at the repository root

### Test Files (Python)

**`tests/` is for official regression tests ONLY.** Do NOT place throwaway diagnostic,
debugging, or exploratory scripts in `tests/` — `pytest tests/` should only execute
curated, maintained tests, never one-off investigations. Subdirectories are listed in
Project Structure above.

### Documentation Naming

`UPPER_CASE.md` for docs under `docs/`.

## Development Commands

### Testing

**Always use `-v -s --tb=short`** when running tests. `-x` (fail fast) stops on the first failure —
no point running 200 more tests after something breaks.

**Check GPU availability first** — the underlying hardware can change between sessions:

```bash
python -c "import torch; print('GPU' if torch.cuda.is_available() else 'CPU only')"
```

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
```

## Key Configuration Parameters

- **`attention_multiplier`**: Attention score scaling (instead of `1/sqrt(head_dim)`)
- **`logits_scaling`**: Applied to final logits
- **`residual_multiplier`**: Applied to residual connections
- **`embedding_multiplier`**: Applied to input embeddings

Always use config values — never hardcode these parameters.

## Common Gotchas

### 1. Adapter Index Convention

`0` = no adapter, `1+` = adapter index. (vLLM Punica kernels use a shifted convention internally —
see `src/granite_switch/vllm/CLAUDE.md`.)

### 2. Control Token Generatability

All control tokens are freely generatable — there is no runtime suppression. The
model can produce any control token during generation.

### 3. Chat Template Token Placement

- **ALORA adapters**: Token placed either in user message by matching invocation sequence or right before generation prompt
- **LORA adapters**: Token placed at sequence beginning

### 4. Hidden Count Offset When Position 0 is in a Hiding Group

When position 0 is a control token in a hiding group (e.g., a LoRA prefix token with
`add_bos_token=False`), `hidden_count` is off by 1, causing a 1-position RoPE offset. This is
acceptable because adapter detection is exact and RoPE is robust to small positional shifts.

### Backend- and module-specific gotchas

Loaded on demand from child CLAUDE.md files when you touch those modules:

- `src/granite_switch/hf/CLAUDE.md` — HF attention backends, fused projections vs upstream HF
- `src/granite_switch/vllm/CLAUDE.md` — Punica `-1` index, TP row-parallel bias, deployment commands
- `src/granite_switch/composer/CLAUDE.md` — compose-infra rule for e2e tests, compose CLI

## Documentation

- `docs/ARCHITECTURE.md` - Architecture overview (control tokens, backends, SingleSwitch)
- `docs/GIT_WORKFLOW.md` - Git branching strategy and commit guidelines
- `docs/SUPPORTED_MODELS.md` - Model compatibility

## Git Workflow

See [docs/GIT_WORKFLOW.md](docs/GIT_WORKFLOW.md) for branch naming, commit format, and
PR workflow. **When committing, never sign as Claude** (per project instructions).

## License

Apache-2.0 (as indicated by SPDX headers in source files)
