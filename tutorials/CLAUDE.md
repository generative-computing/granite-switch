# CLAUDE.md — tutorials/

This file provides guidance when working on notebooks and guides in this directory.
Claude loads it automatically when reading any file under `tutorials/`.

## Notebook Cell Ordering

Every notebook follows this cell order:

1. `%pip install ...` — dependencies
2. HF login cell (see below)
3. Imports
4. Configuration (model path, ports, constants)
5. Long-running steps (corpus build, model load, vLLM launch)

## HF Login Cell

Every notebook that downloads gated HF models (`ibm-granite/`) must have a dedicated cell
immediately after pip install:

```python
from huggingface_hub import notebook_login
notebook_login()  # needed to pull ibm-granite models from the Hub
```

Use cell id `hf-login-call` for consistency.

## Duration Comments

Add `# Estimated duration: ~2 min on A100, ~7 min on T4` to cells that download models or
launch vLLM. Put these in **notebook cells only** — not in code files under `src/`.

## Utility Modules

These live in `src/granite_switch/tutorials/` and are imported by notebooks:

- `vllm_server.py` — `launch_vllm()`, `wait_for_server()` (reads the vLLM log and prints
  stage-based progress), `kill_stale_vllm_processes()`
- `chroma_loader.py` — `load_or_build_chroma()`: builds corpus on GPU, frees GPU memory with
  `torch.cuda.empty_cache()`, then switches to CPU for queries so vLLM can use the full GPU
