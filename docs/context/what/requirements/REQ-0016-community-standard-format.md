---
id: REQ-0016
title: The model file uses each target backend's standard format
status: draft
version: 1
supersedes: []
superseded_by: null
governs_paths:
  - "src/granite_switch/composer/**"
  - "src/granite_switch/hf/__init__.py"
  - "src/granite_switch/vllm/__init__.py"
aspect: deployment-target
issue: null
reviewed_by: null
reviewed_at: null
---

# REQ-0016 · Community-standard model format

## Requirement (normative)

The Granite Switch model file MUST use the checkpoint and metadata conventions expected by each
target backend community, so that it loads through that backend's standard path without
backend-specific patches or custom loaders — e.g. HuggingFace `config.json` + safetensors for
Transformers, GGUF for llama.cpp / Ollama, and vLLM's supported-model registration conventions.
The model definition SHOULD be structured for upstream acceptance (in-tree, mergeable rather than
out-of-tree forks); final acceptance depends on each project's maintainers and is therefore a goal.

## Rationale

Loading through standard paths (and upstreamability) is what makes "deploy where the community
runs" real (traces to **BG-04**).

## Acceptance criteria

- The checkpoint loads through each target backend's standard path (HF `config.json` + safetensors;
  vLLM supported-model registration) with no custom loader, shown by standard-path load tests.

## Notes

- Level **MUST** · Status **HF + vLLM MVP; llama.cpp/Ollama Post-MVP · Partial**.
- vLLM registration via `general_plugins`, no fork (**ADR-0007**); HF via AutoConfig/AutoModel
  registration (**ADR-0001**).
