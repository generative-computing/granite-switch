---
id: REQ-0010
title: The system should support multiple serving backends
status: draft
version: 1
supersedes: []
superseded_by: null
governs_paths:
  - "src/granite_switch/hf/**"
  - "src/granite_switch/vllm/**"
aspect: deployment-target
issue: null
reviewed_by: null
reviewed_at: null
---

# REQ-0010 · Multiple serving backends

## Requirement (normative)

The system SHOULD support multiple serving backends, including vLLM, HuggingFace Transformers,
llama.cpp, and Ollama. Not all backends need to be implemented in the MVP, but the architecture
MUST NOT preclude supporting them later.

## Rationale

Meeting users on their chosen runtime maximizes reach (traces to **BG-04**).

## Acceptance criteria

- The same checkpoint loads and serves on every in-scope backend (vLLM + HF for the MVP), with the
  design documented not to preclude llama.cpp/Ollama.

## Notes

- Level **SHOULD** · Status **vLLM MVP; HF MVP; others Post-MVP · Partial**.
- Two backends from one checkpoint (**ADR-0001**).
