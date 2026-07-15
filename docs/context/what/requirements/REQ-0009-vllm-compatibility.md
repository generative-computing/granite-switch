---
id: REQ-0009
title: The implementation must be compatible with vLLM as a serving backend
status: draft
version: 1
supersedes: []
superseded_by: null
governs_paths:
  - "src/granite_switch/vllm/**"
aspect: deployment-target
issue: null
reviewed_by: null
reviewed_at: null
---

# REQ-0009 · vLLM compatibility

## Requirement (normative)

The implementation MUST be compatible with vLLM as a serving backend.

## Rationale

Deploying where the community already runs (vLLM) is the reach goal (traces to **BG-04**).

## Acceptance criteria

- A composed checkpoint loads and serves on unmodified vLLM via the registered plugin, shown by a
  vLLM serving test.

## Notes

- Level **MUST** · Status **MVP · Satisfied**.
- Registered via the `vllm.general_plugins` entry point, no fork (**ADR-0007**). Version policy —
  default 0.19.1, opt-in 0.20 for CUDA 13+ (**ADR-0011**). Two backends from one checkpoint
  (**ADR-0001**).
