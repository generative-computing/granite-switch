---
id: REQ-0008
title: The implementation must be CUDA-friendly
status: draft
version: 1
supersedes: []
superseded_by: null
governs_paths:
  - "src/granite_switch/vllm/**"
aspect: performance
issue: null
reviewed_by: null
reviewed_at: null
---

# REQ-0008 · CUDA-friendly implementation

## Requirement (normative)

The implementation MUST be CUDA-friendly, avoiding patterns that preclude efficient GPU execution
(e.g. host/device round-trips or graph breaks on the hot path where they can be avoided).

## Rationale

Efficient GPU execution underpins the performance story that makes many-capabilities-at-one-cost
viable (traces to **BG-02**).

## Acceptance criteria

- The decode hot path runs on CUDA with no avoidable host/device round-trips or graph breaks,
  verified by profiling / a graph-break check.

## Notes

- Level **MUST** · Status **MVP · Satisfied**.
- Token-exchange (**ADR-0005**) removed KV-cache padding overhead. Test infrastructure has a
  non-Hopper attention-backend fallback (**ADR-0014**).
