---
id: REQ-0019
title: One checkpoint loads on both HF and vLLM with no conversion
status: draft
version: 1
supersedes: []
superseded_by: null
governs_paths:
  - "src/granite_switch/composer/**"
  - "src/granite_switch/hf/modeling_granite_switch.py"
  - "src/granite_switch/vllm/granite_switch_model.py"
aspect: architecture
issue: null
reviewed_by: null
reviewed_at: null
---

# REQ-0019 · Single unified checkpoint across backends

## Requirement (normative)

A composed model MUST be a single checkpoint that loads and runs on both the HuggingFace and vLLM
backends without a conversion step — the same weights, saved once, are consumed by either backend.

## Rationale

No conversion tax between build/prototype and production serving is the two-backends-one-artifact
promise (traces to **BG-04**).

## Acceptance criteria

- A single saved checkpoint loads and runs on both HF and vLLM with no conversion step, shown by a
  cross-backend load test.

## Notes

- Level **MUST** · Status **MVP · Satisfied**.
- Two backends sharing one weight format (**ADR-0001**); fused projections symmetric across
  backends make the weights interchangeable (**ADR-0006**).
