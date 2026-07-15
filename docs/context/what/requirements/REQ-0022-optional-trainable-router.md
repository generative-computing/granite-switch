---
id: REQ-0022
title: An optional trainable router may select adapters per token
status: draft
version: 1
supersedes: []
superseded_by: null
governs_paths:
  - "src/granite_switch/hf/switch/**"
  - "src/granite_switch/vllm/switch/**"
aspect: switching-mechanism
issue: null
reviewed_by: null
reviewed_at: null
---

# REQ-0022 · Optional trainable router

## Requirement (normative)

The system MAY provide an optional trainable router that selects the active adapter per token
without explicit control tokens, at a small parameter cost relative to the base model. When
present, it MUST NOT be required for the control-token path (**REQ-0001**) to function.

## Rationale

Offering implicit selection for callers who prefer it, without making it mandatory, keeps the
compose-like-software model flexible (traces to **BG-05**, and **BG-01**/**BG-02**).

## Acceptance criteria

- The optional router selects adapters per token without explicit control tokens, and the
  control-token path still works with the router absent, shown by a router-selection test.

## Notes

- Level **MAY** · Status **Optional · Satisfied**.
- `SingleSwitch` router is the optional path in **ADR-0004**.
