---
id: REQ-0006
title: LoRA and aLoRA adapters are composable together
status: draft
version: 1
supersedes: []
superseded_by: null
governs_paths:
  - "src/granite_switch/composer/adapter_discovery.py"
  - "src/granite_switch/composer/tokenizer_setup.py"
aspect: composition
issue: null
reviewed_by: null
reviewed_at: null
---

# REQ-0006 · Mixed adapter types

## Requirement (normative)

LoRA and aLoRA adapters MUST be composable together within the same model.

## Rationale

Heterogeneous composition (mixed types in one checkpoint) is central to compose-like-software
(traces to **BG-01**).

## Acceptance criteria

- At least one LoRA and one aLoRA adapter compose in the same checkpoint and each activates
  correctly, shown by a mixed-type composition test.

## Notes

- Level **MUST** · Status **MVP · Satisfied**.
- Token placement differs by type (aLoRA vs LoRA) — see **ADR-0004**; shared-KV composition is
  **ADR-0003**.
