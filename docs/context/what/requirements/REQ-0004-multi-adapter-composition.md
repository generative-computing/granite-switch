---
id: REQ-0004
title: Compose multiple adapters (at least 20) into one model
status: draft
version: 1
supersedes: []
superseded_by: null
governs_paths:
  - "src/granite_switch/composer/**"
aspect: composition
issue: null
reviewed_by: null
reviewed_at: null
---

# REQ-0004 · Multi-adapter composition

## Requirement (normative)

The system MUST support composing multiple adapters within a single model. It MUST support at
least 20 adapters composed together.

## Rationale

Many capabilities in one deployable artifact is the core composability goal (traces to **BG-01**).

## Acceptance criteria

- The composer produces one checkpoint composing at least 20 adapters that loads and runs
  inference, shown by a ≥20-adapter compose-and-infer test.

## Notes

- Level **MUST** · Status **MVP · Partial**.
- Composition path is the composer (**ADR-0009** — single construction path); aLoRA shared-KV
  composition is **ADR-0003**.
