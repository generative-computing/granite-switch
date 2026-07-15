---
id: REQ-0023
title: Adapter functions are independently developable and benchmarkable
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

# REQ-0023 · Independent development and benchmarking

## Requirement (normative)

Adapter functions MUST be independently developable and benchmarkable: an author MUST be able to
train and evaluate a single adapter function without jointly training it against the other adapters
it will later compose with, and composition MUST NOT require re-training previously composed
adapters.

## Rationale

Independent development at scale — and matching bigger generalists with small targeted pieces —
depends on not having to co-train (traces to **BG-03**).

## Acceptance criteria

- A single adapter function can be trained and benchmarked standalone and later composed without
  retraining the others, shown by an independent-adapter workflow.

## Notes

- Level **MUST** · Status **MVP · Satisfied**.
- Enabled by the shared normalized KV contract (**ADR-0003**).
