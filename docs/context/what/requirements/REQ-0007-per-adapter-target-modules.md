---
id: REQ-0007
title: Adapters may target different module sets independently
status: draft
version: 1
supersedes: []
superseded_by: null
governs_paths:
  - "src/granite_switch/composer/weight_remapper.py"
  - "src/granite_switch/composer/adapter_loader.py"
aspect: composition
issue: null
reviewed_by: null
reviewed_at: null
---

# REQ-0007 · Per-adapter target modules

## Requirement (normative)

Different adapters MAY target different module sets. Each adapter defines its own target modules
independently, and composition MUST correctly handle adapters whose target modules only partially
overlap or do not overlap at all.

## Rationale

Independent adapters make independent target-module choices; composition must not assume a shared
target set (traces to **BG-01**).

## Acceptance criteria

- Adapters whose target modules partially overlap or do not overlap compose and run correctly,
  shown by a partial-/non-overlapping target-module test.

## Notes

- Level **MUST** · Status **MVP · Satisfied**.
- Per-adapter targets are handled over the fused projections — see **ADR-0006**.
