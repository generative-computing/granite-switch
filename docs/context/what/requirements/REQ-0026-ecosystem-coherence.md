---
id: REQ-0026
title: Remain coherent with Granite bases, adapter libraries, and Mellea
status: draft
version: 1
supersedes: []
superseded_by: null
governs_paths:
  - "src/granite_switch/__init__.py"
  - "src/granite_switch/composer/**"
aspect: ecosystem
issue: null
reviewed_by: null
reviewed_at: null
---

# REQ-0026 · Ecosystem coherence

## Requirement (normative)

The system MUST remain coherent and interoperable with the Granite base models, the Granite
adapter-function libraries, and Mellea. (The upstream-acceptance posture is covered by **REQ-0016**.)

## Rationale

Coherence across the Granite + Mellea ecosystem is what makes the composed model usable end-to-end
without bespoke glue (traces to **BG-06**).

## Acceptance criteria

- A model composed from a Granite base + Granite adapter libraries runs end-to-end through Mellea
  without bespoke glue, shown by an integration test.

## Notes

- Level **MUST** · Status **MVP · Satisfied**.
- Product framing and "adapter function" terminology (**ADR-0015**).
