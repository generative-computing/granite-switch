---
id: REQ-0020
title: Each adapter function exposes a typed contract, enforced via Mellea
status: draft
version: 1
supersedes: []
superseded_by: null
governs_paths:
  - "src/granite_switch/__init__.py"
  - "pyproject.toml"
aspect: data-contract
issue: null
reviewed_by: null
reviewed_at: null
---

# REQ-0020 · Typed output-contract enforcement

## Requirement (normative)

Each adapter function MUST expose a defined input/output contract (e.g. a score, a decision, a
rewritten query). Token-level enforcement of that contract is delegated entirely to Mellea; the
composed model MUST be compatible with Mellea's constrained decoding rather than implementing
schema enforcement itself.

## Rationale

A typed contract per adapter function is what makes "capabilities compose like software" legible
and safe (traces to **BG-05**).

## Acceptance criteria

- Each adapter function exposes a typed I/O contract and Mellea enforces schema-valid output at the
  token level, shown by a Mellea constrained-decoding test.

## Notes

- Level **MUST** · Status **MVP · Satisfied**.
- Product framing / "adapter function" terminology (**ADR-0015**); Mellea pin + Python floor
  (**ADR-0013**).
