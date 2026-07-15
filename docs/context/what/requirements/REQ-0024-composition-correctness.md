---
id: REQ-0024
title: Zero-adapter composition equals the base model; composer validates before emit
status: draft
version: 1
supersedes: []
superseded_by: null
governs_paths:
  - "src/granite_switch/composer/validator.py"
  - "src/granite_switch/composer/weight_transfer.py"
  - "src/granite_switch/composer/compose_utils.py"
aspect: build-pipeline
issue: null
reviewed_by: null
reviewed_at: null
---

# REQ-0024 · Composition correctness (skinning equivalence)

## Requirement (normative)

Composing a base model with zero active adapters ("skinning") MUST be numerically equivalent to the
original base model within the serving backend's floating-point tolerance, and the composer MUST
validate a composed checkpoint before emitting it.

## Rationale

If skinning isn't equivalent to the base, nothing composed on top can be trusted; validation before
emit protects every downstream capability (traces to **BG-01**).

## Acceptance criteria

- A zero-adapter ("skinned") composed model matches the base model within backend tolerance (the
  **vLLM** skinning-equivalence tests pass) and the composer validates a checkpoint before emitting
  it.

## Notes

- Level **MUST** · Status **MVP · Satisfied**.
- "Within backend tolerance", not "bit-exact": the HF fused-projection path is not bit-exact with
  upstream HF, so the vLLM tests are authoritative (**ADR-0006**). All construction goes through the
  composer (**ADR-0009**).
