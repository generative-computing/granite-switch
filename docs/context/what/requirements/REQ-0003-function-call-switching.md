---
id: REQ-0003
title: Adapter invocation follows call/return semantics
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

# REQ-0003 · Function-call-style dynamic switching

## Requirement (normative)

Adapter invocation MUST follow call/return semantics: both the base model and an active adapter
can transfer control to another adapter, and control returns to the caller once the invoked
adapter completes — analogous to a function call stack.

## Rationale

Call/return composition is the "capabilities compose like software" premise (traces to **BG-01**,
**BG-05**).

## Acceptance criteria

- An adapter (or the base) can invoke another adapter and control returns to the caller on
  completion, shown by a nested call/return test.

## Notes

- Level **SHOULD** · Status **Post-MVP · Not satisfied**.
- Open questions (from source): (1) is switching always control-token-driven (**REQ-0001**) or can
  an adapter emit a switch at inference time? (2) nested/re-entrant (a true call stack) vs. a flat
  hand-off with no return? The "function call" analogy implies a stack; confirm before finalizing.
