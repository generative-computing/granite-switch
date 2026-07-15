---
id: REQ-0001
title: Adapter selection is controlled per-token via a control token
status: draft
version: 1
supersedes: []
superseded_by: null
governs_paths:
  - "src/granite_switch/hf/switch/**"
  - "src/granite_switch/vllm/switch/**"
  - "src/granite_switch/composer/tokenizer_setup.py"
aspect: switching-mechanism
issue: null
reviewed_by: null
reviewed_at: null
---

# REQ-0001 · Control-token per-token routing

## Requirement (normative)

Adapter selection MUST be controlled via a control token, enabling per-token routing so the
active adapter can change between tokens within a single sequence.

## Rationale

Explicit, inspectable, per-token control is the foundation of composing capabilities into one
model (traces to **BG-02** — build your own model from capabilities you control).

## Acceptance criteria

- A control token placed at position `i` changes the active adapter for all tokens from `i`
  onward within a single sequence, shown by a per-token routing test.

## Notes

- Level **MUST** · Status **MVP · Satisfied**.
- Implemented by **ADR-0004** (control-token activation) and refined by **ADR-0005**
  (token-exchange). Optional implicit selection is **REQ-0022**.
- Open question (from source): what triggers a switch — is switching always driven by the
  control-token mechanism, or can an adapter emit a switch at inference time independent of input
  tokens? See **REQ-0003**.
