---
id: REQ-0005
title: Adapters of differing sizes are composable together
status: draft
version: 1
supersedes: []
superseded_by: null
governs_paths:
  - "src/granite_switch/composer/adapter_loader.py"
  - "src/granite_switch/composer/weight_remapper.py"
  - "src/granite_switch/hf/core/lora.py"
  - "src/granite_switch/vllm/core/lora.py"
aspect: composition
issue: null
reviewed_by: null
reviewed_at: null
---

# REQ-0005 · Heterogeneous adapter sizes

## Requirement (normative)

Adapters of differing sizes (e.g. different LoRA ranks) MUST be composable together. Composition
MUST NOT require all adapters to share the same rank.

## Rationale

Independently-developed adapters arrive at different ranks; forcing a shared rank would break
independent development (traces to **BG-01**).

## Acceptance criteria

- Adapters with different LoRA ranks compose into one checkpoint and run correctly, shown by a
  mixed-rank composition test.

## Notes

- Level **MUST** · Status **MVP · Satisfied**.
- Enabled by the shared-KV aLoRA composition (**ADR-0003**).
