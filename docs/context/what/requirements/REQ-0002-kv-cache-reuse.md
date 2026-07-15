---
id: REQ-0002
title: Switching the active adapter must not invalidate the KV cache
status: draft
version: 1
supersedes: []
superseded_by: null
governs_paths:
  - "src/granite_switch/hf/switch/**"
  - "src/granite_switch/vllm/switch/**"
  - "src/granite_switch/hf/core/lora.py"
  - "src/granite_switch/vllm/core/lora.py"
aspect: performance
issue: null
reviewed_by: null
reviewed_at: null
---

# REQ-0002 · KV-cache reuse across switches

## Requirement (normative)

Switching the active adapter for subsequent tokens MUST NOT invalidate or force recomputation of
the KV cache for tokens already processed. Cache built under one adapter remains valid and
reusable when control passes to another.

## Rationale

KV recompute on every switch is the dominant latency cost; reusing it is what makes composing many
capabilities cheap (traces to **BG-02**).

## Acceptance criteria

- Switching the active adapter mid-sequence reuses the KV cache for already-processed tokens (no
  recomputation), shown by a prefill-reuse / no-recompute test.

## Notes

- Level **MUST** · Status **MVP · Satisfied**.
- Achieved via aLoRA on a shared normalized KV cache (**ADR-0003**) and token-exchange
  (**ADR-0005**). The *mechanics* of reuse are design detail and live in those ADRs, not here.
