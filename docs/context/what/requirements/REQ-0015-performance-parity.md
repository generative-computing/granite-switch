---
id: REQ-0015
title: Serving performance must be competitive with native multi-LoRA
status: draft
version: 1
supersedes: []
superseded_by: null
governs_paths:
  - "src/granite_switch/vllm/**"
aspect: performance
issue: null
reviewed_by: null
reviewed_at: null
---

# REQ-0015 · Performance parity with native LoRA

## Requirement (normative)

Serving performance MUST be competitive with vLLM's built-in multi-LoRA implementation.
Specifically, in the single-active-adapter configuration and under matched conditions (same base
model, batch size, sequence length, and hardware), decode throughput (tokens/sec) and per-token
latency (TPOT / ITL) SHOULD be within an agreed tolerance (e.g. 10–15%) of the native LoRA path.
The additional cost of per-token routing and adapter switching (**REQ-0001**, **REQ-0003**) is
budgeted separately, measured against the single-adapter baseline, and MUST remain bounded.

## Rationale

Matching native LoRA cost is what lets many small capabilities run at the cost of one (traces to
**BG-03**).

## Acceptance criteria

- Under matched conditions, single-adapter decode throughput and TPOT/ITL are within the agreed
  tolerance of the native multi-LoRA path, and switching overhead is within its defined budget,
  verified by the parity benchmark.

## Notes

- Level **MUST** (parity target) · Status **Partial**. Tolerance and switching-overhead budget
  still **TBD**.
- Token-exchange removed the KV-cache tax that undercut this (**ADR-0005**).
