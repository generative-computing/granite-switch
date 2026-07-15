---
id: REQ-0017
title: Support Granite dense models as the base, auto-detected
status: draft
version: 1
supersedes: []
superseded_by: null
governs_paths:
  - "src/granite_switch/composer/arch.py"
  - "src/granite_switch/composer/adapter_discovery.py"
aspect: model-support
issue: null
reviewed_by: null
reviewed_at: null
---

# REQ-0017 · Supported base model family

## Requirement (normative)

The system MUST support Granite dense models as the base, detected automatically from the
HuggingFace `config.model_type` (`granite`). The architecture SHOULD NOT hard-block support for
additional base families in the future.

## Rationale

A bounded, well-tested MVP family maximizes correctness now while keeping the door open for reach
later (traces to **BG-06**).

## Acceptance criteria

- Any Granite dense model (`model_type: granite`) composes and runs, auto-detected from its HF
  config, shown by a Granite-base compose test.

## Notes

- Level **MUST** · Status **MVP (Granite 4.x dense); other families Post-MVP · Satisfied**.
- Family scope + single-GPU scope decided in **ADR-0008**.
