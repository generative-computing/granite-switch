---
id: REQ-0011
title: Users can supply their own pre-trained adapter to the composer
status: draft
version: 1
supersedes: []
superseded_by: null
governs_paths:
  - "src/granite_switch/composer/adapter_discovery.py"
  - "src/granite_switch/composer/compose_granite_switch.py"
aspect: composition
issue: null
reviewed_by: null
reviewed_at: null
---

# REQ-0011 · Bring your own adapter

## Requirement (normative)

Users MUST be able to supply their own adapter to the composer and have it composed alongside the
built-in adapters. This is the first-version scope of "bring your own": the user provides a
pre-trained adapter.

## Rationale

An open ecosystem where anyone can contribute a capability (traces to **BG-07**, and **BG-01**).

## Acceptance criteria

- A user-supplied pre-trained adapter composes alongside the built-in adapters and activates
  correctly, shown by a bring-your-own-adapter compose test.

## Notes

- Level **MUST** · Status **MVP · Satisfied**.
- The next-version scope (supply data, not a finished adapter) is **REQ-0012**.
