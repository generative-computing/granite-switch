---
id: REQ-0012
title: Users can supply their own data, from which an adapter is produced
status: draft
version: 1
supersedes: []
superseded_by: null
governs_paths:
  - "src/granite_switch/composer/**"
aspect: composition
issue: null
reviewed_by: null
reviewed_at: null
---

# REQ-0012 · Bring your own data

## Requirement (normative)

In a later version, users SHOULD be able to supply their own data, from which an adapter is
produced/adapted for composition (rather than supplying a finished adapter as in **REQ-0011**).

## Rationale

Lowering the barrier from "train an adapter" to "bring data" widens the open ecosystem (traces to
**BG-07**, and **BG-01**).

## Acceptance criteria

- A user supplies data and the toolchain produces a composable adapter from it (no finished adapter
  required).

## Notes

- Level **SHOULD** · Status **Post-MVP · Not satisfied**.
- No code path exists yet; `governs_paths` is scoped to the conventional `composer/**` location and
  should be re-pointed when the data-ingestion path lands.
