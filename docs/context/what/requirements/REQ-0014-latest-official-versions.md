---
id: REQ-0014
title: The implementation should track the latest official dependency versions
status: draft
version: 1
supersedes: []
superseded_by: null
governs_paths:
  - "pyproject.toml"
  - "uv.lock"
aspect: dependency-management
issue: null
reviewed_by: null
reviewed_at: null
---

# REQ-0014 · Latest official versions

## Requirement (normative)

The implementation SHOULD support the latest official released versions of its key dependencies and
target backends, and SHOULD be kept current as those release.

## Rationale

Staying current with the ecosystem keeps the project deployable where the community runs (traces to
**BG-06**).

## Acceptance criteria

- The supported versions of key dependencies and backends track their latest official releases
  within the stated compatibility constraints.

## Notes

- Level **SHOULD** · Status **Partial**.
- Deliberate tension recorded in **ADR-0011**: default vLLM is 0.19.1 (not the newest) to preserve
  CUDA 12.x compatibility; 0.20 is opt-in. transformers widened to `<5.10` (**ADR-0012**); Mellea
  exact pin (**ADR-0013**). `uv` lockfile keeps updates reproducible (**ADR-0010**).
