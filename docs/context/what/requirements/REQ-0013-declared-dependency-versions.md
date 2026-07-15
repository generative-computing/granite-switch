---
id: REQ-0013
title: All dependency versions are declared in the TOML manifest
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

# REQ-0013 · Declared dependency versions

## Requirement (normative)

All package dependencies and their supported version ranges MUST be declared in the project's TOML
manifest (e.g. `pyproject.toml`). The implementation MUST build and run against every version
permitted by that manifest.

## Rationale

Reproducible, declared installs are a prerequisite for deploying where the community runs (traces
to **BG-06**).

## Acceptance criteria

- Every dependency and its supported range is declared in `pyproject.toml` with a committed
  lockfile, and CI builds and passes across the permitted range.

## Notes

- Level **MUST** · Status **MVP · Satisfied**.
- Single package with optional extras (**ADR-0002**); `uv` + committed lockfile (**ADR-0010**);
  transformers 5.x range (**ADR-0012**); Mellea pin / Python floor (**ADR-0013**).
