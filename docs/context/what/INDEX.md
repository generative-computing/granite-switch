# Requirements Index (WHAT)

> **CollaborativeDream layer: WHAT** — the versioned, definitive source of truth for implementation.
> This INDEX is a **human registry** for navigation. It is **NOT auto-loaded** into any
> agent context — agents reach requirement *bodies* either automatically (when they read a
> file matching a requirement's `governs_paths`) or on demand via `/cd-context <subsystem>`.
>
> **Rule:** list every requirement here — active *and* retired — so humans can trace the
> chain. Retired requirement *files* are deleted from HEAD (git keeps the history); only
> their row here (with a git ref) remains.

---

## Active requirements

All requirements are `status: draft` and **not authoritative until a human sets `reviewed_by`**.
`src/granite_switch/` is abbreviated `…/` in the Governs-paths column.

| ID | Title | Version | Aspect | Governs paths | Reviewed |
|---|---|---|---|---|---|
| [REQ-0001](requirements/REQ-0001-control-token-routing.md) | Control-token per-token routing | 1 | switching-mechanism | `…/hf/switch/**`, `…/vllm/switch/**`, `…/composer/tokenizer_setup.py` | — |
| [REQ-0002](requirements/REQ-0002-kv-cache-reuse.md) | KV-cache reuse across switches | 1 | performance | `…/hf/switch/**`, `…/vllm/switch/**`, `…/hf/core/lora.py`, `…/vllm/core/lora.py` | — |
| [REQ-0003](requirements/REQ-0003-function-call-switching.md) | Function-call-style dynamic switching | 1 | switching-mechanism | `…/hf/switch/**`, `…/vllm/switch/**` | — |
| [REQ-0004](requirements/REQ-0004-multi-adapter-composition.md) | Compose ≥20 adapters | 1 | composition | `…/composer/**` | — |
| [REQ-0005](requirements/REQ-0005-heterogeneous-adapter-sizes.md) | Heterogeneous adapter sizes | 1 | composition | `…/composer/adapter_loader.py`, `…/composer/weight_remapper.py`, `…/hf/core/lora.py`, `…/vllm/core/lora.py` | — |
| [REQ-0006](requirements/REQ-0006-mixed-adapter-types.md) | LoRA + aLoRA together | 1 | composition | `…/composer/adapter_discovery.py`, `…/composer/tokenizer_setup.py` | — |
| [REQ-0007](requirements/REQ-0007-per-adapter-target-modules.md) | Per-adapter target modules | 1 | composition | `…/composer/weight_remapper.py`, `…/composer/adapter_loader.py` | — |
| [REQ-0008](requirements/REQ-0008-cuda-friendly.md) | CUDA-friendly implementation | 1 | performance | `…/vllm/**` | — |
| [REQ-0009](requirements/REQ-0009-vllm-compatibility.md) | vLLM compatibility | 1 | deployment-target | `…/vllm/**` | — |
| [REQ-0010](requirements/REQ-0010-multiple-serving-backends.md) | Multiple serving backends | 1 | deployment-target | `…/hf/**`, `…/vllm/**` | — |
| [REQ-0011](requirements/REQ-0011-bring-your-own-adapter.md) | Bring your own adapter | 1 | composition | `…/composer/adapter_discovery.py`, `…/composer/compose_granite_switch.py` | — |
| [REQ-0012](requirements/REQ-0012-bring-your-own-data.md) | Bring your own data | 1 | composition | `…/composer/**` | — |
| [REQ-0013](requirements/REQ-0013-declared-dependency-versions.md) | Declared dependency versions | 1 | dependency-management | `pyproject.toml`, `uv.lock` | — |
| [REQ-0014](requirements/REQ-0014-latest-official-versions.md) | Latest official versions | 1 | dependency-management | `pyproject.toml`, `uv.lock` | — |
| [REQ-0015](requirements/REQ-0015-performance-parity.md) | Performance parity with native LoRA | 1 | performance | `…/vllm/**` | — |
| [REQ-0016](requirements/REQ-0016-community-standard-format.md) | Community-standard model format | 1 | deployment-target | `…/composer/**`, `…/hf/__init__.py`, `…/vllm/__init__.py` | — |
| [REQ-0017](requirements/REQ-0017-supported-base-family.md) | Supported base model family (Granite) | 1 | model-support | `…/composer/arch.py`, `…/composer/adapter_discovery.py` | — |
| [REQ-0019](requirements/REQ-0019-single-unified-checkpoint.md) | Single unified checkpoint across backends | 1 | architecture | `…/composer/**`, `…/hf/modeling_granite_switch.py`, `…/vllm/granite_switch_model.py` | — |
| [REQ-0020](requirements/REQ-0020-typed-output-contract.md) | Typed output-contract (Mellea) | 1 | data-contract | `…/__init__.py`, `pyproject.toml` | — |
| [REQ-0022](requirements/REQ-0022-optional-trainable-router.md) | Optional trainable router | 1 | switching-mechanism | `…/hf/switch/**`, `…/vllm/switch/**` | — |
| [REQ-0023](requirements/REQ-0023-independent-dev-benchmarking.md) | Independent development & benchmarking | 1 | composition | `…/composer/**` | — |
| [REQ-0024](requirements/REQ-0024-composition-correctness.md) | Composition correctness (skinning) | 1 | build-pipeline | `…/composer/validator.py`, `…/composer/weight_transfer.py`, `…/composer/compose_utils.py` | — |
| [REQ-0025](requirements/REQ-0025-open-source-license.md) | Open-source license & open development | 1 | licensing | `pyproject.toml`, `README.md` | — |
| [REQ-0026](requirements/REQ-0026-ecosystem-coherence.md) | Ecosystem coherence (Granite + Mellea) | 1 | ecosystem | `…/__init__.py`, `…/composer/**` | — |

> **Numbering:** REQ-0018 and REQ-0021 are intentionally absent (not defined in the source spec).

## Retired requirements

| ID | Title | Superseded by | Retired at (git) |
|---|---|---|---|
| _(none yet)_ | | | |

---

## How this layer works (quick reference)

- **One file per requirement**, `requirements/REQ-<id>-<slug>.md`, with full frontmatter.
- **`governs_paths`** binds a requirement to the code it governs → the requirement enters an
  agent's context automatically when that code is read (native path-scoped rules do the same;
  see `.claude/rules/`). Keep globs tight.
- **Versioning** lives in **git**, not in duplicate files. Bump `version:` on a material change.
- **Retiring:** set `status: superseded` + `superseded_by:`, commit, then **delete the file**.
  A retired requirement must never keep live `governs_paths` (that would leak stale truth into
  future context).
- **Other WHAT objects** (Issue / Alternatives / Gap / Curated-plan / technical-task lists) live
  in sibling folders under `docs/context/what/` as the project grows; they are read only via
  `/cd-context`, never auto-loaded.

*CollaborativeDream · WHAT layer.*
