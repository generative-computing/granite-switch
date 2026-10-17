<!-- SPDX-License-Identifier: Apache-2.0 -->
# vLLM upgrade reference

Repo-specific facts the `upgrade-vllm` skill depends on. Verified against the tree
at the time of writing — **re-check line numbers before editing**, they drift.

## 1. The version surface

Every place a vLLM version is declared. There is no runtime version branching
anywhere in `src/` (confirmed by grep for `vllm.__version__`,
`importlib.metadata.version("vllm")`, `packaging.version`) — differences are
absorbed by `hasattr` capability checks at the call site.

| File | What | Role |
|---|---|---|
| `pyproject.toml` `[project.optional-dependencies]` | `vllm = ["vllm>=0.26.0,<0.31.0"]` | **the supported-range claim** |
| `pyproject.toml` `[dependency-groups]` | `vllm26`, `vllm27`, `vllm30` | per-line pins |
| `pyproject.toml` `[dependency-groups]` | `dev`, `dev-vllm27`, `dev-vllm30` | what the GPU matrix's `dep_group` names |
| `pyproject.toml` `[tool.uv] default-groups` | `["vllm26"]` | the implicit version for `uv sync`/CPU CI |
| `pyproject.toml` `[tool.uv] conflicts` | 12 cross-line pairs | **regenerate, never hand-edit** |
| `pyproject.toml` `dependencies` | `transformers>=5.16,<=5.17.0` | ceiling a new vLLM often forces up |
| `.github/workflows/gpu-tests.yaml` | matrix `include:` (`vllm26`/`vllm30`) | the two GPU legs |
| `.github/workflows/gpu-tests.yaml` | `options: [full, short, dev, audio, multi, eval]` | suite list, **flow style only** |
| `uv.lock` | ~1.8 MB | relock with **uv 0.8.4** |
| `README.md`, `tutorials/PREREQUISITES.md`, `docs/AUDIO.md` | prose version ranges | must match the extra |

`default-groups = ["vllm26"]` is deliberately *not* bumped with the ceiling: CPU CI
runs `uv sync --frozen --group dev ...` and `dev` pins the floor. Leaving it alone
keeps the cheap CPU leg on the floor, where "an API that does not exist yet" breaks.

## 2. The conflicts table

uv requires an explicit entry for every pair of groups that could pull two
different vLLM lines into one environment. With `N` declared lines, each having a
bare (`vllmNN`) and a dev (`dev-vllmNN`) flavour — and noting `dev` *is* the floor's
dev flavour — that is `N*(N-1)/2 * 4` entries. Three lines → 12. Four → 24.

`test` is absent on purpose: it reaches the floor through `include-group = "dev"`
and uv derives that transitively.

`plumb_pyproject.py` regenerates this table from the declared groups. Use it.

## 3. The upstream API surface this repo actually touches

Where breakage has landed before, and therefore what `upstream_api_diff.sh`
watches most closely:

- **Multimodal processor hooks** (`src/granite_switch/vllm/audio/processor.py`).
  0.29 deleted `BaseMultiModalProcessor._call_hf_processor` and its only caller
  `_apply_hf_processor_text_mm`, and narrowed `_apply_hf_processor_main` to
  `(mm_items, hf_processor_mm_kwargs) -> BatchFeature`. The ASR override lived on
  the deleted hook, so on 0.29+ it was **never called and produced no transcript** —
  silently broken, with every test still green. This is the canonical worst case.
- **MoE topk kernels' `is_padding` mask** (`vllm/decoder/shadow_residual/decoder.py`).
  0.27 started passing a mask read off the forward context, sized to the padded
  token count. The SR layer routes a `[2M, H]` base+adapter stack in one expert
  call, so the kernel saw `2M` gating rows against an `M`-entry mask and the engine
  died at init with `is_padding size mismatch`.
- **`AttentionBackend.get_kv_cache_shape`** (tests). Gone at 0.29 — 29 definitions
  at 0.28, zero at 0.29. `tests/shared/vllm_kv_cache.py` prefers
  `compute_layer_kv_cache_shape_bytes` and falls back, cross-checking where both
  exist.
- **`vllm.Attention`** — the switch tier's paged-KV heads
  (`src/granite_switch/vllm/switch/`).
- **`LLM`/`EngineArgs` kwargs** — `tests/shared/vllm_equivalence.py`'s
  `run_vllm_logprobs` is the single funnel for every engine start in the test
  suite, so a renamed engine kwarg shows up there once rather than in five runners.
- **The plugin entry point** `vllm.general_plugins` →
  `granite_switch.vllm:register`.

## 4. Why a dependency group must never be deleted

`workflow_dispatch` executed through the `/gpu-test` comment path always runs the
workflow file from the **default branch** (`gpu_test_command.sh` hardcodes
`--ref <default branch>`) while checking out the PR's commit. So main's matrix
decides which `dep_group` values get synced against your tree.

A branch that deleted `dev-vllm27` while main's matrix still named it produced
`uv sync --group dev-vllm27` → no such group → the leg died before pytest started.
The signature is a **log under 1.1 KB**: "suite never started".

The skill sidesteps the stale-matrix problem by dispatching with
`--ref <pr-branch>` directly. Keep the groups anyway — the comment path is still
what humans use on the PR.

## 5. CI topology

Automatic on every PR to `main`:

| Check | Where | Notes |
|---|---|---|
| `CI / Pre-commit` | `ubuntu-latest` | `needs`-gates the CPU tests. Fails if `uv.lock` is stale under uv 0.8.4. |
| `CI / CPU Tests (Python 3.11)` and `(3.12)` | `ubuntu-latest` | `tests/unit/` only, `-m "not requires_model and not gpu and not slow and not deep"`. `fail-fast: false`. |
| `DCO Check / DCO Sign-off` | `ubuntu-latest` | every non-merge commit needs `Signed-off-by:`. |

`tests/composer/` and `tests/hf/` are **deliberately excluded** from CPU CI — even
filtered they exhaust the GitHub Actions CPU-minutes quota and get killed mid-run.
They run on the GPU cluster.

Manual, GPU: `gpu-tests.yaml`, `workflow_dispatch` only, `runs-on: [self-hosted, gpu]`,
`timeout-minutes: 420`, two version legs with `fail-fast: false`, concurrency keyed
on `sha + suite`. There is **no `actions/checkout`** — every step calls a script
baked into the runner image under `/opt/gsw`, and the in-pod `uv sync` happens
inside `submit_and_poll.sh`, which only receives `--dep-group`. Those scripts are
**not editable from this repo**, so anything a new suite needs must be expressible
as a pytest path or marker passed through `--tests`.

Suite → pytest scope (the `Resolve test scope` case arms):

| suite | `TEST_PATHS` | GPUs |
|---|---|---|
| `full` | `""` → runner-image default, all five suites | 1 |
| `short` | `tests/vllm/ tests/integration/` | 1 |
| `dev` | `tests/vllm/test_single_switch.py` | 1 |
| `audio` | `-m "audio and not deep" tests/` | 1 |
| `multi` | `""` (all five) | 2 |
| `eval` | `-m "eval and not deep" tests/eval/` | 1 |

Adding a suite means editing the `options:` line **and** adding a case arm, and
nothing else. `#` is the one forbidden character in `TEST_PATHS` (it is the `sed`
delimiter in the job template). A bare `-m <marker>` **replaces** `pyproject.toml`'s
`addopts = ["-m", "not deep"]` rather than adding to it — always write
`and not deep`.

### Known-stale things not to trust

- `tests/vllm/test_single_switch.py` — the `dev` suite's target — **does not exist**.
  `/gpu-test-dev` is broken today. Don't rely on it as a smoke check.
- `tests/regression/` does not exist, so the Makefile's `test-gpu-full`,
  `test-regression-*` and `test-peft-equiv*` targets are dead.
- `docs/CICD.md` predates the current GPU setup: it never mentions `/gpu-test`,
  claims the runner is "scaffolded but inert", advertises `make test-gpu-full`, and
  describes coverage failure as non-fatal via `fail_ci_if_error` alone (the
  `continue-on-error: true` that actually enforces it is missing). **The workflow
  files' inline comments are the spec.**

## 6. Authorization

`gpu-tests.yaml`'s first step outside the runner is
`/opt/gsw/check_role.sh "$ACTOR"`, which authorizes only `admin` or `maintain`
(`github-actions[bot]` passes unconditionally, because the comment path already
role-checked upstream). A plain `write` collaborator or a generic PAT bot is
**declined**. Whatever identity `gh` is authenticated as must hold Maintain or
Admin — `preflight.sh` checks this.

## 7. Commit conventions

- `git commit -s` always (DCO is a blocking check).
- **No `Co-authored-by: Claude` trailer.** Removed deliberately on 2026-10-06
  because this work is destined for upstream; AI assistance is disclosed in the PR
  description instead.
- Long, explanatory commit bodies are the house style. Read
  `git log src/granite_switch/vllm/` for the register: what broke, the failure
  signature, which upstream commit caused it, why the fix is shaped the way it is,
  and what was deliberately *not* fixed.
- Verify comments still match the code before committing (CLAUDE.md / GIT_WORKFLOW.md).
