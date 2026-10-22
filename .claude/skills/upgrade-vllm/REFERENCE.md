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
| `pyproject.toml` `[project.optional-dependencies]` | `vllm = ["vllm>=0.26.0,<0.33.0"]` | **the supported-range claim** |
| `pyproject.toml` `[dependency-groups]` | `vllm26`, `vllm27`, `vllm30`, `vllm31`, `vllm32` | per-line pins |
| `pyproject.toml` `[dependency-groups]` | `dev`, `dev-vllm27`, `dev-vllm30`, `dev-vllm31`, `dev-vllm32` | what a leg's `dep_group` names |
| `pyproject.toml` `[dependency-groups]` | `dev-latest` | **the upper GPU leg's identity** — repoint per line |
| `pyproject.toml` `[tool.uv] default-groups` | `["vllm26"]` | the implicit version for `uv sync`/CPU CI |
| `pyproject.toml` `[tool.uv] conflicts` | 16 entries at 3 lines | **regenerate, never hand-edit** |
| `pyproject.toml` `dependencies` | `transformers>=5.16,<=5.17.0` | ceiling a new vLLM often forces up |
| `/opt/gsw/derive_legs.py` (runner image) | reads the tested commit's `dev` + `dev-latest` | the ≤2 GPU legs — **not editable from this repo** |
| `.github/workflows/gpu-tests.yaml` | `options: [full, short, dev, audio, multi, eval]` | suite list, **flow style only** |
| `uv.lock` | ~1.8 MB | relock with **uv 0.8.4** |
| `README.md`, `tutorials/PREREQUISITES.md`, `docs/AUDIO.md` | prose version ranges | must match the extra |

**The floor has three names and they move together.** `vllm26` appears as the
`vllm` extra's lower bound, as `dev`'s `include-group` (the lower GPU leg), and as
`default-groups` (bare `uv sync` and CPU CI). Raising the floor without moving all
three is not merely untidy: `default-groups` left below the extra's new lower bound
makes `uv sync` **unsatisfiable**, which is the failure pyproject's own comment
records. `plumb_pyproject.py --floor` moves all three.

`default-groups` is *not* bumped with the **ceiling**, though — only with the
floor. CPU CI runs `uv sync --frozen --group dev ...` and `dev` pins the floor, so
leaving it there keeps the cheap CPU leg where "an API that does not exist yet"
breaks.

## 2. The conflicts table

uv needs an entry for every pair of groups that could pull two different vLLM lines
into one environment. Each declared line has a bare (`vllmNN`) and a dev flavour,
with two asymmetries that defeat a naive `N*(N-1)/2 * 4` formula:

- the **floor**'s dev flavour is bare `dev`, not `dev-vllmNN`;
- the line `dev-latest` points at has a **third** name, so it appears in more pairs.

Today's three lines → **16** committed entries, not 12. Five lines with the alias →
48. Because the alias's membership moves, repointing `dev-latest` regenerates the
table; `--point-latest` does both.

**What uv actually enforces vs. what the file states.** uv *infers* conflicts
through the `include-group` chain: strip the `vllm31`/`dev-latest` pairs and
`uv sync --no-default-groups --group vllm31 --group dev-latest` still refuses, with
"transitively inferred conflicts" (verified, 2026-10-18). So the alias entries are
not load-bearing for resolution — `test` is left implicit for exactly that reason.
They are written out because main's committed table writes them, and `--check`
compares against that form. Do not "optimize" them away: a table that claims
`vllm31` conflicts with `dev-latest` after the alias moved to 0.32 is false
documentation, which is the failure mode that matters here.

`test` is absent on purpose: it reaches the floor through `include-group = "dev"`
and uv derives that transitively.

`plumb_pyproject.py` regenerates this table from the declared groups. Use it.

## 3. The upstream API surface this repo actually touches

Where breakage has landed before, and therefore what `upstream_api_diff.py`
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

A group is not just a declaration — it is a leg that something still asks for:

- **`dev-latest` can be repointed at any declared line**, and during an upgrade it
  is (one commit per line; see §6). A line whose group was deleted cannot be
  evidenced.
- **`derive_legs.py` resolves groups against the *tested commit*.** The group has
  to exist in the tree being checked out, not in the tree you are reading.

A branch that deleted `dev-vllm27` while something still named it produced
`uv sync --group dev-vllm27` → no such group → the leg died before pytest started.
The signature is a **log under 1.1 KB**: "suite never started". It reads like a
test failure and is not one.

`plumb_pyproject.py` never removes a group, including when `--floor` drops a line
from the supported range: only the extra's lower bound moves.

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

## 6. The legs are derived, not written — and how per-line evidence works

**There is no matrix in `gpu-tests.yaml` to edit.** A `legs` job runs
`/opt/gsw/derive_legs.py`, baked into the runner image alongside the other
`/opt/gsw` scripts and **not editable from this repository**. It reads the *tested
commit's own* `pyproject.toml` and emits `matrix.include` entries of
`{label, dep_group}` from exactly two groups:

| leg | group | label |
|---|---|---|
| lower | `dev` | the floor's line, e.g. `vllm26` |
| upper | `dev-latest` | whatever line the alias points at |

So **at most two legs**, and a third cannot be added by any change a PR can make.
The label is the vLLM minor the group's requirements pin, which the GPU harness
asserts the pod really installed — a mislabelled leg fails rather than reporting
the wrong line's result. When both groups pin the same minor it collapses to one
leg. `dep_group` is always a **dev** flavour because the bare `vllmNN` groups omit
pytest, and the floor's dev flavour is plain `dev`, not `dev-vllm26`.

Two consequences worth stating plainly, because the workflow's own comment frames
them as the design:

- Moving to a new vLLM line is a **pyproject edit and nothing else**. The workflow
  file is identical in every repo however far apart their vLLM lines are.
- The PR that makes the bump **tests its own new lines**, because the legs come
  from the commit under test rather than from `main`.

### Per-line evidence: one commit per line, dispatched by sha

Two legs cannot cover four added lines at one commit. They can cover one added line
at each of four commits. The enabling detail is that workflow concurrency is
`group: gpu-tests-${{ inputs.sha }}-${{ inputs.suite }}` — **keyed on the sha**, so
dispatches at different commits run in parallel instead of cancelling each other.
Were it keyed on the branch, each dispatch would kill the previous line's run.

`plumb_pyproject.py --add` parks `dev-latest` on the lowest added line;
`--point-latest LINE` walks it up, one commit each; `--check --final` refuses the
merge state until the alias is back on the ceiling (otherwise CI's upper leg would
never test the top supported line). The floor is re-tested on *every* dispatch
since it is always the lower leg. See SKILL.md Phase 5.

Dispatch needs **both** `--ref` (which workflow file) and `--sha` (which commit);
`--sha` defaults to the branch tip, which would test the last commit N times.

## 7. Authorization

`gpu-tests.yaml`'s first step outside the runner is
`/opt/gsw/check_role.sh "$ACTOR"`, which authorizes only `admin` or `maintain`
(`github-actions[bot]` passes unconditionally, because the comment path already
role-checked upstream). A plain `write` collaborator or a generic PAT bot is
**declined**. Whatever identity `gh` is authenticated as must hold Maintain or
Admin — `preflight.sh` checks this.

## 8. Commit conventions

- `git commit -s` always (DCO is a blocking check).
- **No `Co-authored-by: Claude` trailer.** Removed deliberately on 2026-10-06
  because this work is destined for upstream; AI assistance is disclosed in the PR
  description instead.
- Long, explanatory commit bodies are the house style. Read
  `git log src/granite_switch/vllm/` for the register: what broke, the failure
  signature, which upstream commit caused it, why the fix is shaped the way it is,
  and what was deliberately *not* fixed.
- Verify comments still match the code before committing (CLAUDE.md / GIT_WORKFLOW.md).
