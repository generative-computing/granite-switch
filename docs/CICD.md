# CI/CD

This document is the reference for the CI/CD setup for granite-switch — what runs locally, what runs on every PR, and the roadmap for future automation.

## Local: Pre-commit Hooks

Pre-commit hooks enforce code quality before a commit lands. They run automatically on `git commit` after a one-time setup.

### Setup

pre-commit is a hook manager — it builds its own isolated environment per hook, so it does **not**
need the project's virtualenv. Install it once as a standalone tool, then wire up the git hooks:

```bash
uv tool install pre-commit                          # one-time, isolated — any platform
pre-commit install                                  # hooks for staged files
pre-commit install --hook-type commit-msg           # verifies the DCO sign-off
pre-commit install --hook-type prepare-commit-msg   # auto-adds your DCO sign-off
```

> Do **not** run `uv run pre-commit install`. `uv run` first syncs the project's default
> dependency group (which includes vLLM + CUDA wheels); that has no macOS build and fails there,
> and needlessly pulls the whole GPU stack on Linux. Installing pre-commit as a tool (`uv tool
> install`, or `pipx`/system package) sidesteps this on every platform. All three `install`
> commands are required — each git hook stage must be wired up separately.

### What runs on every commit

| Hook | What it does | Auto-fix? |
|------|-------------|-----------|
| `check-toml` / `check-yaml` | Validates config file syntax | No — blocks commit |
| `check-merge-conflict` | Blocks commits containing conflict markers (`--assume-in-merge`) | No — blocks commit |
| `check-added-large-files` | Blocks files over 500 KB (`--maxkb=500 --enforce-all`; `uv.lock` excluded) | No — blocks commit |
| `check-case-conflict` | Blocks names that collide on case-insensitive filesystems | No — blocks commit |
| `mixed-line-ending` | Normalizes line endings to LF (`--fix=lf`) | Yes |
| `end-of-file-fixer` / `trailing-whitespace` | Hygiene | Yes |
| `ruff-format` | Formats Python files | Yes (reformats, then blocks until staged) |
| `ruff` | Lints Python files (imports, style) | Yes (fixable issues; then blocks until staged) |
| `check-headers` | Ensures every `.py` file starts with `# SPDX-License-Identifier: Apache-2.0` | Yes (regenerates, then blocks until staged) |
| `add-signoff` | Appends `Signed-off-by:` at the `prepare-commit-msg` stage | Yes (adds the trailer for you) |
| `check-dco` | Validates commit message has `Signed-off-by: Name <email>` | No — blocks commit |
| `validate-links` | Checks broken local links, stale labels, and broken first-party imports in `.ipynb`/`.md`/`.py` | No — blocks commit |
| `nbstripout` | Strips output cells from Jupyter notebooks | Yes |
| `uv-lock` | Regenerates `uv.lock` when out of sync with `pyproject.toml` | Yes (regenerates, then blocks until staged) |

### DCO sign-off

Every commit must be signed off to certify you agree to the [Developer Certificate of Origin](https://developercertificate.org/). Add it automatically with:

```bash
git commit -s -m "Your commit message"
```

To sign off all commits in an existing branch:

```bash
git rebase --signoff origin/main
```

### Running hooks manually

```bash
# Run all hooks on all files — this is exactly what the CI `pre-commit` job runs
pre-commit run --all-files

# Run a specific hook
pre-commit run ruff --all-files
pre-commit run check-headers --all-files
```

---

## On Every PR: GitHub Actions

Two workflows run automatically when a PR is opened against `main` or a commit is pushed.

### `ci.yaml` — Pre-commit + CPU tests + coverage

**Trigger:** Pull request or push to `main`

**Jobs:**
1. `pre-commit` — runs `uvx pre-commit run --all-files`, executing the **exact same hooks** as the local pre-commit stage, pinned to the same versions in `.pre-commit-config.yaml`. This is the single source of truth: CI and local pre-commit can never drift, and any hook added or bumped in the config applies in CI automatically with no workflow edits. It covers ruff-format + ruff, `check-toml` / `check-yaml`, `validate-links` (broken local links / stale labels / broken first-party imports), `check-headers` (SPDX), the hygiene hooks (`end-of-file-fixer`, `trailing-whitespace`, `mixed-line-ending`, `check-merge-conflict`, `check-added-large-files`, `check-case-conflict`), `nbstripout`, and `uv-lock`. All are **blocking**: the tree is clean (see [Ruff rollout](#ruff-rollout-format-first-then-enforce)), so any new violation fails the build. Auto-fixing hooks (ruff, `end-of-file-fixer`, `check-headers`, `uv-lock`) modify files when something is wrong; pre-commit then reports failure and prints the diff (`--show-diff-on-failure`), telling the contributor to run pre-commit locally and re-commit. The DCO hooks (`add-signoff`, `check-dco`) run at the `prepare-commit-msg` / `commit-msg` stages, which `pre-commit run --all-files` does **not** execute — they are covered by `dco.yaml` instead. Hook environments are cached (`~/.cache/pre-commit`, keyed on the config file) so they don't reinstall every run.
2. `test-cpu` (needs `pre-commit`) — runs `tests/unit/` on Python 3.11 and 3.12 in parallel, filtered with `-m "not requires_model and not gpu and not slow and not deep"`. This is the **CI-safe set**: within `tests/unit/`, tests that download/compose real models (`requires_model`), need CUDA (`gpu`), hit the network or run long (`slow`), or are the expensive code-theory suite (`deep`) are still **excluded** by the marker filter. `tests/composer/` and `tests/hf/` are **not run in CI at all**: even filtered, they are heavy enough to exhaust the GitHub Actions CPU-minutes quota and get killed mid-run. All coverage beyond the unit suite — composer, HF, vLLM, and integration — runs on the GPU cluster instead (see [Manual: GPU Tests](#manual-gpu-tests))

Coverage is uploaded to [Codecov](https://codecov.io) after each test run (see [Coverage](#coverage) below).

> **SPDX headers** are checked by the `check-headers` hook inside the `pre-commit` job above —
> there is no separate headers workflow, so CI enforces them identically to local.

#### Whole-tree enforcement for `check-added-large-files` and `check-merge-conflict`

Two hygiene hooks act only on git state by default — `check-added-large-files` inspects
just *staged additions*, and `check-merge-conflict` scans only while a merge is in progress.
On CI's clean checkout nothing is staged and no merge is underway, so out of the box both
would pass without examining anything. The configuration therefore runs them across the full
tree so CI enforces them for real:

- **`check-added-large-files`** is configured with `--enforce-all` and `--maxkb=500`, so every
  tracked file is size-checked under `pre-commit run --all-files`, not only newly staged ones.
  `uv.lock` (~2.3 MB) is intentionally large and is excluded via `exclude: ^uv\.lock$`; it is the
  only file over the limit.
- **`check-merge-conflict`** is configured with `--assume-in-merge`, so it always scans for
  conflict markers instead of returning early when no merge is active. This lets CI catch markers
  that were committed (for example via `--no-verify`). Note that with this setting any line that is
  exactly `=======` or begins with `<<<<<<< ` / `======= ` / `>>>>>>> ` is treated as a conflict
  marker — keep that in mind for docs that use such lines (e.g. reStructuredText section underlines).

With these settings, every hook that can meaningfully run against a clean checkout enforces in CI
exactly as it does locally.

### `dco.yaml` — DCO sign-off check

**Trigger:** Pull request to `main`

Checks that every non-merge commit in the PR has a `Signed-off-by:` line. Skips merge commits automatically.

---

## Manual: GPU Tests

GPU tests (`tests/vllm/`, `tests/integration/`) cannot run on GitHub-hosted runners. They are triggered manually by repository admins via the `gpu-tests.yaml` workflow.

**Trigger:** `workflow_dispatch` — only users with write access to the repo can trigger this from the GitHub UI (`Actions` → `GPU Tests` → `Run workflow`).

**Current state:** The workflow is scaffolded but inert until a [self-hosted runner](https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/about-self-hosted-runners) with the `gpu` label is registered on a machine inside the IBM network. No code changes are needed when the runner is set up — just register it.

In the meantime, run GPU tests manually on the HPC environment:

```bash
make test-gpu        # vllm + integration tests
make test-gpu-full   # + regression suite
```

The `composer` and `hf` suites are CPU-runnable but too heavy for the GitHub
Actions quota, so they are excluded from CI (only `tests/unit/` runs there). Run
them manually on a machine with adequate RAM/disk — locally or on the cluster:

```bash
make test-cpu        # unit + composer + hf
make test-composer   # composer only
make test-hf         # hf only
```

---

## Coverage

CI runs tests with `pytest-cov` and uploads `coverage.xml` to [Codecov](https://codecov.io). Codecov posts a per-PR diff comment showing which lines added in the PR are covered.

**Coverage is scoped** in `pyproject.toml` under `[tool.coverage.run]`: `source = ["granite_switch"]`
with the `hf`, `vllm`, `composer`, and `tutorials` subpackages **omitted**. Those need a GPU, real
model checkpoints, or dependencies CI doesn't install, and are covered on the GPU cluster instead
(see [Manual: GPU Tests](#manual-gpu-tests)). Without the omit list the CI number would be diluted
to ~1% by thousands of lines the unit suite can never reach; with it, the reported percentage
reflects only the code the unit suite is actually responsible for.

**One-time setup (repo admin):**
1. Sign up at [codecov.io](https://codecov.io) and link the repository
2. Add the `CODECOV_TOKEN` secret to the repository (`Settings` → `Secrets and variables` → `Actions`)

Until the token is configured, an upload failure is non-fatal (`fail_ci_if_error: false`): the
upload step logs the error but the build stays green and CI is not blocked.

---

## Releases & Publishing

granite-switch publishes to [public PyPI](https://pypi.org/project/granite-switch/). Releases are created via GitHub Releases; publishing happens automatically.

### Versioning

Semantic versioning: `MAJOR.MINOR.PATCH`. The version lives in `pyproject.toml` as the single source of truth:

```toml
version = "X.Y.Z"
```

While in pre-1.0, increment `MINOR` for new features and `PATCH` for bug fixes. Bump to `1.0.0` when the public API stabilizes.

### Release process

1. **Open a release PR** from a `release/vX.Y.Z` branch:
   - Bump `version` in `pyproject.toml`
   - Add a section to `CHANGELOG.md`

2. **Merge to `main`**

3. **Create a GitHub Release** (`Releases` → `Draft a new release`):
   - Tag: `vX.Y.Z` (created from `main`)
   - Title: `vX.Y.Z`
   - Body: paste the `CHANGELOG.md` section for this release

4. **Click Publish release** — `publish.yaml` triggers automatically

### What `publish.yaml` does

Builds wheel + sdist with `uv build`, then uploads with `uv publish`. Uses **PyPI Trusted Publisher (OIDC)** — no token required.

### One-time PyPI setup (repo admin)

1. Create the project on PyPI (first release can be done manually)
2. Go to `pypi.org` → project page → `Manage` → `Publishing`
3. Add a Trusted Publisher:
   - Owner: `generative-computing`
   - Repository: `granite-switch`
   - Workflow: `publish.yaml`
   - Environment: `pypi`
4. Create a `pypi` environment in the GitHub repo (`Settings` → `Environments`)

---

## Execution

A running record of rollout milestones actually executed against the plan.

### Post ruff formatting

**What:** After applying the repo-wide ruff format + lint autofixes (the PR 1
"format first" step of the [Ruff rollout](#ruff-rollout-format-first-then-enforce)),
the full test suite was run to confirm the mechanical changes introduced no
regressions.

**How:** Submitted to the Vela GPU cluster (namespace `security`, 4 GPUs,
`vllm26` dependency group) via a job config derived from `tests_on_uv_vllm26.yaml`,
pointed at the formatted branch. It runs a ruff sanity check followed by all five
suites with `pytest -n 4`.

- Branch under test: `chore/ruff-format` (format commit)
- Ruff sanity: `ruff check .` → **All checks passed!**; `ruff format --check .` → clean

**Results:** 1,338 passed, 2 failed.

| Suite | Result |
|-------|--------|
| `unit` | PASS — 86 passed |
| `hf` | PASS — 563 passed, 16 skipped |
| `composer` | PASS — 239 passed, 1 skipped |
| `vllm` | PASS — 425 passed, 2 skipped |
| `integration` | FAIL — 2 failed, 25 passed |

**The 2 failures — assessed as pre-existing, not caused by the reformat.**
Both are the same test, `test_hf_vllm_argmax_equivalence`
(`tests/integration/test_switch_e2e_compose.py`), for `granite-4.0-micro` and
`granite-4.1-3b`. Each fails on a **single token position** where the HF and
vLLM backends pick different top-1 tokens:

- `granite-4.0-micro`: position 6 — HF `2163` vs vLLM `1314`
- `granite-4.1-3b`: position 3 — HF `2010` vs vLLM `3575`

This is the classic near-tie / floating-point signature: at one position two
tokens have near-equal logits and tiny numerical differences between the
backends (fused vs. unfused kernels, reduction order) flip the argmax. It is
very unlikely to stem from PR 1 because:

1. The reformat only changed whitespace/layout, import ordering, and removed two
   provably-unused locals — none of which touch any computation.
2. Every other cross-backend equivalence test passed, including all 425 `vllm`
   tests (`test_generation_equivalence`, `test_upstream_equivalence`) and the
   integration `test_forward_logit_equivalence`.

**Follow-up:** Confirm the same test also flips a position on unmodified `main`
(e.g. an integration-only run from `tests_on_uv_vllm26.yaml`). If so, treat it as
a flaky near-tie test to be addressed separately (tolerance/top-k handling), and
consider PR 1 cleared.

### CPU CI marker split

**What:** The first `test-cpu` runs on GitHub Actions were cancelled — not on a
test assertion, but with `Error: The operation was canceled.` at
`STEP 5: Saving model and tokenizer → Writing model shards`. A composer test was
downloading a base model + adapters and writing a multi-shard checkpoint on a
GitHub-hosted runner, exhausting its time/memory/disk budget.

**Root cause:** the heavy-test markers (`requires_model` / `slow` / `gpu`) were
incompletely applied. `tests/composer/test_save_load_compose.py` composes a real
checkpoint and its docstring claimed "Marked slow + requires_model," but it
carried no marker — so the default `-m "not deep"` filter ran it in CI.

**Fix:** completed the module's markers and switched the `test-cpu` job to the
denylist `-m "not requires_model and not gpu and not slow and not deep"`. The
`not slow` clause is load-bearing, not redundant: `TestRealHubMetadata` in
`test_selective_download.py` hits the real Hub for repo metadata and is marked
`slow` only (it needs no checkpoint, so `requires_model` would be inaccurate).
Heavy coverage continues to run on the GPU cluster.

---

## Roadmap

### Ruff rollout: format first, then enforce

The existing tree is not yet ruff-clean — a whole-repo `ruff check .` / `ruff format --check .` reports hundreds of findings across `src/`, `tests/`, `tutorials/`, and `docs/`. Enforcing ruff in the same PR that introduces the hooks would bury a small config change under a massive mechanical reformat.

**Decision:** roll ruff out in **two separate PRs**, formatting first.

Before PR 1 lands, ruff is intentionally **advisory** as a temporary bridge (CI `continue-on-error: true`; pre-commit `ruff` with `--exit-zero`, no auto-format). **On this branch PR 1 has landed** and the tree is ruff-clean, so ruff is now **blocking**: CI drops `continue-on-error` and the pre-commit hooks are `ruff-format` + `ruff --fix` (fail on any change).

#### PR 1 — repo-wide format (branches off `main`, touches source only)

1. Add the final `[tool.ruff]` config to `pyproject.toml` **in this PR** — formatting output depends on the config + ruff version, so they must be fixed here and match what the hooks pin (ruff `v0.9.0`). Include an `__init__.py` guard so autofix does not strip re-exported imports:
   ```toml
   [tool.ruff.lint.per-file-ignores]
   "**/__init__.py" = ["F401"]
   ```
2. Run `ruff format .` then `ruff check --fix .` (safe fixes only — **no** `--unsafe-fixes`) with the pinned ruff version. **Review the `--fix` diff**, especially import removals (F401) and import reordering (I), before committing.
3. Land it as a **single commit** and record that commit's SHA in `.git-blame-ignore-revs` so it doesn't obscure `git blame`.
4. Verify `ruff format --check .` and `ruff check .` are both clean.

#### PR 2 — hooks + CI/CD (this PR, rebased on top of PR 1; touches no source)

Once PR 1 is merged, rebase this branch and flip ruff to blocking:

- **`ci.yaml`**: remove `continue-on-error: true` from the two ruff steps.
- **`.pre-commit-config.yaml`**: re-add the `ruff-format` hook and change the `ruff` hook args to `[--exit-non-zero-on-fix, --fix, --config=pyproject.toml]`.

Because the tree is already clean, this PR contains only config and CI wiring, and passes the (now blocking) gate on the first run.

#### Caveats

- **Coordinate timing.** A repo-wide reformat conflicts with every open branch; contributors with in-flight PRs must rebase through the format commit. Land PR 1 when few PRs are in flight, or warn contributors first.
- **`nbstripout` is orthogonal to CI.** It's a pre-commit-only hook, so it does not gate CI and does not need notebooks pre-stripped for PR 2 to pass. The first notebook commit after it lands will strip that notebook's outputs — expected, not a regression.

### Phase 2: Mypy Type Checking

Mypy is deferred because the codebase has no existing `[tool.mypy]` configuration and retrofitting type annotations across an existing codebase is a significant effort worth its own PR.

#### Strictness level: curated middle ground (not `--strict`)

`--strict` is the wrong choice here. PyTorch, HuggingFace, and vLLM all have incomplete or missing type stubs — `Any` leaks in from those library boundaries and propagates everywhere, meaning most mypy errors would be fighting stub quality rather than catching real bugs in granite-switch code. SPINE uses `strict = true` because it was designed with types in mind from day one; this codebase is retrofitting.

The recommended config enforces the flags that deliver the highest signal-to-noise ratio:

**`pyproject.toml`:**
```toml
[tool.mypy]
python_version = "3.11"
pretty = true
ignore_missing_imports = true   # vllm has no stubs; HF stubs are partial
disallow_untyped_defs = true    # every function must be annotated — highest-value flag
warn_return_any = true          # catch Any leaking out of our own functions
warn_unused_ignores = true      # keep # type: ignore comments from going stale
exclude = ["^tests/"]
```

`disallow_untyped_defs` is the core flag: it ensures every function added going forward is annotated, which is where mypy earns its keep. Tighten toward `--strict` incrementally as stub quality improves and the codebase is fully annotated.

**`.pre-commit-config.yaml`** (add to `local` repo block):
```yaml
- id: mypy
  name: MyPy
  entry: uv run --no-sync mypy src
  pass_filenames: false
  language: system
  files: '\.py$'
```

**`ci.yaml`** (add job):
```yaml
mypy:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: astral-sh/setup-uv@v5
    - run: uv sync --frozen --extra hf --extra compose
    - run: uv run mypy src
```
