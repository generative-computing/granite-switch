# Contributing to Granite Switch

Thank you for your interest in contributing to Granite Switch!

## Prerequisites

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. Install it once before working on the project:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or via pip: `pip install uv`

> **Keep `uv.lock` at `revision = 3`.** The `revision` field at the top of `uv.lock` is the lockfile
> *format* version, bumped automatically by newer uv releases (uv 0.12.x writes `revision = 5`). It is
> **not** a dependency change, and we pin it at `3` so the committed lockfile stays readable by the uv
> versions the team and CI use. If `git diff uv.lock` shows only `revision = 3` → `revision = 5` (or
> similar) after a `uv sync`/`uv lock`, that bump is spurious — revert it before committing:
> ```bash
> git checkout uv.lock   # if revision is the only change
> ```
> If you also changed dependencies, re-lock and then reset just the first line back to `revision = 3`
> by hand (edit `uv.lock` line 2) so only the real dependency changes land. Do not commit a bare
> `revision` bump.

## Getting Started

1. Fork the repository
2. Clone your fork and install dependencies:
   ```bash
   git clone https://github.com/<your-username>/granite-switch.git
   cd granite-switch
   uv sync --group dev
   ```
   > **On macOS (or any machine without a CUDA GPU):** the `dev` group pulls in vLLM + CUDA,
   > which have no macOS wheels, so `uv sync --group dev` fails. Install the CPU-only subset
   > instead — enough for unit, HF, and compose work:
   > ```bash
   > uv sync --frozen --no-default-groups --extra hf --extra compose
   > ```
   > `--no-default-groups` skips the default `vllm26` group; `--frozen` installs strictly from the
   > committed `uv.lock` without modifying it. (What fails on macOS is *installing* the vLLM/CUDA
   > wheels — resolving the lockfile with `uv lock` works fine, so the `uv-lock` pre-commit hook runs
   > locally too; the CPU subset above simply never installs those wheels.) To run the
   > CPU test suites the same way, prefix pytest with the same flags, e.g.
   > `uv run --frozen --no-default-groups --extra hf --extra compose pytest tests/unit tests/hf tests/composer`.
   > vLLM and integration tests are Linux + GPU only and cannot run locally on macOS.
3. Enable the project's pre-commit hooks (ruff, nbstripout, link/import validation, SPDX/DCO checks, and basic hygiene checks). pre-commit manages its own isolated environment per hook, so install it as a standalone tool rather than into the project venv:
   ```bash
   uv tool install pre-commit                              # one-time, isolated — any platform
   pre-commit install                                      # hooks for staged files
   pre-commit install --hook-type prepare-commit-msg       # auto-adds your DCO sign-off
   pre-commit install --hook-type commit-msg               # verifies the DCO sign-off
   git config blame.ignoreRevsFile .git-blame-ignore-revs
   ```
   > **Don't use `uv run pre-commit`.** `uv run` first syncs the project's default dependency group
   > (vLLM + CUDA), which fails on macOS and needlessly pulls the GPU stack on Linux. `uv tool
   > install` (or `pipx`/a system package) installs pre-commit standalone and works on every platform.
   >
   > **All three `install` commands are required.** `pre-commit install` alone only wires up the
   > staged-file hooks; the DCO hooks run at the `prepare-commit-msg` and `commit-msg` stages and
   > must be installed explicitly. With them in place your sign-off is added automatically — see
   > [Sign off your commits (DCO)](#sign-off-your-commits-dco) below.
4. Create a feature branch and make your changes
5. Run tests: `uv run pytest tests/ -v` (see [Tests that need a model](#tests-that-need-a-model) before writing one)
6. Submit a pull request

## Tests that need a model

PR runs download no real model. A test that needs a Granite base or an adapter
library runs on a **fake** on every PR, and on the **real** model only in the
release tier (`/gpu-test-release`). The fakes keep the real config, chat template,
special-token ids and repo layout, with tiny random weights. The full design is in
[docs/CICD.md, "Release test tier"](docs/CICD.md#release-test-tier).

**Get models from `tests/shared/model_sources.py`, never from a Hub id:**

```python
from tests.shared import model_sources

@pytest.mark.parametrize("source", model_sources.sources())   # [fake] on PRs, [real] in release
def test_something(source):
    base = model_sources.base_model(source, "granite-4.1-3b")
    library = model_sources.adapter_library(
        source, "ibm-granite/granitelib-rag-r1.0", "granite-4.1-3b"
    )
```

`sources()` puts the `release` marker on the real case. For a parameter list of
your own, use `pytest.param(..., marks=pytest.mark.release)` on the real cases.

**Rules:**

- **Resolve the model inside the test or fixture, not at collection.** Calling
  `base_model()` or `adapter_library()` at import time builds fakes on every
  collection, including for tests that are deselected, and anything loaded from
  the returned real id at import time downloads on PR runs.
- **Fail, don't skip, when the model doesn't load.** A skip on a load error turns
  a broken fake or a missing capture into a silent pass.
- **Pick the fake's weights by what you assert** (`preset=` from
  `tests/shared/fake_bases.py`):
  - `REALISTIC` (the default): hidden states stay at realistic scale, so exact
    score and logit tolerances hold, but adapters barely move the output.
  - `LIVE_LAYERS`: every layer and adapter changes the output. Use it when a test
    asserts that adapters differ or that a sublayer matters.
  - `dims=TP_DIMS`: more attention heads, for tensor-parallel tests that need
    several KV heads per rank.
- **Don't assert on what a fake can't provide:** generated-text quality, trained
  behaviour, or real token counts. The fake tokenizer splits ordinary words into
  one token per byte, so a prompt is several times longer in tokens. Put such
  assertions only in the real case, or make the test release-only.
- **Release-only tests are the exception.** Mark a test `release` with no fake
  case only when a fake can't stand in (real weights, the real vocabulary, the live
  Hub), and add it to the list in [docs/CICD.md](docs/CICD.md#release-test-tier).
- **A new base or adapter library** needs its files captured: add it to `BASES` in
  `tests/shared/fake_bases.py` or `LIBRARIES` in
  `tests/shared/fake_adapter_libraries.py`, then run
  `python -m tests.shared.fake_bases --capture` or
  `python -m tests.shared.fake_adapter_libraries --capture`, and commit the files
  under `tests/shared/fixtures/`.

**Running a subset with `-m`:** `pyproject.toml` sets `addopts` to
`-m "not deep and not release"`. A `-m` on the command line **replaces** that
filter instead of adding to it, so add the exclusions back yourself:

```bash
pytest -m "audio and not deep and not release" tests/   # not: pytest -m audio
pytest -m release tests/                                  # the release tier only
```

Without them, a run silently includes every release test and downloads real
models. Every `-m` in the CI workflows carries these exclusions for the same
reason.

**Older end-to-end tests** gated on `GRANITE_SWITCH_E2E_MODELS=1` (for example
`tests/hf/test_multi_switch_e2e.py`) still name real models directly and skip
unless the variable is set. They predate this scheme; don't copy them as a
template.

## Sign off your commits (DCO)

Every commit must carry a Developer Certificate of Origin (DCO) sign-off. By adding a sign-off you
certify that you wrote the change (or otherwise have the right to submit it) under the project's
license — see the [full DCO text](https://developercertificate.org/). A commit-msg hook and a CI
check both reject commits that are missing it.

A sign-off is just a trailer line at the end of the commit message:

```
Signed-off-by: Jane Doe <jane.doe@example.com>
```

### 1. Configure your name and email

The sign-off is generated from your Git identity, so set it once (drop `--global` to scope it to
this repository only):

```bash
git config --global user.name "Jane Doe"
git config --global user.email "jane.doe@example.com"
```

The name and email **must match** the values you sign off with — a real name and a reachable email
are expected, and the email must be a valid `name@host` address (the check requires an `@`). Verify
with:

```bash
git config user.name
git config user.email
```

### 2. Let the hook sign off for you (recommended)

Once the `prepare-commit-msg` hook is installed (step 3 of [Getting Started](#getting-started)), your
sign-off is appended **automatically** from the Git identity above — every `git commit` just works,
no flags to remember:

```bash
git commit -m "Your commit message"   # Signed-off-by: is added for you
```

The `commit-msg` hook then verifies the trailer is present, and the CI DCO check enforces it on the
PR as a backstop.

### Signing off manually

If you haven't installed the hook (e.g. a fresh clone), pass `-s` (`--signoff`) and Git appends the
trailer itself:

```bash
git commit -s -m "Your commit message"
```

Forgot it on your last commit? Amend it:

```bash
git commit --amend -s --no-edit
```

To sign off every commit on an existing branch in one go:

```bash
git rebase --signoff origin/main
```

## Contribution Guidelines

See [docs/GIT_WORKFLOW.md](docs/GIT_WORKFLOW.md) for detailed workflow, commit conventions, and code quality standards.

## Areas of Interest

- **Bug fixes** — Identify and fix issues in the codebase
- **Documentation** — Improve tutorials and guides

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). Please read it before participating.

## Questions?

Open an issue or start a discussion.
