# Contributing to Granite Switch

Thank you for your interest in contributing to Granite Switch!

## Prerequisites

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. Install it once before working on the project:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or via pip: `pip install uv`

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
5. Run tests: `uv run pytest tests/ -v`
6. Submit a pull request

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
