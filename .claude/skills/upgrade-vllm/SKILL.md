---
name: upgrade-vllm
description: Autonomously bring granite-switch up to the newest released vLLM lines. Detects which vLLM versions are released but unsupported, ports the backend to them with capability gating, does the pyproject/uv.lock/CI-matrix plumbing, drives CPU + GPU + evaluation gates to green, and opens a PR. Use when asked to upgrade vLLM, support a new vLLM version, check for new vLLM releases, or run the vLLM upgrade.
---

# Upgrade granite-switch to the newest vLLM lines

Runs end to end without supervision. The human's only job is reviewing and merging
the PR. Read `REFERENCE.md` (next to this file) before touching anything — it holds
the exact version surface and the traps that have already cost this repo a red CI.

## Operating rules

These are not suggestions. Each one is a failure this repo has actually hit.

1. **Gate on capability, never on a version number.** Use `hasattr`/`try: import`
   to detect the API shape. A `vllm.__version__` comparison is wrong the moment
   upstream renames something, and it does not state the actual requirement. There
   is currently **zero** version-number branching in `src/` — keep it that way.
2. **Never delete a dependency group that main's GPU matrix still names.** See
   REFERENCE.md §4. Deleting one makes the leg's `uv sync --group X` fail before
   pytest starts, producing a sub-1.1-KB log that looks like a test failure.
3. **Relock with uv 0.8.4 exactly** — the `rev` of the `uv-lock` pre-commit hook.
   A newer uv writes a different lockfile revision and the hook fails CI.
4. **Every commit needs `git commit -s`.** The DCO workflow blocks the PR otherwise.
   Do **not** add a `Co-authored-by: Claude` trailer; disclose AI assistance in the
   PR body instead. (Project rule — see CLAUDE.md and REFERENCE.md §7.)
5. **Never `--no-verify`.** Fix what the hook complains about.
6. **Dispatch GPU tests at the PR branch ref, not the default branch.** `gh workflow
   run gpu-tests.yaml --ref <branch>` runs *that branch's* workflow file, so new
   matrix legs take effect immediately. The `/gpu-test` comment path cannot do this
   (it hardcodes `--ref <default branch>`), which is why this skill bypasses it.
7. **Always pass `no_cache=true`.** The GPU result cache is keyed on sha+leg, and
   the test pod pins `vllm/vllm-openai:latest`, so a cached pass answers "what did
   this commit do" when the question is "is this still true today".
8. **Open the PR as a draft** and only mark it ready when every gate is green.
   A human seeing a ready PR assumes the gates passed.

## Procedure

Keep a state file at `.claude/.vllm-upgrade-state.json` (gitignored) recording the
current phase, branch, PR number, target versions, and every dispatched run id.
Write it after each phase. If the skill is re-invoked and the file exists, resume
from the recorded phase instead of starting over — GPU legs run up to 420 minutes
each and a session will not survive the whole wait.

### Phase 0 — preflight

```bash
.claude/skills/upgrade-vllm/scripts/preflight.sh
```

Checks `gh` is installed and authenticated, that the authenticated identity holds
**Maintain or Admin** on the repo (the GPU workflow's `check_role.sh` rejects
anything less), that `uv 0.8.4` is available, that SSH push works, and that the
working tree is clean. **Abort on any failure and tell the user exactly what to
install or grant.** Do not attempt to work around a missing `gh`.

### Phase 1 — detect the gap

```bash
uv run .claude/skills/upgrade-vllm/scripts/vllm_versions.py
```

Prints the supported ceiling (from the `vllm` extra's upper bound in
`pyproject.toml`), the released lines on PyPI, and the gap. If the gap is empty,
say so and stop — do not open an empty PR.

Target **every** line in the gap, not just the newest: the repo's claim is a
contiguous range, and an untested middle line is a false claim. Yanked and
pre-release versions are excluded by the script.

### Phase 2 — branch

```bash
git checkout main && git pull --ff-only
git checkout -b feature/vllm-<floor>-to-<new-ceiling>
```

`<floor>` is the existing lower bound (unchanged unless Phase 4 forces a drop).

### Phase 3 — research what actually broke

For each new line, diff upstream over only the API surface this repo touches:

```bash
.claude/skills/upgrade-vllm/scripts/upstream_api_diff.sh v<old-ceiling>.0 v<new>.0
```

It clones/updates a cached vLLM checkout and reports, per symbol that
`src/granite_switch/vllm/` and `tests/` import or override, whether it was
**deleted, renamed, or had its signature narrowed** between the two tags, plus the
upstream commits responsible.

Read the commits it names. The historically load-bearing surface is listed in
REFERENCE.md §3 — multimodal processor hooks, the MoE topk kernels' forward-context
mask, `AttentionBackend` KV-cache shape helpers, and `vllm.Attention`. Produce a
written findings list before editing; each finding becomes a paragraph in the PR
body and, where it is a behaviour change rather than a rename, a test.

> A deleted hook whose override is silently never called is the worst case and
> will not show up as a failure — it shows up as a feature quietly not working.
> vLLM 0.29 did exactly this to the ASR path. When a finding is "our override no
> longer gets called", add a test that asserts the *effect*, not the call.

### Phase 4 — port `src/` and `tests/`

Apply the fixes with capability gating. Two things to decide explicitly and record:

- **Does the new line force the floor up?** If an API the port now requires does
  not exist on the current floor and cannot be capability-gated cheaply, the floor
  must rise. Record which lines are dropped — this is a mandatory PR-body section
  (Phase 9) so the `vllm` extra and any downstream pin get updated knowingly.
- **Does the new line force a dependency ceiling up?** `transformers` is pinned
  `<=5.17.0` in `pyproject.toml`. A new vLLM commonly demands a newer one. Raising
  it is in scope; say so in the PR body.

### Phase 5 — plumbing

```bash
uv run .claude/skills/upgrade-vllm/scripts/plumb_pyproject.py --add 0.31 0.32 --ceiling 0.32
```

This edits `pyproject.toml` to add each new line's `vllmNN` group, its `dev-vllmNN`
group, widen the `vllm` extra's upper bound, and **regenerate the entire
`[tool.uv] conflicts` table** from the set of declared lines. Regenerating is the
point: the table needs every cross-line pair (`N` lines → `N*(N-1)/2 * 4` entries),
and hand-editing it is where this goes wrong. The script never removes a group.

Then relock and move the CI ceiling:

```bash
uvx uv@0.8.4 lock   # NOT bare `uv lock`
```

Always the pinned form, even when the local `uv` happens to be 0.8.4 — the local
version is not guaranteed and a newer one writes a different lockfile revision,
which the `uv-lock` pre-commit hook then rewrites and fails on. Take the version
from `.pre-commit-config.yaml`'s `uv-pre-commit` rev rather than this line, so a
hook bump cannot leave the skill relocking with the wrong one.

Edit `.github/workflows/gpu-tests.yaml`'s matrix so the two legs are the **ends of
the new supported range** (floor + new ceiling), matching the rationale already
written in `pyproject.toml`'s `[dependency-groups]` comment. Keep the old ceiling's
group declared even when it stops being a leg (rule 2). Update that comment and the
`vllm` extra's comment to name the new range — a stale comment claiming the old
range is a review failure in this repo.

Also update the version range wherever it is stated as prose: `README.md`,
`tutorials/PREREQUISITES.md`, `docs/`. `grep -rn "0\.<old-ceiling>"` to find them.

### Phase 6 — CPU gate, locally, before pushing

```bash
uvx pre-commit run --all-files --show-diff-on-failure
uv run pytest tests/unit/ -m "not requires_model and not gpu and not slow and not deep" -v -s --tb=short -x
```

Re-stage and re-run if a hook rewrote files. Both must be green before Phase 7 —
burning a CI run on something reproducible locally wastes everyone's time.

### Phase 7 — commit, push, open a draft PR

One commit per concern, following this repo's style: a short summary line, then
prose explaining **what changed and why**, including the evidence (which upstream
commit, which failure signature). Read `git log` on `src/granite_switch/vllm/` for
the register — the commit messages here are long and explanatory by design.

```bash
git commit -s        # per concern
git push -u origin feature/vllm-<floor>-to-<new-ceiling>
gh pr create --draft --title "..." --body-file <generated>
```

### Phase 8 — drive the gates to green

Loop, with a budget of **5 fix attempts per leg**. Exceeding it means stop and
report rather than thrash.

```bash
# one dispatch per suite; run the version legs in parallel (the matrix does that)
.claude/skills/upgrade-vllm/scripts/gpu_dispatch.sh --ref <branch> --suite full --wait
.claude/skills/upgrade-vllm/scripts/gpu_dispatch.sh --ref <branch> --suite eval --wait
```

The script dispatches with `no_cache=true`, polls, and on failure downloads the
`gpu-tests-log-<leg>-<sha>` artifact and prints the failing tests.

Two result-reading rules:

- A **sub-1.1-KB log** means the suite never started — almost always an
  unresolvable `--dep-group`, not a test failure. Check the group exists in
  `pyproject.toml` on the dispatched ref before touching any test.
- The CPU job's **"Upload coverage" step can be red while the job is green**
  (`continue-on-error: true`). Read the `Run CPU tests` step conclusion, not the
  job's step list.

Also watch for environment requirements that are version-dependent. vLLM 0.30
needed `VLLM_USE_FLASHINFER_SAMPLER=0` (now in `tests/conftest.py:44`) because the
runner image's CUDA 12.4 could not build its sampler. If a new line dies at engine
init with an `nvcc` error, that class of workaround — set in `tests/conftest.py`
with a comment naming the version and the error — is the fix.

After each fix: amend nothing, add a **new** commit, push, re-dispatch.

### Phase 9 — PR body and hand-off

Generate the body from `pr_body_template.md` in this directory. It must contain:

- The old and new supported range.
- **A "Support dropped" section.** If the upgrade makes a previously supported
  line unsupported, name each dropped line, say why, and state that the `vllm`
  extra's lower bound was raised to match. If nothing was dropped, say
  "None — the range only widened." Never omit the section; its absence reads as
  "nothing was dropped".
- One paragraph per Phase 3 finding: the upstream change, how it manifested here,
  and how it is gated.
- Any dependency ceiling raised (`transformers`, `torch`).
- The gate results table: CPU 3.11/3.12, each GPU leg, each of the four evals,
  with run links.
- A note that the change was prepared with AI assistance (required by the
  contribution guidance; must **not** be a commit trailer).

Then `gh pr ready <n>` and report the PR URL. **Do not merge.**

## The four evaluation jobs

`tests/eval/` is a GPU suite dispatched as `--suite eval`. The four configs are
`granite42_vllm`, `granite5_vllm`, `sr_vllm`, and `granite42_hf` — Granite 4.2 and
Granite 5 with LoRA/aLoRA adapters on vLLM, an SR adapter on vLLM, and one adapter
on the HF backend.

Each reads its checkpoint from a pre-staged path on the GPU cluster
(`GSW_EVAL_*_DIR`) and compares against a **committed baseline** in
`tests/eval/baselines/`, which records the top-k logprob distributions for a fixed
prompt set together with the vLLM version they were captured on. A config whose env
var is unset skips with a reason — it does not silently pass.

The baseline is the regression gate, so it must be captured on a **known-good**
version. Capture it from the current ceiling *before* porting:

```bash
uv run --group dev-vllm<old-ceiling> \
  .claude/skills/upgrade-vllm/scripts/record_eval_baseline.py --config all
```

Only re-record a baseline when the user explicitly accepts a quality change.
Re-recording to make a failing eval pass destroys the gate — if an eval fails,
that is the signal the upgrade was looking for. Report it and stop.
