---
name: upgrade-vllm
description: Autonomously bring granite-switch up to the newest released vLLM lines. Detects which vLLM versions are released but unsupported, ports the backend to them with capability gating, does the pyproject/uv.lock/CI-matrix plumbing, drives CPU + GPU + evaluation gates to green, and opens a PR. Use when asked to upgrade vLLM, support a new vLLM version, check for new vLLM releases, or run the vLLM upgrade.
---

# Upgrade granite-switch to the newest vLLM lines

Runs end to end without supervision. The human's only job is reviewing and merging
the PR. Read `REFERENCE.md` (next to this file) before touching anything — it holds
the exact version surface and the traps that have already cost this repo a red CI.

## What invoking this skill must deliver

This is the contract. Nothing here is optional, and **none of it asks the user for
permission** — invoking the skill is the authorization. Decide, act, and report at
the end. The four exceptions are enumerated in operating rule 9.

1. **Compare and decide.** Read the newest released vLLM lines and this repo's
   supported ceiling. If released > supported, every line in the gap is a target.
   If the gap is empty, say so and stop — no empty PR.
2. **One PR covering all new lines.** Not one PR per line. The repo's claim is a
   contiguous range, so the range moves in one reviewable change.
3. **Port the code** so every new line works, capability-gated (rule 1).
4. **Verify every added line independently** — CPU, GPU and the regression/eval
   tiers — rather than testing the newest and inferring the rest. "0.31 and 0.32
   were added" means 0.31 and 0.32 each have their own green evidence.
5. **A PR body with per-version proof tables.** One column per added line, a ✅
   per gate that actually ran and passed on that line, and a separate table giving
   the regression/eval numbers **against the existing baseline** for each added
   line. A gate that did not run is `—`, never ✅.

The point of 4 and 5 together: a reviewer must be able to see, per version, that
each gate ran — not take "the suite is green" on faith.

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
9. **Do not ask the user to approve the routine steps.** Branching, editing
   `src/`, relocking, widening the matrix, committing, pushing, dispatching GPU
   runs, submitting the eval jobs, opening the PR and marking it ready are all
   pre-authorized by invoking this skill. Narrate what you are doing; do not stop
   for a yes. There are exactly four reasons to stop and ask:
   - **preflight fails** (Phase 0) — the run cannot complete unattended anyway;
   - **a gate is still red after 5 fix attempts** (Phase 8) — report the failure
     with the log, do not thrash;
   - **a scored eval returns DRIFT** — accepting a quality change is the user's
     decision, and re-recording a baseline to make it pass destroys the gate;
   - **the floor must rise** (Phase 4), i.e. the upgrade would *drop* a supported
     line. Widening the range is routine; narrowing it is not.

   And one thing never to do: **merge.** The hand-off is a ready PR.

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
.claude/skills/upgrade-vllm/scripts/upstream_api_diff.py v<old-ceiling>.0 v<new>.0
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
# every line in the gap, in one invocation; add --floor 0.NN only if Phase 4
# concluded the floor must rise
.claude/skills/upgrade-vllm/scripts/plumb_pyproject.py --add 0.31 0.32
.claude/skills/upgrade-vllm/scripts/plumb_pyproject.py --check   # must print "complete"
```

The ceiling is **derived** from the highest `--add`, so there is no `--ceiling`
flag. The only other flag is `--floor`, which *drops* support below it.

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

Then the GPU matrix in `.github/workflows/gpu-tests.yaml`. **Two distinct matrices
are involved and conflating them is how a version ships untested:**

- **The in-PR matrix — one leg per added line, plus the floor.** Every line this
  PR adds needs its own green evidence (contract item 4), and a leg is the only
  thing that produces it. Adding 0.31 and 0.32 means four legs this run: `vllm26`
  (floor, `dep_group: dev`), `vllm30` (old ceiling), `vllm31`, `vllm32`. They
  cannot share a venv — `[tool.uv] conflicts` forbids it — so each is its own pod,
  and `fail-fast: false` keeps one red leg from hiding the others.
- **The matrix left behind on merge — back to the two ends** (floor + new
  ceiling), matching the rationale already written in `pyproject.toml`'s
  `[dependency-groups]` comment. Per-run cost is why: a middle line is bracketed
  by two tested lines, so it adds install time without adding a failure mode the
  ends would miss.

So the sequence is: widen the matrix to cover every added line, dispatch, collect
the per-line results that become the PR tables, then narrow it back to the two ends
in a **final commit** whose message records which lines were verified out of band
and in which run. That last commit is what makes the existing `[dependency-groups]`
comment true again — it already claims the middle lines were "verified together out
of band", and this is the step that earns the claim.

Do **not** delete a group when its leg goes away (rule 2). Update that comment and
the `vllm` extra's comment to name the new range — a stale comment claiming the old
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
# one dispatch per suite; the widened matrix runs every version leg in parallel
.claude/skills/upgrade-vllm/scripts/gpu_dispatch.sh --ref <branch> --suite full --wait
.claude/skills/upgrade-vllm/scripts/gpu_dispatch.sh --ref <branch> --suite eval --wait
```

The script dispatches with `no_cache=true`, polls, and on failure downloads the
`gpu-tests-log-<leg>-<sha>` artifact and prints the failing tests.

**Record the per-leg conclusion and run URL for every leg as they land** — that is
the raw material for the PR's per-version table (contract item 5), and it is not
reconstructible later once a run ages out. Write them into the state file as you
go. `full` covers the five GPU suites including `tests/vllm/`, `tests/hf/`,
`tests/composer/` and `tests/integration/` — this repo's regression coverage lives
in those, not in a `tests/regression/` directory (it does not exist; see
REFERENCE.md §5's stale-things list).

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

Once a line's leg and drift eval are green, launch the **scored** evals (tier 2 —
see below) for that line against the PR head sha. They take hours, so start them as
soon as the cheap gates pass rather than at the end.

**Once per added line, not once per PR.** Each line gets its own four jobs, which
is what fills that line's column in the regression table:

```bash
SHA="$(git rev-parse HEAD)"
for LINE in 0.31 0.32; do
  # --out PER LINE. Without it every line renders into the same directory and
  # `kubectl apply -f <dir>` re-submits the previous line's jobs as well.
  OUT="scratch/eval-jobs/v${LINE//./}"
  .claude/skills/upgrade-vllm/scripts/render_eval_jobs.py \
      --vllm "$LINE" --sha "$SHA" --out "$OUT"
  kubectl apply -f "$OUT/"
done
kubectl -n security get appwrappers | grep itzikv-eval
```

Job names and result paths both carry the line (`itzikv-eval-v031-…`,
`…-v032-…`), so the two sets coexist in the namespace and neither overwrites the
other's results under `paths.results`. Collect each job's printed accuracy, delta
and verdict — four numbers per line, and that is the regression table.

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
- **The per-version coverage table — one column per added line.** Rows: pre-commit,
  CPU 3.11, CPU 3.12, each GPU suite, each of the four drift evals. A `✅` only
  where that gate actually ran and passed **on that line**; `—` where it does not
  apply (CPU rows do not vary by vLLM line — say so in a footnote rather than
  ticking them per column). Never ✅ a cell by inference from a neighbouring
  column: that is the exact failure this table exists to prevent.
- **The regression table — measured vs. existing baseline, per added line.** For
  each of the four `evals.yaml` configs: the bar, then that line's accuracy, delta
  and verdict. Four configs × N added lines. This is what answers "did the model
  get worse on 0.32 specifically".
- Run links for every ✅, so a reviewer can open the evidence.
- A note that the change was prepared with AI assistance (required by the
  contribution guidance; must **not** be a commit trailer).

Then `gh pr ready <n>` and report the PR URL. **Do not merge.** Marking it ready is
the hand-off; the human reviews and merges.

## Evaluation: two tiers, both required

|  | what it catches | cost | where it runs |
|---|---|---|---|
| **Tier 1** — distribution drift | the adapter stopped firing, or the logits moved | minutes | `tests/eval/`, inside the GPU leg |
| **Tier 2** — scored accuracy | the model got **worse** | hours | `evals.yaml` → Vela AppWrapper |

Both are needed and neither substitutes for the other. A *dead* adapter and a
*degraded* adapter are different failures, and tier 1 cannot see the second one: a
model that answers confidently and wrongly has a perfectly normal top-k
distribution.

### Tier 1 — distribution drift (`tests/eval/`)

A GPU suite dispatched as `--suite eval`. Four configs — `granite42_vllm`,
`granite5_vllm`, `sr_vllm`, `granite42_hf` — each reading its checkpoint from a
pre-staged cluster path (`GSW_EVAL_*_DIR`) and comparing against a **committed
baseline** in `tests/eval/baselines/`, which records top-k logprob distributions
for a fixed prompt set plus the vLLM version they were captured on. A config whose
env var is unset skips with a reason; it does not silently pass.

The baseline is the gate, so capture it on a **known-good** version — the current
ceiling, *before* porting:

```bash
uv run --group dev-vllm<old-ceiling> \
  .claude/skills/upgrade-vllm/scripts/record_eval_baseline.py --config all
```

Only re-record when the user explicitly accepts a quality change. Re-recording to
make a failing eval pass destroys the gate: a failure here is the signal the whole
upgrade was looking for. Report it and stop.

### Tier 2 — scored accuracy (`evals.yaml`)

3565 answerability rows per config against a fixed accuracy bar, on the Vela
cluster. Defined as data in `evals.yaml`; `render_eval_jobs.py` turns it into one
AppWrapper per config for the line under test.

```bash
export HF_TOKEN=...            # NOT committed; injected at render time
export GITHUB_COM_TOKEN=...    # needs read access to this repo
.claude/skills/upgrade-vllm/scripts/render_eval_jobs.py --vllm 0.31 --sha <pr-head-sha>
kubectl apply -f scratch/eval-jobs/
kubectl -n security get appwrappers | grep v031
```

Output lands in `scratch/eval-jobs/` (gitignored) because it contains the live
tokens. The renderer warns and writes `${HF_TOKEN}` placeholders if the vars are
unset, which produces jobs that fail at the clone — so set them.

The four configs and their bars:

| config | checkpoint | backend | bar | unique coverage |
|---|---|---|---|---|
| `granite42_dense` | `ans-4p2-3b-v2-lctx` | vLLM | 0.9052 | dense decoder, ChatML document routing |
| `granite5_lora` | `g5-20b-answerability-lora` | vLLM | 0.9196 (also 0.9168 HF) | sparse MoE + LoRA |
| `granite5_sr` | `g5-20b-answerability-sr` | vLLM | 0.8948 | SR dual-stream through the MoE runner |
| `granite5_sr_hf` | `g5-20b-answerability-sr` | HF | 0.8948 (±0.002) | the only scored cover for `hf/modeling_granite_switch.py` |

Reading the verdict each job prints:

- **MATCH** — `|accuracy - bar| <= tolerance` (0.006 for the vLLM configs, which
  absorbs the HF→vLLM gap; 0.002 for `granite5_sr_hf`, whose bar *is* the HF
  number so there is no gap to allow for).
- **DRIFT** — faithful band but off the bar. Investigate before merging. For
  `granite5_lora`, compare against **both** bars first: a move toward 0.9168 is
  convergence on the HF number, not a regression.
- **STACK BUG** — below 0.60. Not "slightly worse" but structurally broken: ~0.40
  is the unanswerable base rate, i.e. the model answering from the question alone.
  Means dropped documents, a dead adapter, or thinking mode left on. The rendered
  job's render gate is designed to catch this in seconds rather than hours, so a
  STACK BUG verdict that got past it is worth a careful look.

The last two pairs are deliberate: `granite5_sr` and `granite5_sr_hf` use the
**same checkpoint**, same render, same dtype, so the pair isolates the backend as
the only variable.

**To change an adapter, a checkpoint, a bar or the cluster, edit `evals.yaml` —
never the rendered YAML.** That file is the whole reason the four jobs are not four
34 KB copies drifting apart.
