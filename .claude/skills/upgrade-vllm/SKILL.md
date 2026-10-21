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
   were added" means 0.31 and 0.32 each have their own green evidence. The GPU
   workflow gives at most two legs per commit, so this costs **one commit per
   added line** (Phase 5); it is not optional and there is no shortcut.
5. **A PR body with per-version proof tables.** One column per added line, a ✅
   per gate that actually ran and passed on that line, and a separate table giving
   the regression/eval numbers **against the existing baseline** for each added
   line. A gate that did not run is `—`, never ✅.

The point of 4 and 5 together: a reviewer must be able to see, per version, that
each gate ran — not take "the suite is green" on faith.

**A green run is not proof a gate ran.** pytest exits 0 when every test skips, the
pod then prints `ALL GPU TESTS PASSED`, and the workflow posts success. Measured,
not hypothesised: run `39299824463` collected 4 eval tests, skipped all 4 because
no checkpoint was staged, and reported PASSED in 0.03s. A ✅ written from that run
would be a false claim of coverage. `gpu_dispatch.sh` now fails on a green run
whose log has no `N passed`, so the only ✅ you can write is one a real test
earned. If a leg comes back vacuous, the cell is `—` plus a note saying why.

## Operating rules

These are not suggestions. Each one is a failure this repo has actually hit.

1. **Gate on capability, never on a version number.** Use `hasattr`/`try: import`
   to detect the API shape. A `vllm.__version__` comparison is wrong the moment
   upstream renames something, and it does not state the actual requirement. There
   is currently **zero** version-number branching in `src/` — keep it that way.
2. **Never delete a dependency group.** Every declared `vllmNN`/`dev-vllmNN` group
   is a leg some commit's `dev-latest` can point at, and `derive_legs.py` resolves
   groups against the *tested* commit. See REFERENCE.md §4. Deleting one makes that
   leg's `uv sync --group X` fail before pytest starts, producing a sub-1.1-KB log
   that looks like a test failure.
3. **Relock with uv 0.8.4 exactly** — the `rev` of the `uv-lock` pre-commit hook.
   A newer uv writes a different lockfile revision and the hook fails CI.
4. **Every commit needs `git commit -s`.** The DCO workflow blocks the PR otherwise.
   Do **not** add a `Co-authored-by: Claude` trailer; disclose AI assistance in the
   PR body instead. (Project rule — see CLAUDE.md and REFERENCE.md §8.)
5. **Never `--no-verify`.** Fix what the hook complains about.
6. **Dispatch at the PR branch ref AND an explicit `--sha`.** `--ref <branch>` runs
   *that branch's* workflow file (the `/gpu-test` comment path cannot: it hardcodes
   `--ref <default branch>`, which is why this skill bypasses it). `--sha` picks the
   commit, and since each added line is evidenced by its own commit (Phase 5),
   omitting it tests the branch tip N times instead of N lines once each.
7. **Always pass `no_cache=true`.** The GPU result cache is keyed on sha+leg, and
   the test pod pins `vllm/vllm-openai:latest`, so a cached pass answers "what did
   this commit do" when the question is "is this still true today".
8. **Open the PR as a draft** and only mark it ready when every gate is green.
   A human seeing a ready PR assumes the gates passed.
9. **Do not ask the user to approve the routine steps.** Branching, editing
   `src/`, relocking, walking `dev-latest` per line, committing, pushing, dispatching GPU
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

**Branch from main without ever checking main out.** `git checkout main` would
delete this skill from the working tree on any clone where the skill is not yet
committed to main — and a deleted skill has no Phase 3 onward left to read, so
the run ends mid-flight with the branch already created. Create the branch at
`origin/main` directly, which never puts main itself in the tree:

```bash
git fetch origin main
git checkout -b feature/vllm-<floor>-to-<new-ceiling> origin/main
```

Then confirm the skill survived, because the same hazard applies to the new
branch if it is a faithful copy of a main that lacks the skill:

```bash
test -f .claude/skills/upgrade-vllm/SKILL.md || echo "SKILL GONE"
```

If it is gone, the skill is not on `origin/main` yet. Stop and report that
rather than improvising — an unattended run cannot proceed on instructions it
can no longer read.

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
flag. The other flags are `--floor` (which *drops* support below it),
`--point-latest` (below), and `--check [--final]`.

This edits `pyproject.toml` to add each new line's `vllmNN` group, its `dev-vllmNN`
group, widen the `vllm` extra's upper bound, point `dev-latest` at the lowest added
line, and **regenerate the entire `[tool.uv] conflicts` table** from the set of
declared lines. Regenerating is the point: the table needs every pair of groups
that would pull two lines into one venv, it grows quadratically, and a *missing*
pair does not fail loudly — it lets `uv sync` resolve two vLLM wheels into one
environment. The script never removes a group.

Then relock and move the CI ceiling:

```bash
uvx uv@0.8.4 lock   # NOT bare `uv lock`
```

Always the pinned form, even when the local `uv` happens to be 0.8.4 — the local
version is not guaranteed and a newer one writes a different lockfile revision,
which the `uv-lock` pre-commit hook then rewrites and fails on. Take the version
from `.pre-commit-config.yaml`'s `uv-pre-commit` rev rather than this line, so a
hook bump cannot leave the skill relocking with the wrong one.

#### There is no GPU matrix to edit. Per-line evidence comes from one commit per line.

This is the single most important mechanical fact in the skill, and the thing most
likely to be remembered wrong, because an older version of this document said
"widen the matrix" — which is **impossible**.

`.github/workflows/gpu-tests.yaml` does not contain its legs. A `legs` job runs
`/opt/gsw/derive_legs.py`, which is **baked into the runner image and not editable
from this repository**. It reads the *tested commit's own* `pyproject.toml` and
emits **at most two legs**: one from the `dev` group (the floor) and one from
`dev-latest` (the ceiling alias). A fifth leg cannot be added by any edit a PR can
make.

So "every added line gets its own green evidence" (contract item 4) is bought a
different way: **make each added line be `dev-latest` at some commit, and dispatch
at that commit's sha.** Two properties make this work rather than merely sound
clever:

- `derive_legs.py` reads the *tested* commit, so a commit whose `dev-latest` points
  at 0.31 produces a 0.31 leg, regardless of what `main` says.
- Workflow concurrency is `group: gpu-tests-${{ inputs.sha }}-${{ inputs.suite }}`
  — **keyed on the sha**. Dispatches at different commits therefore run in
  **parallel** and do not cancel one another. If it were keyed on the branch, each
  new dispatch would kill the previous line's run and per-line coverage would cost
  N sequential rounds instead of one.

`plumb_pyproject.py` implements the walk. `--add` parks `dev-latest` on the
**lowest** added line; one `--point-latest` commit per remaining line walks it up;
the last one lands it on the true ceiling, which is also the correct state to
merge:

```bash
# commit A: declares every line, dev-latest -> 0.31  (legs: floor + 0.31)
.claude/skills/upgrade-vllm/scripts/plumb_pyproject.py --add 0.31 0.32
uvx uv@0.8.4 lock && git commit -s -a && git push

# commit B: dev-latest -> 0.32  (legs: floor + 0.32)
.claude/skills/upgrade-vllm/scripts/plumb_pyproject.py --point-latest 0.32
uvx uv@0.8.4 lock && git commit -s -a && git push

# the state that merges: alias on the ceiling
.claude/skills/upgrade-vllm/scripts/plumb_pyproject.py --check --final
```

Every line in the gap gets a leg, the floor gets one on every single dispatch (it
is always the lower leg), and the old ceiling keeps its coverage because it is
still a declared line that one of these commits points at. **Dispatch at each
commit's sha as you create it** — Phase 8 does the collecting.

`--check --final` is the merge gate: it fails while `dev-latest` is parked on an
intermediate line. Merging in that state would leave the top supported line
permanently untested by CI, which is exactly the failure the per-line walk exists
to prevent.

Two things `--point-latest` does that are easy to get wrong by hand, and which is
why it exists rather than a note saying "edit the alias": `dev-latest` is a *third
group name* for whichever line it points at, so uv's `conflicts` table must move
with it (otherwise `uv sync --group vllm31 --group dev-latest` resolves two vLLM
wheels into one venv), and the prose comment above the alias names the line and
goes stale.

Do **not** delete a group when the alias moves off it (rule 2). Update the
`[dependency-groups]` and `vllm`-extra comments to name the new range — a stale
comment claiming the old range is a review failure in this repo.

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

Dispatch is **per line × per suite**, because each line is a different commit (see
Phase 5). Collect the shas first, then fan out — concurrency is sha-keyed, so these
do not cancel one another and the whole grid runs in parallel:

```bash
# the sha of each per-line commit, in the order Phase 5 created them
SHA_031=<sha of the --add commit>        # legs: floor + 0.31
SHA_032=<sha of the --point-latest 0.32 commit>   # legs: floor + 0.32

BRANCH=feature/vllm-<floor>-to-<new-ceiling>
for SHA in "$SHA_031" "$SHA_032"; do
  for SUITE in full audio eval; do
    .claude/skills/upgrade-vllm/scripts/gpu_dispatch.sh \
        --ref "$BRANCH" --sha "$SHA" --suite "$SUITE" --wait &
  done
done
wait
```

**Both flags, and they mean different things.** `--ref` picks which *workflow file*
runs and is required; `--sha` picks which *commit* the cluster checks out and
defaults to `origin/<ref>`'s head. Letting it default is the trap: every dispatch
would then test the branch tip, so each intermediate line would silently report the
ceiling's result — a table full of ✅ that proves the top line N times and the
others never.

The leg label in each result is the receipt that this went right. Two runs reporting
the same label (`vllm32` twice) means the shas were wrong, not that the lines agree.

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

**The measurement is teacher-forced.** Both the baseline and the check push the
BASELINE's recorded `continuations` through the model and read the distribution at
each position, so position *i* always sits on a byte-identical prefix. Only
recording generates. Do not "simplify" this into a greedy rollout: one flipped
near-tie argmax desynchronizes the prefix and every later position then compares
unrelated contexts. If `mean_jsd` and `mean_jaccard_dist` both land on exactly
`(N − p)/N` for some integer p, that is trajectory desynchronization at position
p, not a logit regression — see `tests/eval/_eval_core.py`'s docstring.

The baseline is the gate, so capture it on a **known-good** version — the current
ceiling, *before* porting — and capture it **on the GPU runner**, the same
environment that checks it:

```bash
gh workflow run gpu-tests.yaml --ref <branch> -f suite=eval-record -f sha=<sha>
```

Then lift each JSON out of the run's `gpu-tests.log` artifact between its
`--8<-- <path>` and `-->8--` markers, write it to that path, and commit. The job
writes nothing into the repo on purpose: a job that could rewrite a baseline in
place is one bad dispatch away from erasing the thing being gated.

Recording elsewhere has been tried and does not hold. A Vela-recorded baseline was
off by a systematic ~0.004 mean JSD on the CI runner, identically on the 0.26 and
0.30 legs — so not version drift, and not torch either (the HF config reproduced
across torch 2.11 vs 2.13 at exactly 0.000000). Recording where the check runs
makes the baseline reproducible by construction, so later movement is attributable
to the one thing that changed: the vLLM line. `scripts/record_eval_baseline.py`
still exists for a dev box with checkpoints staged, but it is not the sanctioned
path.

Only re-record when the user explicitly accepts a quality change. Re-recording to
make a failing eval pass destroys the gate: a failure here is the signal the whole
upgrade was looking for. Report it and stop. Likewise never widen a threshold. If
a failure reproduces identically across vLLM lines — *including the line the
baseline was recorded on* — it is a defect in the measurement, not stale data.

Two further invariants the gate depends on:

- **The adapter is pinned by NAME** (`_eval_core.EVAL_ADAPTERS`), never by index.
  The composer walks `rglob` in filesystem order without sorting, so "adapter 0"
  of a composed checkpoint is not the same adapter on two machines.
- **Prefix caching is off** in the capture path, for reproducibility rather than
  speed. Leave it off.

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
