<!-- SPDX-License-Identifier: Apache-2.0 -->
<!--
PR body template for the upgrade-vllm skill. Fill every section; delete none.

The "Support dropped" section is mandatory even when nothing was dropped. An
absent section reads as "nothing was dropped", and a silently narrowed range is
how a downstream pin breaks weeks later with no record of the decision.
-->

## Supported vLLM range

| | before | after |
|---|---|---|
| `[vllm]` extra | `>=0.XX.0,<0.YY.0` | `>=0.XX.0,<0.ZZ.0` |
| lines claimed | 0.XX … 0.YY | 0.XX … 0.ZZ |
| CI GPU legs | `vllmXX` / `vllmYY` | `vllmXX` / `vllmZZ` |

New lines in this PR: **0.ZZ** (and 0.ZZ-1 …).

## Support dropped

<!-- EITHER: -->
**None — the range only widened.** Every line previously claimed is still claimed
and still resolvable.

<!-- OR, if a line left the range: -->
~~**0.XX is no longer supported.**~~ `<API>` is required by the port and does not
exist before 0.NN; capability-gating it would mean maintaining a second code path
for `<reason it is not worth it>`. The `[vllm]` extra's lower bound is raised to
`>=0.NN.0` to match.

**Action for maintainers:** anything pinning granite-switch against vLLM 0.XX must
move. The `vllmXX` dependency *group* is intentionally kept (a group named by
main's GPU matrix cannot be deleted without killing that leg before pytest starts),
so `uv sync --group vllmXX` still works — it is simply no longer a supported
configuration.

## Dependency ceilings touched

- `transformers`: `<=5.17.0` → `<=X.Y.Z` — <why the new vLLM requires it>
- `torch`: unchanged / `>=X.Y`

## What upstream changed, and how it is handled

One subsection per finding. Each names the upstream change, the symptom here, and
the gate.

### <symbol or subsystem>

**Upstream:** <what changed, with the upstream commit or PR>.

**Here:** <how it manifested — the error, or "silently stopped working and no test
noticed">.

**Fix:** <what changed in this repo>. Gated on `hasattr(...)` rather than a version
comparison, so it states the actual requirement and survives the next rename.

**Test:** <which test covers it>. <If the failure mode was silent, say that the
test asserts the effect rather than the call.>

## Per-version coverage

One column per vLLM line this PR adds, plus the floor and previous ceiling as
controls. **✅ means that gate ran and passed on that line**; `—` means it does not
apply. No cell is ticked by inference from its neighbours.

<!--
Add or remove version columns to match --add. Fill ONLY from an actual run; an
unticked cell is honest and a wrongly ticked one is the failure this table exists
to prevent.
-->

| gate | 0.XX (floor) | 0.YY (prev ceiling) | **0.ZZ-1 (new)** | **0.ZZ (new)** |
|---|---|---|---|---|
| Pre-commit | ✅¹ | ✅¹ | ✅¹ | ✅¹ |
| CPU tests (3.11) | ✅¹ | ✅¹ | ✅¹ | ✅¹ |
| CPU tests (3.12) | ✅¹ | ✅¹ | ✅¹ | ✅¹ |
| GPU `full` (unit, composer, hf, vllm, integration) | | | | |
| GPU `audio` | | | | |
| Drift eval `granite42_vllm` | | | | |
| Drift eval `granite5_vllm` | | | | |
| Drift eval `sr_vllm` | | | | |
| Drift eval `granite42_hf` | | | | |

¹ The CPU jobs install the `dev` group and do not vary by vLLM line — one run
covers every column. They are shown per column so the table has no silent holes,
not because they ran four times.

Run links: <one per ✅, or a single link per matrix run with the leg names>.

GPU legs were dispatched with `no_cache=true` — the test pod pins
`vllm/vllm-openai:latest`, so a cached pass would answer "what did this commit do"
rather than "is this still true today".

**Matrix note.** This PR's GPU matrix carried a leg per added line so each column
above has its own evidence. The final commit narrows the committed matrix back to
the two ends of the new range (floor + new ceiling), which is where breakage
actually lands: an API deleted upstream fails at the ceiling, and an API that does
not exist yet fails at the floor. A middle line is bracketed by two tested lines.
The middle lines' evidence is the run linked above, not a future CI run.

## Scored accuracy evals

3565 answerability rows per config on the Vela cluster, against the bars in
`.claude/skills/upgrade-vllm/evals.yaml`. These answer a question the drift evals
above cannot: *did the model get worse.* A model that answers confidently and
wrongly has a perfectly normal top-k distribution.

Reported **per added line**, against the same unchanged bars — so a reviewer can
see that 0.ZZ specifically did not regress, rather than that "the evals passed".

| config | backend | bar | 0.ZZ-1 accuracy | Δ | 0.ZZ accuracy | Δ | verdict |
|---|---|---|---|---|---|---|---|
| `granite42_dense` | vLLM | 0.9052 | | | | | |
| `granite5_lora` | vLLM | 0.9196 | | | | | |
| `granite5_sr` | vLLM | 0.8948 | | | | | |
| `granite5_sr_hf` | HF | 0.8948 | | | | | |

Tolerance is ±0.006 for the vLLM configs, which absorbs the HF→vLLM gap for bars
measured on the HF path, and ±0.002 for `granite5_sr_hf`, whose bar *is* the HF
number so there is no gap to allow for.

`granite5_lora` has a second bar at 0.9168 (the switch-HF number). A move toward it
is convergence on HF, not a regression — compare against both before calling drift.

`granite5_sr` and `granite5_sr_hf` run the **same checkpoint** through the two
backends, so the pair isolates the backend as the only variable; `granite5_sr_hf`
is also the only scored cover for `hf/modeling_granite_switch.py`, which a break
confined to the HF path could otherwise hide behind three green vLLM legs.

## Drift eval baselines

Baselines in `tests/eval/baselines/` were recorded on vLLM 0.YY (the previously
supported ceiling) and are **unchanged** by this PR. <If any baseline was
re-recorded, name it, say why, and name the human who accepted the quality change.>

---

Prepared with AI assistance (Claude Code). Reviewed by the PR author before
submission.
