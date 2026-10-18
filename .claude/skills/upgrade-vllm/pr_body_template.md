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

## Gate results

| gate | result | run |
|---|---|---|
| Pre-commit | | |
| CPU tests (3.11) | | |
| CPU tests (3.12) | | |
| GPU `vllmXX` (floor) | | |
| GPU `vllmZZ` (new ceiling) | | |
| drift eval `granite42_vllm` | | |
| drift eval `granite5_vllm` | | |
| drift eval `sr_vllm` | | |
| drift eval `granite42_hf` | | |

GPU legs were dispatched with `no_cache=true` — the test pod pins
`vllm/vllm-openai:latest`, so a cached pass would answer "what did this commit do"
rather than "is this still true today".

Lines in the middle of the range (0.XX+1 … 0.ZZ-1) are not per-run CI legs. They
were verified out of band for this PR: <result>. The two legs are the ends of the
range, where breakage actually lands — a deleted API shows up at the ceiling and an
API that does not exist yet shows up at the floor.

## Scored accuracy evals

3565 answerability rows per config on the Vela cluster, against the bars in
`.claude/skills/upgrade-vllm/evals.yaml`. These answer a question the drift evals
above cannot: *did the model get worse.* A model that answers confidently and
wrongly has a perfectly normal top-k distribution.

| config | backend | bar | accuracy | delta | verdict |
|---|---|---|---|---|---|
| `granite42_dense` | vLLM | 0.9052 | | | |
| `granite5_lora` | vLLM | 0.9196 | | | |
| `granite5_sr` | vLLM | 0.8948 | | | |
| `granite5_sr_hf` | HF | 0.8948 | | | |

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
