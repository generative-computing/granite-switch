<!-- SPDX-License-Identifier: Apache-2.0 -->
# Eval baselines — populated per run, not committed

This directory is **empty on `main` by design.** The tier-1 drift gate's code is
here (`tests/eval/`); its comparison data is not.

One `<config>.json` per eval config holds the *reference half* of the gate: for a
fixed prompt set, the top-k logprob distribution at every teacher-forced position,
plus the vLLM/torch/transformers versions it was captured on, the adapter it
evaluated, a fingerprint of the checkpoint, and the thresholds to compare against.
`test_adapter_evals.py` fails on drift beyond those thresholds.

Those files, and `../cases/prompts.json`, live with the internal `upgrade-vllm`
maintainer tool, which commits them onto its upgrade branch for the shas it
dispatches eval legs at and removes them again before its PR goes ready.

## Why it works that way

The gate is maintainer-only to begin with: `/opt/gsw/check_role.sh` is baked into
the runner image and authorizes only **maintain or admin**, so `-f suite=eval`
and `-f suite=eval-record` are not dispatchable by an outside contributor. The
data therefore served one internal tool and was moved to live with it.

It still has to be committed here *temporarily*, because the GPU pod does no
checkout of its own: the workflow points `/opt/gsw/submit_and_poll.sh` at this
repository and it fetches the dispatched sha as a tarball before running pytest.
There is no side channel. So an eval leg can only ever prove the tree at its own
sha — and the sha carrying a green eval leg contains this data while the sha that
merges does not. Any PR relying on eval evidence has to say so in its body.

## What that means if you are reading this on `main`

**Dispatching `suite=eval` here errors rather than skipping.** `prompts()` reads
the prompt set unconditionally and raises a `FileNotFoundError` naming this
situation. That is deliberate — an eval that quietly skipped would report PASSED
while measuring nothing, which is the exact failure mode this suite is built to
avoid.

The corollary, stated plainly: the drift gate does not guard ordinary PRs. An
adapter regression introduced by a non-upgrade change surfaces at the next vLLM
upgrade, not on the PR that caused it.

## Recording a baseline

Record on the GPU runner that also does the checking — not locally, not on another
cluster. A baseline captured elsewhere was not reproducible here: it missed by a
systematic ~0.004 mean JSD on both vLLM configs, identically on the 0.26 and 0.30
legs. Same-environment recording makes the baseline reproducible by construction,
so later movement is attributable to the vLLM line and nothing else.

```bash
gh workflow run gpu-tests.yaml --ref <branch> \
  -f suite=eval-record -f sha=<sha> -f no_cache=true
```

The run prints each baseline JSON into the pod log for extraction; it writes
nothing to the repo. `eval-record` is deliberately non-cacheable, since the log
*is* the artifact. See `tests/eval/test_record_baselines.py`. Extracted baselines
go to the `upgrade-vllm` tool's asset directory, **not** into this one.

Capture baselines **before** a port, on the version you already trust. One
captured on the version under test proves nothing.

## Re-recording

Legitimate when the checkpoint was intentionally re-composed or replaced (the
fingerprint check fails first and says so), when the prompt set changed
deliberately, or when a quality change was reviewed and **explicitly accepted** by
a human.

Not legitimate for a failing eval on a new vLLM line. That failure is the entire
reason the gate exists; re-recording to clear it silently adopts the regression
and leaves the gate blind to it.

`min_adapter_effect_jsd` is the check to be most careful with: it fails when the
adapter stops measurably changing the output, i.e. when the adapter is not being
applied at all. vLLM 0.29 did exactly this to the ASR path by deleting the hook
its override sat on — nothing raised, nothing went red, the feature just stopped.
Never paper over that one.
