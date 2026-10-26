<!-- SPDX-License-Identifier: Apache-2.0 -->
# Eval baselines

One `<config>.json` per eval config, committed. Each records, for a fixed prompt
set, the top-k logprob distribution at every greedily generated position — plus the
vLLM/torch/transformers versions it was captured on, the adapter it evaluated, and
a fingerprint of the checkpoint.

A baseline is a **signed-off result**. `tests/eval/test_adapter_evals.py` compares
the current run against it and fails on drift beyond the thresholds stored in the
file.

## Recording one

Record on the GPU runner that also does the checking, via the workflow's
`eval-record` suite — not locally and not on another cluster. A baseline captured
elsewhere was not reproducible here: it missed by a systematic ~0.004 mean JSD on
both vLLM configs, identically on the 0.26 and 0.30 legs. Same-environment
recording makes the baseline reproducible by construction, so later movement is
attributable to the vLLM line and nothing else.

```bash
gh workflow run gpu-tests.yaml --ref <branch> \
  -f suite=eval-record -f sha=<sha> -f no_cache=true
```

The run prints each baseline JSON into the pod log for a human to extract and
commit; it writes nothing to the repo. `eval-record` is deliberately
non-cacheable, since the log *is* the artifact. See
`tests/eval/test_record_baselines.py`.

Baselines must be captured **before** a port, on the version you already trust. A
baseline captured on the version under test proves nothing.

## When re-recording is legitimate

- The checkpoint was intentionally re-composed or replaced (the fingerprint check
  will have failed first and told you so).
- The prompt set changed deliberately.
- A quality change was reviewed and **explicitly accepted** by a human.

## When it is not

A failing eval on a new vLLM line. That failure is the entire reason this gate
exists. Re-recording to clear it silently adopts the regression and leaves the
gate permanently blind to it. Report it instead.

The `min_adapter_effect_jsd` check is the one to be most careful with: it fails
when the adapter stops measurably changing the output, which means the adapter is
not being applied at all. vLLM 0.29 did exactly this to the ASR path by deleting
the hook its override sat on — nothing raised, nothing went red, the feature just
stopped. Never paper over that one.
