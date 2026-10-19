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

From a known-good vLLM line, with the checkpoint staged:

```bash
uv run --group dev-vllm30 \
  .claude/skills/upgrade-vllm/scripts/record_eval_baseline.py --config all
```

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
