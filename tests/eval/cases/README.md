<!-- SPDX-License-Identifier: Apache-2.0 -->
# Eval prompt set — populated per run, not committed

`prompts.json` belongs here but is **not committed**. It is the tier-1 drift
gate's input data, held by the internal `upgrade-vllm` maintainer tool and
committed onto its upgrade branch only for the shas it dispatches eval legs at.
See `../baselines/README.md` for why, and for what that means if you are reading
this on `main`.

The set is **fixed** rather than generated. The baselines record a distribution
per prompt per position, so changing it invalidates all of them — `compare()`
detects a changed prompt set and fails rather than comparing misaligned
positions, because a drift number measured over different prompts is not a drift
number.
