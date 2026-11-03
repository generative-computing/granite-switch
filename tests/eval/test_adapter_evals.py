# SPDX-License-Identifier: Apache-2.0
"""The four evaluation jobs: adapter output vs a committed baseline.

Run on the GPU cluster as its own suite::

    /gpu-test-eval                      # as a maintainer, on a PR
    pytest -m "eval and not deep" tests/eval/ -v -s --tb=short

Each config needs a checkpoint and a baseline in ``tests/eval/baselines/``.

**The baselines and ``cases/prompts.json`` are not committed to this repository.**
They are the comparison data for an internal maintainer tool (the ``upgrade-vllm``
skill), which commits them onto its upgrade branch for the shas it dispatches eval
legs at, and removes them again before the PR goes ready. The GPU pod does no
checkout of its own -- it fetches this repo at the dispatched sha as a tarball --
so that is the only way the data can reach it. Consequence, stated so it is not
rediscovered as a bug: dispatching ``suite=eval`` on ``main`` ERRORS rather than
skipping, because the prompt set is read unconditionally. The gate is reachable
only through an upgrade run.

The checkpoint comes from ``GSW_EVAL_*_DIR`` if staged, and is otherwise COMPOSED from
public `ibm-granite` repos into the e2e cache (``_eval_core.COMPOSE_RECIPES``) --
because requiring the env var meant this suite could only ever skip: nothing sets
it on the GPU pod, and an all-skipped suite exits 0 and reports PASSED.

The skip/fail split is deliberate:

* no checkpoint AND
  no compose recipe    -> SKIP. Genuinely cannot run here (e.g. `sr_vllm`, whose
                          SR adapters are not public).
* checkpoint available,
  no baseline          -> FAIL. The eval could have run and had nothing to compare
                          against. Skipping here would let a vLLM upgrade pass its
                          eval gate without ever evaluating anything.
* compose attempted
  and failed           -> FAIL (raises). A broken recipe is a real break, not an
                          absent checkpoint.

Re-recording a baseline to clear a failure destroys the gate. If an eval fails on a
new vLLM line, that failure IS the result the upgrade was looking for.
"""

import pytest

from tests.eval import _eval_core as core

pytestmark = [
    pytest.mark.eval,
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.requires_model,
]


@pytest.mark.parametrize(
    "config_name", list(core.EVAL_CONFIGS), ids=list(core.EVAL_CONFIGS)
)
def test_adapter_eval(config_name):
    cfg = core.EVAL_CONFIGS[config_name]
    model_dir = core.checkpoint_dir(cfg)
    if model_dir is None:
        envs = " or ".join(filter(None, (cfg.env, cfg.fallback_env)))
        pytest.skip(
            f"{cfg.description}: no checkpoint and no compose recipe. "
            f"Set {envs} to a staged checkpoint to run it."
        )

    baseline = core.load_baseline(cfg.name)
    assert baseline is not None, (
        f"{cfg.name}: checkpoint is staged at {model_dir} but "
        f"{core.baseline_path(cfg.name)} does not exist, so there is nothing to "
        "compare against. Record it from a KNOWN-GOOD vLLM line, on the GPU "
        "runner that does the checking -- a baseline captured in a different "
        "environment missed by a systematic ~0.004 mean JSD:\n"
        "  gh workflow run gpu-tests.yaml --ref <branch> "
        "-f suite=eval-record -f sha=<sha> -f no_cache=true\n"
        f"The run prints every baseline into its log. The {cfg.name} baseline is "
        "NOT committed here: it belongs to the internal upgrade-vllm skill, which "
        "injects it onto its upgrade branch for the shas it dispatches eval legs "
        "at and removes it before the PR goes ready. So seeing this on main is "
        "expected, and the fix is to run the eval through that skill rather than "
        "to commit a baseline into this repository."
    )

    # The baseline's continuations, not freshly generated ones: the measurement is
    # teacher-forced so every compared position has a byte-identical prefix. See
    # _eval_core's module docstring for the greedy-rollout failure that forced this.
    current = core.capture(cfg, model_dir, continuations=baseline.get("continuations"))
    failures, report = core.compare(baseline, current)

    print(f"\n[{cfg.name}] {cfg.description}")
    for line in report:
        print(line)

    assert not failures, f"{cfg.name} regressed against its baseline:\n" + "\n".join(
        f"  - {f}" for f in failures
    )
