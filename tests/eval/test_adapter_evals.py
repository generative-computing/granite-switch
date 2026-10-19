# SPDX-License-Identifier: Apache-2.0
"""The four evaluation jobs: adapter output vs a committed baseline.

Run on the GPU cluster as its own suite::

    /gpu-test-eval                      # as a maintainer, on a PR
    pytest -m "eval and not deep" tests/eval/ -v -s --tb=short

Each config needs its checkpoint staged on the runner (``GSW_EVAL_*_DIR``) and a
baseline in ``tests/eval/baselines/``. The skip/fail split is deliberate:

* env var unset       -> SKIP. This runner does not have that checkpoint.
* env var set, no
  baseline             -> FAIL. The eval could have run and had nothing to compare
                          against. Skipping here would let a vLLM upgrade pass its
                          eval gate without ever evaluating anything.

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
        pytest.skip(f"{cfg.description}: set {envs} to a staged checkpoint to run it")

    baseline = core.load_baseline(cfg.name)
    assert baseline is not None, (
        f"{cfg.name}: checkpoint is staged at {model_dir} but "
        f"{core.baseline_path(cfg.name)} does not exist, so there is nothing to "
        "compare against. Record it from a KNOWN-GOOD vLLM line:\n"
        f"  uv run --group dev-vllm<known-good> "
        f".claude/skills/upgrade-vllm/scripts/record_eval_baseline.py "
        f"--config {cfg.name}"
    )

    current = core.capture(cfg, model_dir)
    failures, report = core.compare(baseline, current)

    print(f"\n[{cfg.name}] {cfg.description}")
    for line in report:
        print(line)

    assert not failures, f"{cfg.name} regressed against its baseline:\n" + "\n".join(
        f"  - {f}" for f in failures
    )
