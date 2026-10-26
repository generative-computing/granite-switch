# SPDX-License-Identifier: Apache-2.0
"""Record eval baselines ON THE GPU RUNNER, printing them for extraction.

Not a test -- a recording job that happens to be reachable through pytest,
because pytest is the only thing `/opt/gsw/submit_and_poll.sh` runs on the pod
and that script is baked into the runner image. Dispatch it with::

    gh workflow run gpu-tests.yaml --ref <branch> -f suite=eval-record -f sha=<sha>

then lift each JSON out of the run's `gpu-tests.log` artifact between its
``--8<-- <path>`` and ``-->8--`` markers, write it to that path, and commit.

WHY RECORD HERE rather than on a dev pod. A baseline is only a reference if the
environment that checks it can reproduce it. Recording on a Vela pod produced a
baseline the CI runner missed by a systematic ~0.004 mean JSD on both vLLM
configs -- identical on the 0.26 and 0.30 legs, so not version drift, and not
torch either (the HF config reproduced that same baseline at exactly 0.000000
across torch 2.11 vs 2.13). Something else in the vLLM capture path differs
between the two machines. Rather than keep guessing at it, record where the
check runs: then the baseline is reproducible BY CONSTRUCTION, and any later
movement is attributable to the one thing that changed, the vLLM line.

It writes nothing into the repo. Printing and committing by hand is deliberate:
a baseline is the gate's reference, and a job that could rewrite it in place is
one bad dispatch away from erasing the thing being gated.

Marked `deep` as well as `eval_record` so pyproject's default `-m "not deep"`
keeps it out of every ordinary selection, including a bare `pytest tests/`. Only
an explicit `-m eval_record` reaches it.
"""

import json

import pytest

from tests.eval import _eval_core as core

pytestmark = [
    pytest.mark.eval_record,
    pytest.mark.deep,
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.requires_model,
]


@pytest.mark.parametrize(
    "config_name", list(core.EVAL_CONFIGS), ids=list(core.EVAL_CONFIGS)
)
def test_record_baseline(config_name):
    cfg = core.EVAL_CONFIGS[config_name]
    model_dir = core.checkpoint_dir(cfg)
    if model_dir is None:
        pytest.skip(f"{cfg.description}: no checkpoint and no compose recipe")

    data = core.capture(cfg, model_dir)

    # Same formatting record_eval_baseline.py writes, so a baseline captured here
    # and one captured there are byte-comparable rather than differing by layout.
    text = json.dumps(data, indent=1, sort_keys=True)
    path = core.baseline_path(cfg.name).relative_to(core.HERE.parent.parent)

    print(f"\n--8<-- {path}")
    print(text)
    print("-->8--")

    # Assert the capture is well-formed, so a broken recording fails the job
    # instead of printing something unusable that is noticed only at commit time.
    n = len(data["prompts"])
    assert len(data["continuations"]) == n
    assert len(data["adapter_dists"]) == n
    assert len(data["base_dists"]) == n
    for i in range(n):
        steps = len(data["continuations"][i])
        assert steps > 0, f"prompt {i} produced no continuation"
        assert len(data["adapter_dists"][i]) == steps
        assert len(data["base_dists"][i]) == steps
