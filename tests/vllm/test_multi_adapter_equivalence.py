# SPDX-License-Identifier: Apache-2.0
"""Multi-adapter routing equivalence under vLLM — LoRA/aLoRA and Shadow-Residual.

test_generation_equivalence.py pins a ONE-adapter zero-delta checkpoint against
upstream. This module pins what only exists once a checkpoint carries SEVERAL
adapters that are really trained, for both decoder tiers:

``passthrough``  no control token fired == the upstream base model, even though
                 the checkpoint's adapters carry real weights (adapter id 0 must
                 mean "no delta").
``distinct``     anti-vacuity: each adapter changes the distribution versus base,
                 and no two adapters agree. Without it the equality gates below
                 would pass on adapters that did nothing.
``slots``        the same adapter composed at a DIFFERENT slot produces the same
                 output. Two checkpoints hold the same adapters in opposite
                 order, so every per-slot kernel table — remap tables, the
                 rank-ordered tier assignment, the per-tile bitmasks, the
                 control-token ids — is permuted between them while the
                 mathematics is not.
``batch``        a prompt's output does not depend on what else is in the batch.

Every capture is taken twice, eager and compiled: ``GraniteSwitchModel`` is
wrapped in ``@support_torch_compile``, and the routing metadata is rebuilt each
forward on a shared ctx object, so a graph that specialized on a stale shape or
length is exactly the failure this suite exists for. ``slots`` and ``batch`` gate
both modes, but with different budgets on each: vLLM 0.20 does not reproduce a
compiled run across processes, and ``slots`` captures its two sides in two
processes, so its compiled arm is sized against that measured variance while the
eager arm -- where the slot-independence claim is actually established -- stays
near-exact. The worker's ``SLOTS_COMPILED_*`` comments carry the measurements.

``slots`` is the regression test for the lifted SR restriction: the vLLM SR
decoder used to reject ``num_adapters > 1`` outright, and this is what has to
hold for that rejection to have been unnecessary. Its CPU counterpart,
tests/composer/test_multi_adapter_compose.py, establishes that the composed
weights are slot-independent, so a failure here is a runtime routing bug rather
than a compose artifact.

Adapters and prompts
--------------------
Four adapters, at three different ranks, one covering the MLP and three
attention-only, two of each technology. The compose goes through the
``compose_granite_switch`` CLI, so the checkpoint carries a configured tokenizer
and chat template, and every prompt is rendered by
``apply_chat_template(..., adapter_name=...)``. Each adapter's control token
therefore lands where its own technology puts it — sequence start for LoRA, the
invocation point for aLoRA, the generation-prompt anchor for Shadow Residual —
which is exactly the difference between the technologies at runtime, and it is
not hand-placed here.

Every adapter is generated from a profile fitted to a published granitelib adapter
— delta norm relative to the base weight, singular-value spectrum and A/B asymmetry
all reproduced (tests/shared/synthetic_adapters.py, pinned by
tests/unit/test_synthetic_adapters.py). Nothing is downloaded but the base model,
and no base needs adapters published against it, so **any** base works — which is
what lets this run on granite-4.2-3b, where none exist. Rank and module coverage
are chosen rather than inherited, which is how a specific kernel rank tier gets
reached.

The profiles were fitted on granite-4.1-3b, but they are expressed relative to each
layer's base weight, and the aLoRA invocation sequence and SR activation anchor are
read off the base's own tokenizer rather than hardcoded — so both Granite template
families (4.0/4.1 role markers and 4.2 ChatML) work without special-casing.

What this cannot support is any claim about adapter *quality*; these tests make
none. It measures where an adapter's delta is routed, not whether the delta is
good.

Each step runs in its own subprocess so only one vLLM model is on GPU at a time
(vLLM does not free GPU memory on ``del``). Per (base model, flavor): one adapter
generation, two composes, two case renders, five vLLM loads (upstream, then both
slot orders eager and compiled). Two bases x two flavors is therefore 20 test cases
and roughly 35 minutes; narrow it with the env vars below.

Markers: requires_model (downloads a base model and adapters), consistent with
test_generation_equivalence.py.

Env:
  MULTI_ADAPTER_BASE_MODELS  comma-separated base checkpoints (default
                             granite-4.1-3b and granite-4.2-3b, one per template
                             family). Each must be dense and all-attention for the
                             ``sr`` flavor — the SR decoder has no routed-expert or
                             mamba path.
  MULTI_ADAPTER_FLAVORS      comma-separated subset of lora,sr (default both).
  and the gates/knobs documented in _multi_adapter_equivalence_worker.py.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

WORKER = Path(__file__).parent / "_multi_adapter_equivalence_worker.py"
#: One base per Granite chat-template family: 4.1-3b renders the
#: ``<|start_of_role|>`` role markers, 4.2-3b the ChatML ``<|im_start|>`` form. Both
#: are dense and all-attention, so both serve the SR decoder as well as the LoRA
#: one, and they are the same shape (40 layers, 2560 hidden, 40/8 heads) so a
#: difference between them is the template rather than the geometry.
BASE_MODELS = [
    m.strip()
    for m in os.environ.get(
        "MULTI_ADAPTER_BASE_MODELS",
        "ibm-granite/granite-4.1-3b,ibm-granite/granite-4.2-3b",
    ).split(",")
    if m.strip()
]
FLAVORS = [
    f.strip()
    for f in os.environ.get("MULTI_ADAPTER_FLAVORS", "lora,sr").split(",")
    if f.strip()
]
PREPARE_TIMEOUT = 1800  # adapter downloads
BUILD_TIMEOUT = 3600  # base-model download + a multi-adapter compose
CASES_TIMEOUT = 600
RUN_TIMEOUT = 2400  # vLLM load + one scored generate per case
COMPARE_TIMEOUT = 300

CASES = [(base, flavor) for base in BASE_MODELS for flavor in FLAVORS]


def _run_step(step_name, *cmd_args, timeout):
    """Run one worker step as a subprocess and assert it succeeded."""
    cmd = [sys.executable, str(WORKER), *cmd_args]
    print(
        f"\n{'=' * 70}\n  Step: {step_name}\n  {' '.join(str(c) for c in cmd)}\n{'=' * 70}"
    )
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.stdout:
        print(result.stdout[-8000:])
    if result.stderr:
        print("STDERR:", result.stderr[-3000:])
    assert result.returncode == 0, (
        f"multi-adapter step '{step_name}' failed (exit {result.returncode}).\n"
        f"STDOUT (last 4000):\n{result.stdout[-4000:]}\n"
        f"STDERR (last 2000):\n{result.stderr[-2000:]}"
    )


@pytest.fixture(
    scope="module", params=CASES, ids=lambda c: f"{c[0].split('/')[-1]}-{c[1]}"
)
def flavor_run(request):
    """Compose both checkpoints and score all three models once per case.

    Module-scoped: the composes and vLLM loads are the expensive part, and all
    four checks read the same captured distributions. Only one parametrization is
    live at a time, so at most two composed checkpoints sit on disk at once.
    """
    base, flavor = request.param
    label = f"{flavor}@{base}"
    with tempfile.TemporaryDirectory(prefix="multi_adapter_") as work_dir:
        _run_step(
            f"{label}: prepare adapters",
            "prepare",
            "--flavor",
            flavor,
            "--base",
            base,
            "--work-dir",
            work_dir,
            timeout=PREPARE_TIMEOUT,
        )
        for order in ("forward", "reversed"):
            _run_step(
                f"{label}: compose {order}",
                "build",
                "--work-dir",
                work_dir,
                "--order",
                order,
                timeout=BUILD_TIMEOUT,
            )
            _run_step(
                f"{label}: render cases {order}",
                "cases",
                "--work-dir",
                work_dir,
                "--order",
                order,
                timeout=CASES_TIMEOUT,
            )
        # Upstream first, on purpose: it generates the shared teacher-forcing
        # continuation, so the continuation comes from the base model and is
        # guaranteed to lie inside the base vocabulary that every run must score.
        _run_step(
            f"{label}: run upstream",
            "run",
            "--model",
            base,
            "--work-dir",
            work_dir,
            "--tag",
            "upstream",
            "--cases",
            os.path.join(work_dir, "cases_forward.json"),
            "--only",
            "base",
            timeout=RUN_TIMEOUT,
        )
        # Both slot orders, eager and then compiled. The compiled pass is the
        # one that goes through torch.compile and (for the greedy decode) CUDA
        # graphs, which is what a default-configured server runs.
        for eager in ("true", "false"):
            suffix = "" if eager == "true" else "_compiled"
            for order in ("forward", "reversed"):
                _run_step(
                    f"{label}: run {order}{suffix}",
                    "run",
                    "--model",
                    os.path.join(work_dir, f"ckpt_{order}"),
                    "--work-dir",
                    work_dir,
                    "--tag",
                    f"{order}{suffix}",
                    "--cases",
                    os.path.join(work_dir, f"cases_{order}.json"),
                    "--eager",
                    eager,
                    timeout=RUN_TIMEOUT,
                )
        yield label, work_dir


def _check(flavor_run, check):
    label, work_dir = flavor_run
    _run_step(
        f"{label}: compare {check}",
        "compare",
        "--work-dir",
        work_dir,
        "--check",
        check,
        "--label",
        label,
        timeout=COMPARE_TIMEOUT,
    )


@pytest.mark.requires_model
def test_no_control_token_equals_upstream(flavor_run):
    """Adapter id 0 is a true no-op even with several trained adapters embedded."""
    _check(flavor_run, "passthrough")


@pytest.mark.requires_model
def test_adapters_are_distinguishable(flavor_run):
    """Every adapter changes the distribution, and no two adapters agree.

    Ordered before the equality checks: if this fails, they are vacuous.
    """
    _check(flavor_run, "distinct")


@pytest.mark.requires_model
def test_output_is_independent_of_adapter_slot(flavor_run):
    """The same adapter at a different slot produces the same distribution."""
    _check(flavor_run, "slots")


@pytest.mark.requires_model
def test_output_is_independent_of_batch_composition(flavor_run):
    """A prompt is unaffected by other adapters sharing its batch.

    Checked in both execution modes, each against its own no-adapter control.
    """
    _check(flavor_run, "batch")


# A fifth check, ``decode``, used to live here: the same adapter had to generate a
# token-identical greedy continuation from a different slot, which was this suite's
# only coverage of the graph-captured decode path. It was removed rather than
# recalibrated. vLLM 0.20 does not reproduce a compiled run across processes -- one
# checkpoint loaded twice and compared against ITSELF diverges at the same token
# indices the slot comparison reported, with nothing permuted -- and exact token
# equality has no tolerance to widen. See the worker's module docstring.
