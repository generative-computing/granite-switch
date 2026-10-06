# SPDX-License-Identifier: Apache-2.0
"""FA3 AOT-schedule correctness under FULL CUDA graphs + prefix caching (issue #139).

On vLLM 0.26 a Shadow-Residual MultiSwitch checkpoint served with
``cudagraph_mode=FULL`` AND prefix caching ON emits tokens outside the adapter's
allowed set and distributions that diverge from the reference — only on small,
prefix-cached decode steps. Turning off EITHER FULL or prefix caching matches the
reference exactly. Root cause and fix: ``docs/FA3_SCHEDULE_FULL_CUDAGRAPH_BUG.md``.

Two tests, both built on the SR MultiSwitch checkpoint composed by the worker
(``GraniteSwitchComposer``, CLAUDE.md gotcha #5):

- ``test_fa3_full_cudagraph_prefix_caching_is_correct`` — the bug-identifying
  test. It asserts the broken setting (FULL + prefix-caching on) matches the
  ``(eager, prefix-off)`` reference. **It is expected to FAIL on a tree without
  the fa3_schedule fix** (that is the point); it goes green once the fix lands.

- ``test_fa3_config_matrix_matches_reference`` — a regression suite comparing
  ``{FULL, FULL_AND_PIECEWISE, eager} x {prefix on/off}`` each against the same
  ``(eager, prefix-off)`` reference, so a future numerics/schedule regression in
  any of these settings is caught, not just the one that broke in #139.

Each worker step is a fresh subprocess so only one vLLM model is on GPU at a
time (CUDA context fully torn down between steps). The FULL runs use real CUDA
graphs (NOT enforce_eager), so assertions are on generated tokens and logprobs,
never on the eager-only MultiSwitch ``_debug_*`` attributes (CLAUDE.md gotcha #11).

Worker: _fa3_schedule_worker.py (build / run / compare phases).
Markers: requires_model (consistent with test_generation_equivalence.py).
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

WORKER = Path(__file__).parent / "_fa3_schedule_worker.py"
TIMEOUT = 1200  # 20 min: build + several vLLM loads + generate

# The reference setting known-correct across all environments (matches vLLM 0.19).
REF_CUDAGRAPH, REF_PREFIX = "eager", "off"
REF_TAG = "ref_eager_prefixoff"

# (cudagraph_mode, prefix_caching) cells for the regression matrix. The
# bug-identifying cell (FULL, on) is included here and also tested on its own.
MATRIX = [
    ("FULL", "on"),
    ("FULL", "off"),
    ("FULL_AND_PIECEWISE", "on"),
    ("FULL_AND_PIECEWISE", "off"),
    ("eager", "on"),
    # (eager, off) is the reference itself — compared trivially, so not re-listed.
]


def _tag(cudagraph, prefix):
    return f"run_{cudagraph.lower()}_prefix{prefix}"


def _run_step(step_name, *cmd_args, timeout):
    """Run one worker step as a subprocess and assert success (fresh CUDA ctx)."""
    cmd = [sys.executable, str(WORKER), *cmd_args]
    print(f"\n{'=' * 60}\n  Step: {step_name}\n  {' '.join(str(c) for c in cmd)}\n{'=' * 60}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.stdout:
        print(result.stdout[-4000:])
    if result.stderr:
        print("STDERR:", result.stderr[-2000:])
    assert result.returncode == 0, (
        f"FA3 schedule step '{step_name}' failed (exit {result.returncode}).\n"
        f"STDOUT (last 2000):\n{result.stdout[-2000:]}\n"
        f"STDERR (last 1000):\n{result.stderr[-1000:]}"
    )


def _build(work_dir):
    _run_step("build SR MultiSwitch", "build", "--work-dir", work_dir, timeout=TIMEOUT)


def _run(work_dir, cudagraph, prefix, tag):
    _run_step(
        f"run cudagraph={cudagraph} prefix={prefix}",
        "run",
        "--work-dir",
        work_dir,
        "--tag",
        tag,
        "--cudagraph",
        cudagraph,
        "--prefix",
        prefix,
        timeout=TIMEOUT,
    )


def _compare(work_dir, ref_tag, cand_tag, label):
    _run_step(
        f"compare {label}",
        "compare",
        "--work-dir",
        work_dir,
        "--ref",
        ref_tag,
        "--cand",
        cand_tag,
        "--label",
        label,
        timeout=120,
    )


@pytest.mark.requires_model
def test_fa3_full_cudagraph_prefix_caching_is_correct():
    """FULL + prefix-caching must match the (eager, prefix-off) reference.

    This is the issue #139 repro. It FAILS on a tree without the fa3_schedule
    fix (out-of-set tokens / diverged distributions) and PASSES once the fix is
    installed. Keeping it a hard assertion makes the broken state loud.
    """
    with tempfile.TemporaryDirectory(prefix="fa3_sched_") as work_dir:
        _build(work_dir)
        _run(work_dir, REF_CUDAGRAPH, REF_PREFIX, REF_TAG)
        bug_tag = _tag("FULL", "on")
        _run(work_dir, "FULL", "on", bug_tag)
        _compare(work_dir, REF_TAG, bug_tag, "FULL+prefix-caching vs eager+no-prefix")


@pytest.mark.requires_model
def test_fa3_config_matrix_matches_reference():
    """Every {cudagraph x prefix} setting must match the same reference.

    Guards against a schedule/numerics regression in any supported serving
    configuration, not only the one that broke in #139. Builds the model once,
    runs the reference, then each matrix cell, comparing all against the ref.
    """
    with tempfile.TemporaryDirectory(prefix="fa3_sched_matrix_") as work_dir:
        _build(work_dir)
        _run(work_dir, REF_CUDAGRAPH, REF_PREFIX, REF_TAG)

        failures = []
        for cudagraph, prefix in MATRIX:
            tag = _tag(cudagraph, prefix)
            label = f"{cudagraph}+prefix-{prefix}"
            _run(work_dir, cudagraph, prefix, tag)
            # Compare in a subprocess that returns nonzero on divergence; collect
            # all cell results so one run gets the full picture rather than
            # stopping at the first broken setting.
            cmd = [
                sys.executable,
                str(WORKER),
                "compare",
                "--work-dir",
                work_dir,
                "--ref",
                REF_TAG,
                "--cand",
                tag,
                "--label",
                label,
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if res.stdout:
                print(res.stdout[-3000:])
            if res.returncode != 0:
                failures.append(f"{label} (exit {res.returncode})")

        assert not failures, (
            "FA3 schedule regression — these settings diverged from the "
            f"(eager, prefix-off) reference:\n  " + "\n  ".join(failures)
        )
