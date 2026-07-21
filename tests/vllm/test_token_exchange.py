# SPDX-License-Identifier: Apache-2.0
"""vLLM backend tests for token-exchange mode.

Mirrors tests/hf/test_token_exchange.py — verifies that on the vLLM
SingleSwitch path:

1. The control_to_substitute_lut tensor maps each adapter control token id
   to its configured substitute id, and leaves all other ids at -1.
2. Non-control positions in modified_input_ids are unchanged from the
   original input_ids tensor.
3. Control positions in modified_input_ids are rewritten to the
   substitute id from the LUT.

Tests #2 and #3 require a forward pass, so they go through the long-lived
SingleSwitch worker subprocess (the same one used by test_single_switch.py)
via two new commands: 'query_lut' and 'forward_with_modified'. The worker's
mock config now populates adapter_token_ids + adapter_substitute_token_ids
so the LUT path is exercised — see _single_switch_worker.py:_setup.

Requires CUDA GPU and vLLM installed. All tests skipped otherwise.
All GPU work happens in the subprocess worker — the parent pytest process
never creates a CUDA context (required for Exclusive_Process GPU mode).
"""

import atexit
import importlib.util
import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

_VLLM_AVAILABLE = importlib.util.find_spec("vllm") is not None

pytestmark = pytest.mark.skipif(
    not _VLLM_AVAILABLE,
    reason="requires vLLM installed (GPU checked by worker)",
)

from tests.shared.single_switch_cases import (
    ADAPTER_TOKEN_IDS_LIST,
    NUM_ADAPTERS,
    TEXT_TOKEN,
)

# Worker's deterministic substitute mapping: control_id (1000+i) → sub_id (i+1).
# Matches ADAPTER_SUBSTITUTE_TOKEN_IDS_LIST in _single_switch_worker.py:_setup.
ADAPTER_SUBSTITUTE_TOKEN_IDS_LIST = [i + 1 for i in range(NUM_ADAPTERS)]


# ── Worker management ─────────────────────────────────────────────
# Same pattern as test_single_switch.py — own module-private worker so
# pytest can run the two files independently or together.

_WORKER_PATH = Path(__file__).parent / "_single_switch_worker.py"
_worker_proc = None
_worker_lock = threading.Lock()
_fatal_startup_error = None


def _ensure_worker():
    global _worker_proc, _fatal_startup_error
    if _fatal_startup_error is not None:
        pytest.fail(_fatal_startup_error, pytrace=False)
    if _worker_proc is not None and _worker_proc.poll() is None:
        return
    proc = subprocess.Popen(
        [sys.executable, str(_WORKER_PATH)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    ready_line = proc.stdout.readline()
    if not ready_line:
        stderr = proc.stderr.read()
        raise RuntimeError(f"Worker failed to start:\n{stderr}")
    ready = json.loads(ready_line)
    if "fatal" in ready:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        stderr_tail = (proc.stderr.read() or "")[-2000:]
        backend = ready.get("backend_name", "unknown")
        _fatal_startup_error = (
            f"vLLM worker cannot start: {ready['fatal']}\n"
            f"Backend: {backend}\n"
            f"Hint: {ready.get('hint', '')}\n"
            f"--- worker stderr (tail) ---\n{stderr_tail}"
        )
        pytest.fail(_fatal_startup_error, pytrace=False)
    assert ready.get("ready"), f"Unexpected ready message: {ready}"
    _worker_proc = proc
    atexit.register(_shutdown_worker)


def _shutdown_worker():
    global _worker_proc
    if _worker_proc is not None and _worker_proc.poll() is None:
        _worker_proc.stdin.close()
        _worker_proc.wait(timeout=30)
    _worker_proc = None


def _send_command(req):
    """Send a JSON request to the worker and return its 'result' field."""
    _ensure_worker()
    with _worker_lock:
        _worker_proc.stdin.write(json.dumps(req) + "\n")
        _worker_proc.stdin.flush()
        resp_line = _worker_proc.stdout.readline()
    if not resp_line:
        stderr = _worker_proc.stderr.read()
        raise RuntimeError(f"Worker died unexpectedly:\n{stderr}")
    resp = json.loads(resp_line)
    if "error" in resp:
        raise RuntimeError(f"Worker error:\n{resp['error']}")
    return resp["result"]


@pytest.fixture(autouse=True, scope="module")
def _worker_lifecycle():
    yield
    _shutdown_worker()


# ── Tests ─────────────────────────────────────────────────────────


class TestLUTMapping:
    """control_to_substitute_lut is the canonical control→substitute table.

    It is built once at SingleSwitch construction from
    config.adapter_token_ids + config.adapter_substitute_token_ids; tested
    here against the worker's mock config (control 1000+i → substitute i+1).
    """

    def test_lut_maps_control_to_substitute(self):
        lut = _send_command({"command": "query_lut"})
        assert lut is not None, (
            "control_to_substitute_lut was None — adapter_substitute_token_ids "
            "missing from worker mock config?"
        )
        for ctrl_id, sub_id in zip(
            ADAPTER_TOKEN_IDS_LIST, ADAPTER_SUBSTITUTE_TOKEN_IDS_LIST
        ):
            assert lut[ctrl_id] == sub_id, (
                f"lut[{ctrl_id}]={lut[ctrl_id]}, expected substitute {sub_id}"
            )

    def test_lut_marks_non_control_with_sentinel(self):
        lut = _send_command({"command": "query_lut"})
        assert lut is not None
        # TEXT_TOKEN (50) and a few arbitrary non-control ids should be -1.
        for non_control in [TEXT_TOKEN, 0, 51, 52, 999]:
            assert lut[non_control] == -1, (
                f"lut[{non_control}]={lut[non_control]}, expected -1 sentinel"
            )


class TestInputRewrite:
    """SingleSwitch.forward returns (adapter_indices, modified_input_ids).

    modified_input_ids must equal input_ids at non-control positions and
    equal lut[ctrl_id] (the substitute) at control positions. The decoder
    embeds modified_input_ids; the switch itself reads the original
    input_ids so adapter detection is unaffected.
    """

    def test_non_control_positions_unchanged(self):
        # Mix of non-control tokens with one control token in the middle.
        ctrl_id = ADAPTER_TOKEN_IDS_LIST[0]
        seq = [TEXT_TOKEN, 51, ctrl_id, 53, 54]
        result = _send_command(
            {
                "command": "forward_with_modified",
                "seq": seq,
                "num_adapters": 4,
                "control_token_gain": 15.0,
            }
        )
        modified = result["modified_input_ids"]
        # Positions 0, 1, 3, 4 are non-control — must be unchanged.
        assert modified[0] == seq[0]
        assert modified[1] == seq[1]
        assert modified[3] == seq[3]
        assert modified[4] == seq[4]

    def test_control_positions_rewritten_to_substitute(self):
        # Control token at position 2 — must be rewritten to its substitute.
        ctrl_id = ADAPTER_TOKEN_IDS_LIST[0]
        expected_sub = ADAPTER_SUBSTITUTE_TOKEN_IDS_LIST[0]
        seq = [TEXT_TOKEN, 51, ctrl_id, 53, 54]
        result = _send_command(
            {
                "command": "forward_with_modified",
                "seq": seq,
                "num_adapters": 4,
                "control_token_gain": 15.0,
            }
        )
        modified = result["modified_input_ids"]
        assert modified[2] == expected_sub, (
            f"control position rewrite failed: got {modified[2]}, "
            f"expected substitute {expected_sub}"
        )

    def test_multiple_control_tokens_each_rewritten(self):
        # Two distinct control tokens; each must map to its own substitute.
        ctrl0 = ADAPTER_TOKEN_IDS_LIST[0]
        ctrl1 = ADAPTER_TOKEN_IDS_LIST[1]
        sub0 = ADAPTER_SUBSTITUTE_TOKEN_IDS_LIST[0]
        sub1 = ADAPTER_SUBSTITUTE_TOKEN_IDS_LIST[1]
        seq = [TEXT_TOKEN, ctrl0, TEXT_TOKEN, ctrl1, TEXT_TOKEN]
        result = _send_command(
            {
                "command": "forward_with_modified",
                "seq": seq,
                "num_adapters": 4,
                "control_token_gain": 15.0,
            }
        )
        modified = result["modified_input_ids"]
        assert modified[0] == TEXT_TOKEN
        assert modified[1] == sub0
        assert modified[2] == TEXT_TOKEN
        assert modified[3] == sub1
        assert modified[4] == TEXT_TOKEN

    def test_switch_still_detects_adapter_after_rewrite(self):
        # The rewrite must NOT confuse adapter detection — the switch reads
        # the original input_ids before the rewrite happens.
        ctrl_id = ADAPTER_TOKEN_IDS_LIST[2]
        seq = [TEXT_TOKEN, ctrl_id, TEXT_TOKEN, TEXT_TOKEN]
        result = _send_command(
            {
                "command": "forward_with_modified",
                "seq": seq,
                "num_adapters": 4,
                "control_token_gain": 15.0,
            }
        )
        adapter_indices = result["adapter_indices"]
        # Position 0 fires before any control: adapter 0 (base).
        # Position 1 is the control for adapter index 3 (1-indexed: ctrl_idx 2 → adapter 3).
        # SingleSwitch persists adapter id once fired → positions 1+ all 3.
        assert adapter_indices[0] == 0
        assert adapter_indices[1] == 3
        assert adapter_indices[2] == 3
        assert adapter_indices[3] == 3
