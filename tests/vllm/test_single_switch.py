# SPDX-License-Identifier: Apache-2.0
"""vLLM SingleSwitch tests.

Test cases are defined once in tests/shared/single_switch_cases.py.
This file provides:
- CUDA/vLLM availability gating (lightweight, no CUDA context)
- A long-lived subprocess worker (_single_switch_worker.py) that owns the GPU
- _VLLMSingleSwitchBase with _run() that delegates to the worker via JSON-line

Requires CUDA GPU and vLLM installed. All tests are skipped otherwise.
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
    NUM_ADAPTERS,
    TEXT_TOKEN,
    ADAPTER_TOKEN_IDS_LIST,
    SingleSwitchTokenMatchingCases,
    SingleSwitchAdapterRetrievalCases,
    SingleSwitchEdgeCases,
    SingleSwitchShapeCorrectnessCases,
    SingleSwitchContextLengthSweepCases,
    SingleSwitchGainSensitivityCases,
)

# ── Worker management ─────────────────────────────────────────────

_WORKER_PATH = Path(__file__).parent / "_single_switch_worker.py"
_worker_proc = None
_worker_lock = threading.Lock()
_backend_name = None
# Sticky error: once the worker reports a fatal startup condition (e.g. a
# kernel-image mismatch in the auto-selected attention backend), we keep the
# message here and re-raise it on every subsequent test instead of re-spawning
# a worker that will hit the same crash. This converts what used to be ~hundreds
# of cascading BrokenPipeErrors into one clear, actionable failure per test.
_fatal_startup_error = None


def _ensure_worker():
    """Lazily start the long-lived worker subprocess."""
    global _worker_proc, _backend_name, _fatal_startup_error
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
    # Read the ready message
    ready_line = proc.stdout.readline()
    if not ready_line:
        stderr = proc.stderr.read()
        raise RuntimeError(f"Worker failed to start:\n{stderr}")
    ready = json.loads(ready_line)
    if "fatal" in ready:
        # Worker detected a non-recoverable environment problem (e.g. the
        # auto-selected attention backend has no kernel image for this GPU).
        # Drain stderr for context, mark this poison sticky, and fail.
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
    _backend_name = ready.get("backend_name", "unknown")
    _worker_proc = proc
    atexit.register(_shutdown_worker)


def _shutdown_worker():
    """Shut down the worker subprocess."""
    global _worker_proc
    if _worker_proc is not None and _worker_proc.poll() is None:
        _worker_proc.stdin.close()
        _worker_proc.wait(timeout=30)
    _worker_proc = None


def _send_request(seq, num_adapters=NUM_ADAPTERS, control_token_gain=15.0):
    """Send a request to the worker and return the result."""
    _ensure_worker()
    req = {
        "seq": seq,
        "num_adapters": num_adapters,
        "control_token_gain": control_token_gain,
    }
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


def _query_geometry():
    """Query the worker for switch geometry and cache attributes."""
    _ensure_worker()
    req = {"command": "query_geometry"}
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


# ── Module-scoped teardown ────────────────────────────────────────
# Release the GPU when this module's tests are done, so Pattern B
# subprocess tests in later files can claim it.

@pytest.fixture(autouse=True, scope="module")
def _worker_lifecycle():
    yield
    _shutdown_worker()


# ── vLLM _run adapter ───────────────────────────────────────────────

class _VLLMSingleSwitchBase:
    """Provides _run() for shared mixin tests via worker subprocess."""

    def _run(self, seq, num_adapters=NUM_ADAPTERS, control_token_gain=15.0):
        """Shared mixin contract: seq in, flat list of ints out."""
        return _send_request(seq, num_adapters, control_token_gain)


# ── Shared test classes (from mixin) ────────────────────────────────

class TestTokenMatching(_VLLMSingleSwitchBase, SingleSwitchTokenMatchingCases):
    pass


class TestAdapterRetrieval(_VLLMSingleSwitchBase, SingleSwitchAdapterRetrievalCases):
    pass


class TestEdgeCases(_VLLMSingleSwitchBase, SingleSwitchEdgeCases):
    pass


class TestShapeCorrectness(_VLLMSingleSwitchBase, SingleSwitchShapeCorrectnessCases):
    pass


class TestContextLengthSweep(_VLLMSingleSwitchBase, SingleSwitchContextLengthSweepCases):
    pass


class TestGainSensitivity(_VLLMSingleSwitchBase, SingleSwitchGainSensitivityCases):
    pass


# ── vLLM-specific tests (geometry, gain, FA3 boundary) ────────────


class TestGeometry:
    """Verify SingleSwitch geometry matches the mock backbone config."""

    def test_geometry_matches_config(self):
        info = _query_geometry()
        assert info["num_heads"] == 4
        assert info["num_kv_heads"] == 2
        assert info["head_dim"] == 64
        assert info["scaling"] == 0.125
        assert info["effective_gain"] == 15.0 / 0.125  # 120.0
        assert info["control_token_gain"] == 15.0
        assert info["num_adapters"] == 32


class TestGainRoundTrip:
    """Verify gain compensation round-trips through bf16 exactly."""

    @pytest.mark.parametrize("attention_multiplier", [0.0078125, 0.015625, 0.0625, 0.125, 1.0])
    def test_gain_roundtrip_bf16(self, attention_multiplier):
        import torch
        gain = torch.tensor(15.0, dtype=torch.bfloat16)
        multiplier = torch.tensor(attention_multiplier, dtype=torch.bfloat16)
        effective = gain / multiplier
        recovered = effective * multiplier
        assert recovered.item() == 15.0


class TestFA3Boundary(_VLLMSingleSwitchBase):
    """Verify adapter routing near FA3's 512-token CUDA graph boundary."""

    @pytest.mark.parametrize("seq_len", [511, 512, 513, 768, 1024])
    def test_fa3_boundary_lengths(self, seq_len):
        seq = [TEXT_TOKEN] * seq_len
        seq[1] = ADAPTER_TOKEN_IDS_LIST[0]
        result = self._run(seq, num_adapters=4)
        assert len(result) == seq_len
        assert result[0] == 0
        assert all(v == 1 for v in result[1:])


class TestKVCacheShape:
    """Verify KV cache dimensions match the multi-head geometry."""

    def test_kv_cache_shape(self):
        info = _query_geometry()
        shape = info["kv_cache_shape"]
        assert shape[3] == info["num_kv_heads"], (
            f"Expected num_kv_heads={info['num_kv_heads']} at dim 3, got {shape[3]}"
        )
        assert shape[4] == info["head_dim"], (
            f"Expected head_dim={info['head_dim']} at dim 4, got {shape[4]}"
        )


class TestFallbackGeometry:
    """Verify config geometry is active (not fallback 1-head/32d/scale=1.0)."""

    def test_config_overrides_fallback(self):
        info = _query_geometry()
        assert info["num_heads"] == 4, "Expected config geometry, got fallback"
        assert info["head_dim"] == 64, "Expected config geometry, got fallback"
        assert info["scaling"] == 0.125, "Expected config geometry, got fallback"
