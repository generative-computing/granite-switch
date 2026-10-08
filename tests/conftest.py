# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for granite_switch tests."""

import os

import pytest

from granite_switch.config import GraniteSwitchConfig

# ── Multi-GPU xdist worker pinning ────────────────────────────────
# When running with pytest-xdist (-n N), each worker pins to one GPU
# from the CUDA_VISIBLE_DEVICES list via round-robin.  With 1 GPU
# every worker gets GPU 0 (no-op).  Without xdist this is skipped.


def pytest_configure(config):
    # ── flashinfer sampler: off unless the runner asks for it ─────
    #
    # MEASURED, not precautionary. On an image shipping CUDA 12.4
    # (/usr/local/cuda-12.4, while the driver advertises 13.0 -- that is the driver
    # ceiling, not the toolkit) the vLLM 0.30 engine failed to start 171 times on
    #
    #     nvcc fatal : Unknown option '--compress-mode=size'
    #
    # flashinfer JIT-compiles its sampling kernels with `--compress-mode`, an nvcc
    # option added in CUDA 12.8. Every GPU suite went red downstream of that --
    # including on UPSTREAM's own GraniteMoeHybridForCausalLM, so it is not a
    # granite-switch defect, and it masks everything else because the engine never
    # reaches a test.
    #
    # It is version-dependent, which is why this lives here now: vLLM 0.26 is green
    # on that image WITHOUT this (test_logs/gputest-v26, 219 passed) and 0.30 is not
    # (test_logs/gputest-v30), so raising the CI ceiling to 0.30 is exactly what
    # exposes it. The variable is read and honoured on all of 0.26-0.30
    # (vllm/v1/sample/ops/topk_topp_sampler.py).
    #
    # SCOPED TO THE SAMPLER, and inert for what these tests assert: the attention
    # backend resolves to FlashAttention independently, and the equivalence suites
    # compare logprobs, which the model produces before any sampling op runs.
    #
    # setdefault, so an image with CUDA >= 12.8 can export
    # VLLM_USE_FLASHINFER_SAMPLER=1 and genuinely exercise that path. The real fix is
    # a newer toolkit in the image; this keeps the suites meaningful until then.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    if worker_id is None:
        return  # not running under xdist
    # "gw0" -> 0, "gw1" -> 1, ...
    worker_num = int(worker_id.lstrip("gw"))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        gpus = visible.split(",")
    else:
        # No restriction set — discover count via nvidia-smi to avoid
        # initializing a CUDA context in the parent process.
        import subprocess

        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                text=True,
                timeout=5,
            )
            gpus = [line.strip() for line in out.splitlines() if line.strip()]
        except Exception:
            gpus = ["0"]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus[worker_num % len(gpus)]


@pytest.fixture
def tiny_config():
    """Minimal GraniteSwitchConfig for fast CPU tests.

    2 decoder layers (+2 for the MultiSwitch cache slots = 4 total), 2 adapters,
    rank 4. hidden_size=64, 4 heads -> head_dim=16.
    """
    return GraniteSwitchConfig(
        vocab_size=300,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,  # 2 switch slots + 2 decoder
        num_attention_heads=4,
        num_key_value_heads=4,
        num_adapters=2,
        adapter_token_ids=[250, 251],
        adapter_substitute_token_ids=[1, 1],
        adapter_names=["adapter_a", "adapter_b"],
        max_lora_rank=4,
        adapter_ranks=[4, 4],
        switch_head_dim=16,
    )


@pytest.fixture
def tiny_config_no_adapters():
    """Minimal GraniteSwitchConfig with no adapters."""
    return GraniteSwitchConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_adapters=0,
    )
