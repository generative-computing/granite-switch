# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for granite_switch tests."""

import os

import pytest
import torch

from granite_switch.config import GraniteSwitchConfig


# ── Multi-GPU xdist worker pinning ────────────────────────────────
# When running with pytest-xdist (-n N), each worker pins to one GPU
# from the CUDA_VISIBLE_DEVICES list via round-robin.  With 1 GPU
# every worker gets GPU 0 (no-op).  Without xdist this is skipped.

def pytest_configure(config):
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
                text=True, timeout=5,
            )
            gpus = [line.strip() for line in out.splitlines() if line.strip()]
        except Exception:
            gpus = ["0"]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus[worker_num % len(gpus)]


@pytest.fixture
def tiny_config():
    """Minimal GraniteSwitchConfig for fast CPU tests.

    2 layers (+1 for switch = 3 total), 2 adapters, rank 4.
    hidden_size=64, 4 heads -> head_dim=16.
    """
    return GraniteSwitchConfig(
        vocab_size=300,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,  # 1 switch + 2 decoder
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
