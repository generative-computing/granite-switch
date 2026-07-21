# SPDX-License-Identifier: Apache-2.0
"""Shared single-GPU distributed init for all vLLM tests.

All vLLM test files should call ``ensure_distributed()`` from here instead
of maintaining their own per-module ``_DISTRIBUTED_INITIALIZED`` flag.
A per-module flag breaks when pytest runs multiple vLLM test files in the
same process — the second file's flag is False but vLLM's global parallel
state is already initialized, causing an AssertionError.

This module uses a single process-wide flag + a guard on
``torch.distributed.is_initialized()`` to be safe regardless of import order.
"""

import socket

_INITIALIZED = False


def _free_port():
    """Return an OS-assigned free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def ensure_distributed(vllm_config=None):
    """Initialize distributed + vLLM parallel context for single-GPU testing.

    Safe to call multiple times from any test module — only the first call
    actually initializes.

    Args:
        vllm_config: VllmConfig instance. Required by vLLM 0.19.1+ where
            initialize_model_parallel() reads the current vllm config.
    """
    global _INITIALIZED
    if _INITIALIZED:
        return

    import torch

    if torch.distributed.is_initialized():
        # Another module already initialized in this process
        _INITIALIZED = True
        return

    # Use a dynamically allocated free port so that multiple processes
    # (e.g. pytest parent and a long-lived worker subprocess) can each
    # initialize their own world_size=1 group without conflicting.
    port = _free_port()

    from vllm.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=f"tcp://localhost:{port}",
    )

    from vllm.config import VllmConfig, set_current_vllm_config

    if vllm_config is None:
        vllm_config = VllmConfig()
    with set_current_vllm_config(vllm_config):
        initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )

    _INITIALIZED = True
