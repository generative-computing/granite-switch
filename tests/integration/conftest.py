# SPDX-License-Identifier: Apache-2.0
"""Integration-suite collection policy: keep the parent pytest process CUDA-free.

Same invariant, same hazard, same remedy as ``tests/vllm/conftest.py`` -- see
its module docstring for the full story. Under **Exclusive_Process** GPU compute
mode only one process may hold the device's primary context at a time. The
vLLM-serving integration tests (``test_audio_serving_smoke``,
``test_audio_uncached_processor``, ``test_multi_switch_vllm_generate`` ...) do
their CUDA work in a vLLM ``EngineCore`` **subprocess**; that subprocess dies
with ``cudaErrorDevicesUnavailable`` ("CUDA-capable device(s) is/are busy or
unavailable") the instant the parent pytest process is already holding a
context. ``torch.cuda.empty_cache()`` frees cached memory but not the primary
context, so a teardown fixture cannot hand it back -- only the holding process
exiting releases it.

Two integration modules create a CUDA context *directly in the test process*
(``.cuda()`` / ``device="cuda"`` in a module-scoped fixture or test body),
because they drive a HuggingFace / transformers model in-process with nothing to
delegate to a worker:

* ``test_adapter_routing_audio_enabled.py`` -- ``.cuda()`` on the HF model
* ``test_asr_ctc_default_gpu.py``           -- ``ASRTranscriber(device="cuda")``

Collected in-parent, either one poisons every vLLM-subprocess test ordered after
it (alphabetical collection puts both ahead of the serving tests, which is
exactly how the suite went red). ``ISOLATED_MODULES`` lists them; each runs in
its own fronted ``pytest`` subprocess so its context lives and dies inside that
child and the parent stays clean regardless of collection order.

To add a file here: it must create a CUDA context directly in the test process,
rather than inside a subprocess worker.
"""

import os

# Reuse the proven isolation machinery from the vLLM suite rather than duplicate
# it. ``_IsolatedModuleItem`` shells out to ``pytest <module>`` in a fresh
# process; ``_CHILD_ENV_FLAG`` makes the hook inert inside that child so the real
# tests run (and own the child's context harmlessly).
from tests.vllm.conftest import _CHILD_ENV_FLAG, _IsolatedModuleItem

# Integration modules that create a CUDA context in-process and must therefore
# run in their own subprocess. Keep this minimal -- prefer the subprocess-worker
# pattern for new GPU tests.
ISOLATED_MODULES = frozenset(
    {
        "test_adapter_routing_audio_enabled.py",
        "test_asr_ctc_default_gpu.py",
    }
)


def pytest_collection_modifyitems(config, items):
    """Front each isolated module as a single subprocess runner.

    In the parent we drop the in-module items (so the parent never imports or
    executes their CUDA bodies) and prepend one item per isolated module that
    shells out to ``pytest <module>`` in a fresh process. In the spawned child
    (flagged via the env var) collection is left untouched.
    """
    if os.environ.get(_CHILD_ENV_FLAG):
        return  # inside the isolated child: run the real tests as collected

    if not items:
        return

    kept = []
    isolated_paths = {}  # module fspath -> basename, first-seen order preserved
    for item in items:
        basename = os.path.basename(str(item.path))
        if basename in ISOLATED_MODULES:
            isolated_paths.setdefault(str(item.path), basename)
        else:
            kept.append(item)

    if not isolated_paths:
        return

    session = items[0].session
    runner_items = [
        _IsolatedModuleItem.from_parent(
            session,
            name=f"{basename}::<isolated-subprocess>",
            module_path=path,
        )
        for path, basename in isolated_paths.items()
    ]
    # Run the isolated subprocess modules FIRST, before any in-parent vLLM
    # EngineCore subprocess starts. Under Exclusive_Process only one process can
    # hold the device: each isolated child must fully exit (releasing the GPU)
    # before the serving tests acquire it, and vice versa. Fronting them
    # guarantees that ordering regardless of how the rest of the suite is
    # collected.
    items[:] = runner_items + kept


def shutdown_llm(llm):
    """Deterministically tear down a vLLM ``LLM`` and release its GPU.

    ``del llm; gc.collect(); torch.cuda.empty_cache()`` is **not** enough. vLLM
    runs its ``EngineCore`` in a child process (``VLLM_WORKER_MULTIPROC_METHOD=
    spawn``); ``empty_cache()`` frees cached memory but neither tears that
    subprocess down nor releases the device's primary CUDA context -- the engine
    only exits via its own ``atexit`` hook, i.e. at *pytest* exit, not at a
    fixture boundary.

    That gap is fatal for a ``scope="module"`` fixture parametrized over several
    checkpoints: when the fixture is rebuilt for the next checkpoint, the
    previous engine is still holding the GPU, and under **Exclusive_Process** the
    new ``EngineCore`` dies with ``cudaErrorDevicesUnavailable``. Calling the
    engine-core shutdown explicitly makes the subprocess exit and hand the device
    back before the next engine boots.

    Best-effort and version-tolerant: the private path has moved across vLLM
    releases, so every step is guarded and a missing attribute is a no-op (the
    ``atexit`` hook remains the backstop).
    """
    import gc

    try:
        engine = getattr(llm, "llm_engine", None)
        engine_core = getattr(engine, "engine_core", None)
        shutdown = getattr(engine_core, "shutdown", None)
        if callable(shutdown):
            shutdown()
    except Exception:
        pass  # best-effort: fall back to del + atexit below
    finally:
        del llm
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
