# SPDX-License-Identifier: Apache-2.0
"""vLLM-suite collection policy: keep the parent pytest process CUDA-context-free.

Almost every GPU test in ``tests/vllm/`` does its CUDA work in a long-lived
subprocess worker (``_*_worker.py``) precisely so that **the parent pytest
process never creates a CUDA context** -- an invariant documented across the
suite (``test_model_forward.py``, ``test_conversation_routing.py``,
``test_tp_lora.py`` ...). It is load-bearing under **Exclusive_Process** GPU
compute mode: once the parent holds a context it owns the device, and every
subsequent worker subprocess that tries to acquire it dies with
``cudaErrorDevicesUnavailable`` ("CUDA-capable device(s) is/are busy or
unavailable"). The context cannot be handed back without the holding process
exiting -- ``torch.cuda.empty_cache()`` frees cached memory but not the primary
context -- so a teardown fixture cannot repair it.

A handful of kernel-equivalence files legitimately run Triton / vLLM CUDA
kernels *in-process* (there is nothing to delegate to a model worker): they
would create a context in the parent and poison every worker-backed test that
collection happens to order after them. ``ISOLATED_MODULES`` lists them; this
conftest runs each in its own ``pytest`` subprocess so the context lives and
dies entirely inside that child, and the parent stays clean regardless of
collection order.

To add a file here: it must create a CUDA context directly in the test process
(``.cuda()`` / ``device="cuda"`` in the test body or a module-level fixture),
rather than inside a ``_*_worker.py`` subprocess.
"""

import os
import subprocess
import sys

import pytest

# Test modules (basename) that create a CUDA context in-process and must
# therefore run in their own subprocess. Keep this list minimal -- prefer the
# subprocess-worker pattern for new GPU tests.
ISOLATED_MODULES = frozenset(
    {
        "test_bitmask_kernel_equivalence.py",
        "test_swiglu_activation_equivalence.py",
    }
)

# Set in the child subprocess we spawn, so the hook below is inert there and the
# real tests run normally (and own the child's CUDA context harmlessly).
_CHILD_ENV_FLAG = "GRANITE_SWITCH_ISOLATED_CUDA_CHILD"


def pytest_collection_modifyitems(config, items):
    """Replace each isolated module's items with a single subprocess runner.

    In the parent process we drop the in-module items (so the parent never
    imports/executes their CUDA bodies) and append one item per isolated module
    that shells out to ``pytest <module>`` in a fresh process. In the spawned
    child (flagged via the env var) we leave collection untouched.
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
    # Run the isolated subprocess modules FIRST, before any in-parent worker
    # subprocess starts. Under Exclusive_Process only one process can hold the
    # device: each isolated child must fully exit (releasing the GPU) before the
    # worker-backed tests acquire it, and vice versa. Fronting them guarantees
    # that ordering regardless of how the rest of the suite is collected.
    items[:] = runner_items + kept


@pytest.fixture(autouse=True, scope="module")
def _release_shared_gpu_workers():
    """Tear down the shared ``_multi_switch_worker`` after every module.

    ``test_multi_switch`` keeps a lazily-started, long-lived worker subprocess
    (``_workers[...]``) and owns a module-scoped teardown for it -- but **other**
    modules reuse that same worker by importing ``_send`` (e.g.
    ``test_conversation_routing``). When such a module starts the worker, the
    teardown defined *in test_multi_switch* does not fire at that module's end,
    so the worker leaks past the module boundary and keeps holding the GPU. Under
    Exclusive_Process the next GPU test's own worker then dies with
    ``cudaErrorDevicesUnavailable``.

    Shutting the shared workers down after each module makes the device free at
    every module boundary regardless of which module happened to start them.
    Idempotent: if nothing was started (or test_multi_switch already shut them
    down) this is a no-op.
    """
    yield
    try:
        from tests.vllm.test_multi_switch import _shutdown_all_workers
    except Exception:
        return  # module not importable in this environment -> nothing to release
    _shutdown_all_workers()


class _IsolatedModuleItem(pytest.Item):
    """A pytest item that runs one test module in its own pytest subprocess."""

    def __init__(self, *, name, parent, module_path):
        super().__init__(name, parent)
        self._module_path = module_path

    @classmethod
    def from_parent(cls, parent, *, name, module_path):
        return super().from_parent(parent, name=name, module_path=module_path)

    def runtest(self):
        env = dict(os.environ)
        env[_CHILD_ENV_FLAG] = "1"
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                self._module_path,
                "-v",
                "-s",
                "--tb=short",
                "-p",
                "no:cacheprovider",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        # pytest exit codes: 0 = all passed, 5 = no tests collected (e.g. the
        # whole module skipped because a backend is unavailable) -- both are a
        # non-failure here. Anything else (1 failures, 2 interrupted, 3 internal
        # error, ...) propagates as a failure of this item.
        if proc.returncode == 5:
            pytest.skip(f"{os.path.basename(self._module_path)}: no tests collected")
        if proc.returncode != 0:
            raise _IsolatedModuleFailed(
                self._module_path, proc.returncode, proc.stdout, proc.stderr
            )

    def repr_failure(self, excinfo):
        exc = excinfo.value
        if isinstance(exc, _IsolatedModuleFailed):
            return (
                f"Isolated module {exc.module_path} failed "
                f"(pytest subprocess exit {exc.returncode}).\n"
                f"--- subprocess stdout ---\n{exc.stdout}\n"
                f"--- subprocess stderr (tail) ---\n{exc.stderr[-4000:]}"
            )
        return super().repr_failure(excinfo)

    def reportinfo(self):
        return self._module_path, 0, f"isolated-subprocess: {self.name}"


class _IsolatedModuleFailed(Exception):
    def __init__(self, module_path, returncode, stdout, stderr):
        super().__init__(module_path)
        self.module_path = module_path
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
