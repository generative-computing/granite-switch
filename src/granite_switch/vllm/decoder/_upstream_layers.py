# SPDX-License-Identifier: Apache-2.0
"""Version-tolerant access to the two upstream vLLM Granite building blocks the
switch decoders borrow: the sparse expert bank (``GraniteMoeMoE``) and the dense
shared MLP (``GraniteMoeSharedMLP``).

These are pure attention/MLP modules — nothing hybrid about them. vLLM has
historically kept them in the consolidated ``granitemoehybrid`` model file and
also exposes them from the non-hybrid ``granitemoe`` / ``granitemoeshared``
files, but which module is the canonical home varies across vLLM releases
(0.19.x vs 0.20.x). Prefer the non-hybrid module and fall back to the
consolidated one so a single codebase works on every pinned vLLM without a
hard dependency on the hybrid module name.
"""


def get_granite_moe_moe():
    """Return the upstream vLLM ``GraniteMoeMoE`` (frozen sparse expert bank)."""
    try:
        from vllm.model_executor.models.granitemoe import GraniteMoeMoE
    except ImportError:
        from vllm.model_executor.models.granitemoehybrid import GraniteMoeMoE
    return GraniteMoeMoE


def get_granite_moe_shared_mlp():
    """Return the upstream vLLM ``GraniteMoeSharedMLP`` (dense shared MLP)."""
    try:
        from vllm.model_executor.models.granitemoeshared import GraniteMoeSharedMLP
    except ImportError:
        from vllm.model_executor.models.granitemoehybrid import GraniteMoeSharedMLP
    return GraniteMoeSharedMLP
