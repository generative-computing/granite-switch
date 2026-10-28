# SPDX-License-Identifier: Apache-2.0
"""Which KV-cache storage dtypes the vLLM MultiSwitch heads can run on.

Kept free of any vLLM import so it can be unit-tested on CPU (the same pattern
as ``decoder/shadow_residual/_sr_ops.py``).

The switch's counting and memory heads are ordinary paged-KV ``vllm.Attention``
modules: their keys and values are built in the model dtype and stored in the
KV-cache dtype. Two of those values survive only in bfloat16 or float32:

* **The counting head's mask.** Non-participating keys hold the finite
  ``MASK_VALUE`` (``-1e9``), never ``-inf``, because the one-hot query has zeros
  where the mask sits and ``0 * -inf = NaN``. float16 and fp8 e5m2 cannot hold
  ``-1e9`` and store ``-inf`` -- the exact NaN the finite mask exists to avoid --
  so every write address comes back wrong. Built directly in float16
  (``--dtype float16``) the mask fares no better: depending on the code path
  ``torch.full`` either raises an overflow error or quietly fills ``-inf``.
* **The memory head's values.** Each control token stores its adapter id as a
  value. fp8 e4m3 represents integers exactly only up to 16, so above 16
  adapters an id rounds to a neighbour and its tokens route to that adapter.

Apart from that overflow, which names neither cause nor fix, nothing here fails
loudly: routing is just wrong, and the base model keeps generating. So the switch
checks once at construction, and a misconfigured server fails at startup with a
message naming the flag to change.
"""

import torch

# Masking key for the counting head. Finite on purpose -- see the module
# docstring -- and the reason the KV dtype needs float32's exponent range.
MASK_VALUE = -1e9

# Storage dtypes validated for the switch heads (both hold MASK_VALUE finite).
SUPPORTED_KV_DTYPES = (torch.bfloat16, torch.float32)

# vLLM ``--kv-cache-dtype`` names that are plain float dtypes. Every other
# non-"auto" name (fp8, fp8_e4m3, fp8_e5m2, ...) is a quantized cache.
_FLOAT_CACHE_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float": torch.float32,
    "float16": torch.float16,
    "half": torch.float16,
}


def check_switch_kv_dtype(
    model_dtype: torch.dtype, cache_dtype: str | None = "auto"
) -> None:
    """Raise ``ValueError`` unless the switch heads' KV storage is bf16 or fp32.

    Args:
        model_dtype: The model dtype (vLLM ``--dtype``). The switch builds its
            keys and values in it, and an ``"auto"`` cache stores them in it.
        cache_dtype: vLLM's ``cache_config.cache_dtype`` (``--kv-cache-dtype``).
            ``None`` is treated as ``"auto"``.
    """
    if model_dtype not in SUPPORTED_KV_DTYPES:
        # The keys are built in the model dtype, so no cache dtype can help.
        source = f"--dtype {str(model_dtype).removeprefix('torch.')}"
    else:
        name = "auto" if cache_dtype is None else str(cache_dtype)
        storage = model_dtype if name == "auto" else _FLOAT_CACHE_DTYPES.get(name)
        if storage in SUPPORTED_KV_DTYPES:
            return
        source = f"--kv-cache-dtype {name}"

    raise ValueError(
        f"Granite Switch's MultiSwitch needs its KV cache stored in bfloat16 or "
        f"float32, but this server is configured with {source}. Its adapter "
        f"routing stores a -1e9 attention mask and integer adapter ids in that "
        f"cache; float16 and fp8 cannot hold them exactly, so tokens would "
        f"silently route to the wrong adapter. Serve with --dtype bfloat16 and "
        f"the default --kv-cache-dtype auto. See section 8 of "
        f"docs/MULTISWITCH_EXPLAINED.md."
    )
