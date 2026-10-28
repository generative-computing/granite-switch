# SPDX-License-Identifier: Apache-2.0
"""The vLLM MultiSwitch refuses KV-cache dtypes that corrupt its routing.

``check_switch_kv_dtype`` (``granite_switch.vllm.switch._kv_dtype``) runs in
``MultiSwitch.__init__`` and allows only bfloat16 / float32 storage for the switch
heads. Both directions are asserted: the validated configurations must still pass,
and every float16 / fp8 configuration must fail with a message naming the flag.

The second class pins the arithmetic the guard rests on, in the style of
``test_counting_ceiling.py``: if a torch change made float16 hold -1e9 or fp8 e4m3
hold integers past 16, these fail and the guard should be revisited rather than
silently kept.

The module is loaded by file path so the vLLM package ``__init__`` is never
imported (same pattern as ``test_sr_doubled_q_tp_mapping.py``). CPU-only,
no model, milliseconds.
"""

import importlib.util
from pathlib import Path

import pytest
import torch

_KV_DTYPE = (
    Path(__file__).resolve().parents[2] / "src/granite_switch/vllm/switch/_kv_dtype.py"
)
assert _KV_DTYPE.is_file(), f"_kv_dtype not found at {_KV_DTYPE}"
_spec = importlib.util.spec_from_file_location("_kv_dtype", _KV_DTYPE)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
MASK_VALUE = _mod.MASK_VALUE
SUPPORTED_KV_DTYPES = _mod.SUPPORTED_KV_DTYPES
check_switch_kv_dtype = _mod.check_switch_kv_dtype


class TestGuard:
    @pytest.mark.parametrize(
        "model_dtype, cache_dtype",
        [
            (torch.bfloat16, "auto"),
            (torch.bfloat16, None),
            (torch.float32, "auto"),
            (torch.bfloat16, "bfloat16"),
            (torch.float32, "bfloat16"),
            (torch.bfloat16, "float32"),
        ],
    )
    def test_validated_configurations_pass(self, model_dtype, cache_dtype):
        check_switch_kv_dtype(model_dtype, cache_dtype)

    @pytest.mark.parametrize("cache_dtype", ["auto", "bfloat16", "fp8"])
    def test_float16_model_is_rejected_whatever_the_cache(self, cache_dtype):
        """The keys are built in the model dtype, so the cache cannot rescue fp16."""
        with pytest.raises(ValueError, match="--dtype float16"):
            check_switch_kv_dtype(torch.float16, cache_dtype)

    @pytest.mark.parametrize(
        "cache_dtype",
        ["float16", "half", "fp8", "fp8_e4m3", "fp8_e5m2", "fp8_inc", "not_a_dtype"],
    )
    def test_low_precision_cache_is_rejected(self, cache_dtype):
        with pytest.raises(ValueError, match=f"--kv-cache-dtype {cache_dtype}"):
            check_switch_kv_dtype(torch.bfloat16, cache_dtype)

    def test_message_names_the_fix(self):
        with pytest.raises(ValueError, match="Serve with --dtype bfloat16"):
            check_switch_kv_dtype(torch.bfloat16, "fp8")


class TestTheReasonsHold:
    @pytest.mark.parametrize("dtype", SUPPORTED_KV_DTYPES)
    def test_supported_dtypes_keep_the_mask_finite(self, dtype):
        """A finite mask is what keeps 0 * mask == 0 for the one-hot query."""
        mask = torch.tensor(MASK_VALUE).to(dtype)
        assert torch.isfinite(mask)
        assert (torch.zeros((), dtype=dtype) * mask).item() == 0.0

    @pytest.mark.parametrize("dtype", [torch.float16, torch.float8_e5m2])
    def test_rejected_caches_turn_the_mask_into_inf(self, dtype):
        """Stored as -inf, the mask gives 0 * -inf = NaN in the counting head."""
        mask = torch.tensor(MASK_VALUE, dtype=torch.bfloat16).to(dtype)
        assert torch.isinf(mask)
        assert torch.isnan(torch.zeros((), dtype=torch.float32) * mask.float())

    def test_fp8_e4m3_cannot_hold_adapter_ids_past_16(self):
        """The memory head stores adapter ids as values; e4m3 rounds 17+."""
        ids = torch.arange(1, 41, dtype=torch.float32)
        stored = ids.to(torch.float8_e4m3fn).float()
        assert torch.equal(stored[:16], ids[:16])
        assert stored[16].item() != 17.0
