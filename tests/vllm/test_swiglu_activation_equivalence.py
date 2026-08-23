# SPDX-License-Identifier: Apache-2.0
"""Isolated equivalence: our Triton SwiGLU activation vs vLLM's SiluAndMul.

The fused gate/up kernel computes the activation as ``(gate * tl.sigmoid(gate))
* up`` (fp32 compute, output dtype store). vLLM's SiluAndMul computes
``silu(x)*y`` with a hand-written CUDA kernel (``x / (1 + expf(-x))``). silu
involves exp + division, so the two intrinsics could disagree beyond rounding —
and since this runs on EVERY token (base path included, no adapter required),
any divergence would drift the whole model, not just adapted tokens.

This test isolates JUST the activation math: it feeds identical contiguous
gate|up tensors to (a) vLLM's SiluAndMul CUDA kernel and (b) a standalone Triton
kernel using the IDENTICAL expression from our fused kernel, and compares —
across dtypes and value magnitudes (incl. large |x| that saturates exp).
"""

import pytest
import torch
import triton
import triton.language as tl

_CUDA = torch.cuda.is_available()


def _vllm_ok():
    try:
        from vllm.model_executor.layers.activation import SiluAndMul  # noqa: F401

        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _CUDA or not _vllm_ok(), reason="requires CUDA GPU and vLLM"
)

if _CUDA and _vllm_ok():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.activation import SiluAndMul


@triton.jit
def _silu_mul_kernel(
    GateUp,
    stride_gm,
    stride_gn,
    Out,
    stride_om,
    stride_on,
    M,
    H,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # EXACT activation expression used by _switch_lora_expand_swiglu_kernel:
    # fp32 load, (gate * tl.sigmoid(gate)) * up, store in output dtype.
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    h = pid_h * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (m[:, None] < M) & (h[None, :] < H)
    gate = tl.load(
        GateUp + m[:, None] * stride_gm + h[None, :] * stride_gn, mask=mask, other=0.0
    ).to(tl.float32)
    up = tl.load(
        GateUp + m[:, None] * stride_gm + (H + h[None, :]) * stride_gn,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    out = (gate * tl.sigmoid(gate)) * up
    tl.store(
        Out + m[:, None] * stride_om + h[None, :] * stride_on,
        out.to(Out.dtype.element_ty),
        mask=mask,
    )


def _triton_silu_mul(gateup):
    M, N = gateup.shape
    H = N // 2
    out = torch.empty(M, H, device=gateup.device, dtype=gateup.dtype)
    grid = (triton.cdiv(M, 32), triton.cdiv(H, 32))
    _silu_mul_kernel[grid](
        gateup,
        gateup.stride(0),
        gateup.stride(1),
        out,
        out.stride(0),
        out.stride(1),
        M,
        H,
        BLOCK_M=32,
        BLOCK_N=32,
    )
    return out


# Per-dtype tolerance: comparison is done in fp32 against vLLM's fp32-internal
# CUDA silu. fp32 isolates the pure intrinsic (exp/division) agreement; bf16/fp16
# additionally round the stored output, so allow ~1 ULP of the output dtype.
_TOL = {
    torch.float32: (1e-5, 1e-5),
    torch.float16: (2e-3, 2e-3),
    torch.bfloat16: (1e-2, 1e-2),
}


class TestSiluMulEquivalence:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("scale", [1.0, 5.0, 20.0, 50.0])
    def test_triton_silu_mul_matches_vllm(self, dtype, scale):
        torch.manual_seed(0)
        M, H = 256, 4096
        gateup = (torch.randn(M, 2 * H, device="cuda", dtype=torch.float32) * scale).to(
            dtype
        )
        gateup = gateup.contiguous()

        with set_current_vllm_config(VllmConfig()):
            act = SiluAndMul()
        ref = act.forward_cuda(gateup)  # vLLM C++ silu_and_mul
        ours = _triton_silu_mul(gateup)  # our Triton (gate*sigmoid(gate))*up

        rf, of = ref.float(), ours.float()
        max_abs = (rf - of).abs().max().item()
        denom = rf.abs().clamp_min(1e-6)
        max_rel = ((rf - of).abs() / denom).max().item()
        ndiff = (rf != of).sum().item()
        print(
            f"\n[dtype={dtype} scale={scale}] max_abs={max_abs:.3e} "
            f"max_rel={max_rel:.3e} elems_differ={ndiff}/{rf.numel()}"
        )

        atol, rtol = _TOL[dtype]
        torch.testing.assert_close(
            of,
            rf,
            atol=atol,
            rtol=rtol,
            msg=f"Triton silu*mul diverges from vLLM SiluAndMul "
            f"(dtype={dtype}, scale={scale})",
        )

    def test_fp32_intrinsic_agreement(self):
        """Tightest check: in fp32 (no output rounding) the two silu intrinsics
        must agree to near machine precision across a wide value sweep."""
        # gate|up layout [1, 2N]: gate = x sweeping [-60, 60], up = 1.
        x = torch.linspace(-60, 60, 4096, device="cuda", dtype=torch.float32)
        N = x.numel()
        gu = torch.empty(1, 2 * N, device="cuda", dtype=torch.float32)
        gu[0, :N] = x
        gu[0, N:] = 1.0

        with set_current_vllm_config(VllmConfig()):
            act = SiluAndMul()
        ref = act.forward_cuda(gu)
        ours = _triton_silu_mul(gu)
        max_abs = (ref - ours).abs().max().item()
        print(f"\n[fp32 silu sweep x in [-60,60]] max_abs_diff={max_abs:.3e}")
        torch.testing.assert_close(ours, ref, atol=1e-5, rtol=1e-5)
