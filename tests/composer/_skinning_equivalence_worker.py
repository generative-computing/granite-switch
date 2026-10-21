# SPDX-License-Identifier: Apache-2.0
"""Subprocess worker: verify a skinned GraniteSwitch model matches the original's logits.

Invoked as::

    python tests/composer/_skinning_equivalence_worker.py <model_name_or_path>

Loads the original model in its **native dtype** (the precision it was published
in), computes reference logits, frees it, then builds a GraniteSwitch skin
(num_adapters=0) via the builder's weight-transfer pipeline, and asserts the two
agree.

Native dtype is critical: skinned models must be equivalent in the precision
users will actually deploy.  Float32 may mask real differences caused by
attention scaling order (pre-multiply Q vs post-divide scores).

AGREEMENT IS CHECKED WITHIN A TOLERANCE, NOT BIT-FOR-BIT.  This worker asserted
``max_diff == 0.0`` until it was found to be asking for something unreachable:
the skin uses fused QKV / gate-up projections while upstream HuggingFace uses
separate ones, and fusing changes the order of a floating-point sum.  ``a+b+c``
and ``(a+b)+c`` are not bit-identical, so a few ULP of disagreement is arithmetic,
not a defect, and the test could only ever fail.  It sat behind an unconditional
``@pytest.mark.skip`` for that reason, which cost the composer its only
weight-transfer check that runs without a GPU.

See ``_compare_logits`` for what replaced it and why that still catches real bugs.

Runs in its own process so CUDA context is fully torn down on exit.
"""

import argparse
import gc
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from granite_switch.composer import GraniteSwitchComposer

SEQ_LEN = 12  # Short fixed sequence — enough to exercise all layer types.

# How many ULP of disagreement to allow, as a multiple of the dtype's epsilon scaled
# by the logit magnitude. DERIVED, not a fixed atol: the sibling vLLM worker can use a
# flat 1e-5 because it compares logprobs, but this one compares raw logits in the
# model's native dtype, and one ULP there spans four orders of magnitude depending on
# dtype -- measured, at a logit of ~43 it is 5.2e-06 in fp32 but 3.4e-01 in bf16. A
# flat tolerance tight enough to mean anything in fp32 would reject plain arithmetic
# in bf16, and one loose enough for bf16 would be vacuous in fp32.
#
# 8 ULP, because the measured noise is far below that: a matmul accumulates in fp32
# and rounds once at the end, so two reduction orders usually produce the SAME output
# value and at worst adjacent ones. Simulating fused-vs-blockwise projections at
# H=2048 gave a max difference of 0 ULP in both dtypes, and an fp32 chunked-vs-whole
# dot product differed by ~1.4 ULP. 8 leaves room for a wider model without being the
# kind of bound that passes anything.
ULP_SLACK = 8

# Positions must also agree on WHICH tokens are most likely, not just on magnitudes.
# This is the load-bearing check, and the magnitude bound above is the weaker one: a
# constant added to every logit shifts max|diff| arbitrarily far while being invisible
# after softmax (log_softmax is shift-invariant), so magnitude alone can neither
# convict nor acquit. Ranking cannot be fooled that way -- a wrong weight permutation
# or a swapped K/V reorders the distribution, which is what a caller actually sees.
# Mirrors the vLLM worker, which compares top-K sets for the same reason.
TOP_K = 100


def _native_dtype(config):
    """Determine the model's native dtype from its HuggingFace config.

    Returns the dtype the model was published in.  Falls back to float32
    if the config doesn't specify (common for small research models).
    """
    dt = getattr(config, "torch_dtype", None)
    if dt is None:
        return torch.float32
    if isinstance(dt, torch.dtype):
        return dt
    if isinstance(dt, str):
        return getattr(torch, dt, torch.float32)
    return torch.float32


def _compare_logits(reference, switch, label, dtype):
    """Check skinned logits against the original's. Returns 0 on agreement, 1 on diff.

    Three checks, ordered most to least meaningful. They fail for different reasons,
    and reporting them separately is what makes a failure diagnosable rather than
    just "the numbers moved":

    1. **argmax agreement, exact.** Every position must rank the same token first.
       No tolerance -- this is what greedy decoding actually emits, and a skin that
       flips the top token is wrong however small the gap that flipped it.
    2. **top-K set agreement, exact.** The ``TOP_K`` most likely tokens must be the
       same set at every position. Catches a distribution that is subtly reordered
       below the argmax, which the vLLM worker checks for the same reason.
    3. **magnitude agreement, within ``ULP_SLACK`` ULP.** The weakest of the three
       and deliberately last: a uniform shift of every logit leaves the
       distribution identical (log_softmax is shift-invariant) yet moves max|diff|
       arbitrarily, so this check alone can neither convict nor acquit. It earns
       its place by catching the case the ranking checks cannot -- a scale change
       that preserves order, e.g. a wrong ``logits_scaling`` or
       ``attention_multiplier``.

    All three are printed either way, so a run drifting toward the bound is visible
    before it crosses it.
    """
    # Compare in fp32: a bf16 subtraction would itself round the difference being
    # measured, and the two tensors are already materialized at native precision.
    ref = reference.float()
    sw = switch.float()

    diff = (ref - sw).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    num_nonzero = int((diff > 0).sum().item())
    total = diff.numel()

    # Tolerance from the dtype's resolution at this model's logit magnitude.
    scale = max(ref.abs().max().item(), 1.0)
    tol = ULP_SLACK * torch.finfo(dtype).eps * scale

    ref_top = ref.argmax(dim=-1)
    sw_top = sw.argmax(dim=-1)
    argmax_mismatches = int((ref_top != sw_top).sum().item())

    # Top-K as a set per position: ties can order two equal logits either way, which
    # is not a regression, so compare membership rather than the sorted sequence.
    k = min(TOP_K, ref.shape[-1])
    ref_k = ref.topk(k, dim=-1).indices
    sw_k = sw.topk(k, dim=-1).indices
    flat_ref = ref_k.reshape(-1, k)
    flat_sw = sw_k.reshape(-1, k)
    topk_mismatches = sum(
        1
        for i in range(flat_ref.shape[0])
        if set(flat_ref[i].tolist()) != set(flat_sw[i].tolist())
    )

    print(
        f"  dtype       = {dtype}  (1 ULP at |logit|<={scale:.3g} is {tol / ULP_SLACK:.3e})"
    )
    print(f"  max  |diff| = {max_diff:.6e}   tolerance = {tol:.6e} ({ULP_SLACK} ULP)")
    print(f"  mean |diff| = {mean_diff:.6e}")
    print(f"  differing elements: {num_nonzero} / {total}")
    print(f"  argmax mismatches:  {argmax_mismatches} / {ref_top.numel()}")
    print(f"  top-{k} set mismatches: {topk_mismatches} / {flat_ref.shape[0]}")

    if argmax_mismatches:
        print(
            f"\nFAIL: {label} — {argmax_mismatches} position(s) predict a different "
            f"top token; the skin is not equivalent (dtype={dtype})"
        )
        return 1
    if topk_mismatches:
        print(
            f"\nFAIL: {label} — {topk_mismatches} position(s) have a different top-{k} "
            f"token set; the distribution is reordered (dtype={dtype})"
        )
        return 1
    if max_diff > tol:
        print(
            f"\nFAIL: {label} — logits differ by more than reduction-order noise "
            f"(max |diff| = {max_diff:.6e} > {tol:.6e}, dtype={dtype})"
        )
        return 1
    if max_diff == 0.0:
        print(f"\nPASS: {label} — bit-exact equivalence ({dtype})")
        return 0
    print(
        f"\nPASS: {label} — equivalent within {ULP_SLACK} ULP "
        f"(max |diff| = {max_diff:.6e} <= {tol:.6e}, dtype={dtype})"
    )
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="HuggingFace model name or local path")
    args = parser.parse_args()
    model_name = args.model

    # ── Resolve dtype ────────────────────────────────────────────
    print(f"Loading config for {model_name}...")
    base_config = AutoConfig.from_pretrained(model_name)
    dtype = _native_dtype(base_config)
    print(f"  model_type={base_config.model_type}  native_dtype={dtype}")

    # ── Phase 1: reference logits ─────────────────────────────────
    print(f"\nPhase 1: loading original model ({model_name})...")
    original = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).eval()

    # Fixed input_ids — avoids tokenizer dependency / compatibility issues.
    torch.manual_seed(42)
    input_ids = torch.randint(1, min(base_config.vocab_size, 1000), (1, SEQ_LEN))
    print(f"  input_ids shape: {input_ids.shape}")

    with torch.no_grad():
        reference_logits = original(input_ids=input_ids, use_cache=False).logits.cpu()
    print(f"  reference_logits shape: {reference_logits.shape}")

    del original
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("  original model freed")

    # ── Phase 2: build switch skin ────────────────────────────────
    print("\nPhase 2: building GraniteSwitch skin (num_adapters=0)...")
    switch_model = GraniteSwitchComposer.from_base_and_adapters(
        model_name,
        torch_dtype=dtype,
    ).eval()

    with torch.no_grad():
        switch_logits = switch_model(input_ids=input_ids, use_cache=False).logits.cpu()
    print(f"  switch_logits shape: {switch_logits.shape}")

    # ── Phase 3: compare ──────────────────────────────────────────
    print("\nPhase 3: comparing logits...")
    return _compare_logits(reference_logits, switch_logits, model_name, dtype)


if __name__ == "__main__":
    sys.exit(main())
