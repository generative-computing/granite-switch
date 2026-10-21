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

DISTRIBUTION AGREEMENT IS CHECKED, NOT BIT-FOR-BIT EQUALITY.  This worker asserted
``max_diff == 0.0`` until it was found to be asking for something unreachable:
the skin uses fused QKV / gate-up projections while upstream HuggingFace uses
separate ones, and fusing changes the order of a floating-point sum.  ``a+b+c``
and ``(a+b)+c`` are not bit-identical, so a small disagreement is arithmetic,
not a defect, and the test could only ever fail.  It sat behind an unconditional
``@pytest.mark.skip`` for that reason, which cost the composer its only
weight-transfer check that runs without a GPU.

What replaced it is the same gate the vLLM generation-equivalence worker uses
(top-k JSD + Jaccard over a swept k), plus an exact argmax check and a slope fit.
See the constants below for the measured noise floor those are calibrated against,
and ``_compare_logits`` for why each of the four is there.

Runs in its own process so CUDA context is fully torn down on exit.
"""

import argparse
import gc
import os
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from granite_switch.composer import GraniteSwitchComposer

# Make tests.shared importable when run as a bare subprocess (cwd-independent),
# the same way the vLLM generation-equivalence worker does it.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from tests.shared.logit_metrics import jaccard, jsd_bits

SEQ_LEN = 12  # Short fixed sequence — enough to exercise all layer types.

# Distribution-agreement gates, and the same three the vLLM generation-equivalence
# worker uses -- deliberately, since both are asking the identical question (does a
# fused-projection skin match the unfused original) and a second set of numbers here
# would just be a second thing to keep calibrated.
#
# WHAT THESE REPLACED, AND WHY. This file first compared raw logit MAGNITUDES within
# an 8-ULP bound plus an exact top-100 SET match. Both were wrong for bf16 and the
# GPU run that caught it is worth recording:
#
#     max |diff| = 7.5e-01   tolerance = 9.1e-01 (8 ULP)   <- passed
#     argmax mismatches:  0 / 12                           <- passed
#     top-100 set mismatches: 11 / 12                      <- FAILED
#
# The top-K set check fails on a correct skin because bf16 has ~0.11 resolution at
# these logits, so the rank-100 boundary is a pile of EXACT TIES -- measured 4, 7 and
# 7 tokens sharing the identical bf16 value at the cut on the first three positions.
# Which of those tie-mates lands inside the cut is arbitrary, not a regression.
# Comparing sets rather than sorted order was an attempt to be tie-robust, but
# membership *at the boundary* is exactly as tie-sensitive as order is. Jaccard over
# a swept k replaces it: it measures HOW MUCH the top-k overlaps instead of demanding
# it match exactly, so a few tie-mates trading places costs a few percent instead of
# failing outright.
#
# The magnitude bound was the weaker problem: it is not wrong, just nearly vacuous.
# A uniform shift of every logit moves max|diff| arbitrarily while leaving the
# distribution identical (log_softmax is shift-invariant), so it can neither convict
# nor acquit. JSD is measured on log_softmax output and is therefore immune to that
# by construction.
#
# MEASURED on granite-4.0-micro (bf16, 12 positions, k swept over 1..100): mean JSD
# 6.5e-04, max JSD 1.9e-03, mean(1-Jaccard) 0.094. The gates sit ~2-25x above that
# floor. Verified to still convict: a swapped K/V (max JSD 0.52), a 100-token weight
# permutation (0.77), one corrupted position (0.046), a +0.9 per-token shift (mean
# JSD 0.030) and a 30% logit scaling (0.024) all fail.
MEAN_JSD_THRESH = 0.003
MAX_JSD_THRESH = 0.05
JACC_THRESH = 0.20

# Swept rather than fixed, because the two failure modes live at opposite ends: a
# flipped near-tie only shows up at small k, and a reordered tail only at large k.
K_SWEEP = (1, 5, 10, 20, 50, 100)

# Max allowed |a-1| where `a` is the least-squares slope of skinned-vs-reference
# logits.
#
# This check exists because the three gates above are all computed AFTER
# log_softmax, which makes them blind to one real bug: a uniform SCALING of every
# logit. Scaling preserves ranking (so argmax and Jaccard see nothing) and barely
# dents the normalized shape at small factors -- a 5% error measured mean JSD
# 7.2e-04, inside the benign floor. That is not a hypothetical: it is the signature
# of a wrong ``logits_scaling`` or ``attention_multiplier``, the two Granite config
# multipliers a skin could plausibly mistransfer, and `logits_scaling` is typically
# 8.0 so a one-step error is a ~12% scale change.
#
# A slope fit separates it cleanly because a scale error is SYSTEMATIC across all
# 1.2M elements while reduction-order noise is random: measured |a-1| = 3.2e-05 for
# the real skin versus 1.0e-02 for a 1% scaling -- a 300x gap. The intercept is left
# free, so a benign uniform shift still reads as slope 1 exactly.
SLOPE_TOL = 1e-3


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


def _slope(ref, sw):
    """Least-squares slope of ``sw`` against ``ref``, intercept free.

    Intercept free on purpose: a uniform additive shift is invisible after softmax,
    so it must not register here either. Returns 1.0 exactly for a shifted copy.
    """
    x = ref.reshape(-1)
    y = sw.reshape(-1)
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / (xc * xc).sum())


def _compare_logits(reference, switch, label, dtype):
    """Check skinned logits against the original's. Returns 0 on agreement, 1 on diff.

    Four checks. The first three are computed on ``log_softmax`` output -- the
    distribution a caller actually observes -- and the fourth covers the one real
    bug that normalizing hides. They fail for different reasons, and reporting them
    separately is what makes a failure diagnosable rather than just "the numbers
    moved":

    1. **argmax agreement, exact.** Every position must rank the same token first.
       No tolerance: this is what greedy decoding emits, and a skin that flips the
       top token is wrong however small the gap that flipped it. Safe to demand
       exactly even in bf16 -- measured, the argmax is a strict maximum at all 12
       positions (no tie-mates), unlike the rank-100 boundary.
    2. **mean and max JSD over a swept k.** The primary signal. Mean catches a
       distribution that is wrong everywhere a little; max catches one position
       that is catastrophically wrong, which a mean over 12 positions would dilute.
    3. **mean top-k Jaccard distance.** A loose set-overlap guard, same role as in
       the vLLM worker. Deliberately loose: at these bf16 logits several tokens tie
       exactly at the k-th rank, so a few percent of churn is arithmetic.
    4. **slope of skinned-vs-reference logits.** The only check here that is NOT
       scale-invariant, which is precisely its job -- see ``SLOPE_TOL``.

    All four are printed either way, so a run drifting toward a bound is visible
    before it crosses it.
    """
    # Compare in fp32: a bf16 subtraction would itself round the difference being
    # measured, and the two tensors are already materialized at native precision.
    ref = reference.float()[0]
    sw = switch.float()[0]

    ref_lp = torch.log_softmax(ref, dim=-1)
    sw_lp = torch.log_softmax(sw, dim=-1)

    argmax_mismatches = int((ref.argmax(dim=-1) != sw.argmax(dim=-1)).sum().item())
    slope = _slope(ref, sw)
    num_pos = ref.shape[0]

    print(f"  dtype = {dtype}   positions = {num_pos}   vocab = {ref.shape[-1]}")
    print(f"  argmax mismatches: {argmax_mismatches} / {num_pos}")
    print(f"  slope |a-1| = {abs(slope - 1):.3e}   tolerance = {SLOPE_TOL:.3e}")
    print(f"  {'k':>5}{'mean(1-Jacc)':>15}{'mean JSD':>12}{'max JSD':>12}{'@pos':>7}")

    failures = []
    for k in K_SWEEP:
        if k > ref.shape[-1]:
            continue
        jacc_dists = []
        jsds = []
        for pos in range(num_pos):
            a = ref[pos].topk(k).indices.tolist()
            b = sw[pos].topk(k).indices.tolist()
            ids = sorted(set(a) | set(b))
            # Dicts only over the union, not the whole 100k vocab: jsd_bits
            # renormalizes over `ids` anyway, so the tail would be discarded.
            p_lp = {t: ref_lp[pos, t].item() for t in ids}
            q_lp = {t: sw_lp[pos, t].item() for t in ids}
            jacc_dists.append(1.0 - jaccard(a, b))
            jsds.append(jsd_bits(p_lp, q_lp, ids))

        mean_jd = sum(jacc_dists) / len(jacc_dists)
        mean_js = sum(jsds) / len(jsds)
        max_js = max(jsds)
        max_pos = jsds.index(max_js)
        print(f"  {k:5d}{mean_jd:15.5f}{mean_js:12.6f}{max_js:12.6f}{max_pos:7d}")

        if mean_jd > JACC_THRESH:
            failures.append(f"k={k}: mean(1-Jaccard)={mean_jd:.4f} > {JACC_THRESH}")
        if mean_js > MEAN_JSD_THRESH:
            failures.append(f"k={k}: mean JSD={mean_js:.6f} > {MEAN_JSD_THRESH}")
        if max_js > MAX_JSD_THRESH:
            failures.append(
                f"k={k}: max JSD={max_js:.6f} @pos {max_pos} > {MAX_JSD_THRESH}"
            )

    if argmax_mismatches:
        print(
            f"\nFAIL: {label} — {argmax_mismatches} position(s) predict a different "
            f"top token; the skin is not equivalent (dtype={dtype})"
        )
        return 1
    if abs(slope - 1) > SLOPE_TOL:
        print(
            f"\nFAIL: {label} — skinned logits are scaled by {slope:.6f} relative to "
            f"the original (|a-1| = {abs(slope - 1):.3e} > {SLOPE_TOL:.3e}); suspect a "
            f"mistransferred logits_scaling / attention_multiplier (dtype={dtype})"
        )
        return 1
    if failures:
        print(f"\nFAIL: {label} — distributions disagree beyond the noise floor:")
        for line in failures:
            print(f"  {line}")
        return 1
    print(
        f"\nPASS: {label} — distribution-equivalent (mean JSD <= {MEAN_JSD_THRESH}, "
        f"max JSD <= {MAX_JSD_THRESH}, mean 1-Jaccard <= {JACC_THRESH}, "
        f"slope within {SLOPE_TOL:.0e}, dtype={dtype})"
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
