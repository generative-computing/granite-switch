# SPDX-License-Identifier: Apache-2.0
"""Subprocess worker: verify a skinned GraniteSwitch model is logit-exact vs the original.

Invoked as::

    python tests/composer/_skinning_equivalence_worker.py <model_name_or_path>

Loads the original model in its **native dtype** (the precision it was published
in), computes reference logits, frees it, then builds a GraniteSwitch skin
(num_adapters=0) via the builder's weight-transfer pipeline, and asserts
bit-exact equivalence.

Native dtype is critical: skinned models must be equivalent in the precision
users will actually deploy.  Float32 may mask real differences caused by
attention scaling order (pre-multiply Q vs post-divide scores).

Runs in its own process so CUDA context is fully torn down on exit.
"""

import argparse
import gc
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from granite_switch.composer import GraniteSwitchComposer

SEQ_LEN = 12  # Short fixed sequence — enough to exercise all layer types.


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

    diff = (switch_logits - reference_logits).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    num_nonzero = (diff > 0).sum().item()
    total = diff.numel()

    print(f"  max  |diff| = {max_diff:.6e}")
    print(f"  mean |diff| = {mean_diff:.6e}")
    print(f"  nonzero elements: {num_nonzero} / {total}")

    if max_diff == 0.0:
        print(f"\nPASS: {model_name} — bit-exact equivalence ({dtype})")
        return 0

    print(
        f"\nFAIL: {model_name} — logits differ (max |diff| = {max_diff:.6e}, dtype={dtype})"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
