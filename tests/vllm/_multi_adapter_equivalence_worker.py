# SPDX-License-Identifier: Apache-2.0
"""Subprocess worker: multi-adapter routing equivalence under vLLM.

Sibling of :mod:`_generation_equivalence_worker`, which pins a ONE-adapter
zero-delta checkpoint against upstream. This one pins the properties that only
exist once a checkpoint carries SEVERAL adapters that are really trained::

    python worker.py prepare --base <model> --flavor <f> --work-dir <d>
    python worker.py build   --work-dir <d> --order forward|reversed
    python worker.py cases   --work-dir <d> --order forward|reversed
    python worker.py run     --model <name-or-path> --work-dir <d> --tag <t> \
                             --cases <file> [--only base] [--eager true|false]
    python worker.py compare --work-dir <d> --check <c> --label <l>

Fitted adapters, real placement
-------------------------------
Every adapter is generated from a profile fitted to a published granitelib adapter
— delta norm relative to the base weight, singular-value spectrum, and A/B
asymmetry all reproduced (tests/shared/synthetic_adapters.py). Nothing is
downloaded, and no base model needs adapters published against it, which is what
lets the same suite run on a base where none exist.

The compose runs through the ``compose_granite_switch`` CLI, so the checkpoint gets
its tokenizer and chat template configured exactly as a shipped one does. Prompts
are then rendered with ``apply_chat_template(..., adapter_name=...)``, which means
each adapter's control token lands where its OWN technology puts it: at the
sequence start for LoRA, at the invocation point for aLoRA, at the
generation-prompt anchor for Shadow Residual. Nothing about placement is
hand-chosen here.

Because placement uses token-exchange SUBSTITUTES, an adapter's prompt has the
same length as the no-adapter prompt and differs from it at exactly one position
(asserted in ``cases``). Positions are therefore directly comparable across
cases, and every case is scored on ``prompt + continuation`` where the
continuation is the base case's own greedy output — so the decision positions
are covered too, which matters for aLoRA and SR whose control token sits near
the end of the prompt.

The profiles were fitted on granite-4.1-3b but are expressed relative to each
layer's base weight, so they carry to any base — the aLoRA invocation sequence and
the SR activation anchor are likewise read off that base's own tokenizer rather
than hardcoded, which is why both Granite template families work.

The four checks
---------------
``passthrough``
    The multi-adapter checkpoint with NO control token fired == the upstream base
    model. Adapter id 0 must mean "no delta" even though the checkpoint's
    adapters carry real, trained weights. The reference is the unmodified
    upstream model, so this is a fused-vs-native comparison and uses the looser
    gates.

``slots``
    The SAME adapter composed at a DIFFERENT slot produces the same output. Two
    checkpoints hold the same adapters in opposite order, so adapter X sits at
    slot 0 in one and slot N-1 in the other, and every per-slot kernel table —
    the per-module remap tables, the rank-ordered tier assignment, the per-tile
    bitmasks, the control-token ids themselves — is permuted between the two
    while the mathematics is not. Wrong-slot routing, a tier-ordering bug, or a
    remap-table off-by-one shows up here and nowhere else.
    tests/composer/test_multi_adapter_compose.py establishes on CPU that the
    composed weights are slot-independent, so a difference here is a runtime
    routing bug and not a compose artifact.

``batch``
    A prompt's output does not depend on what else is in the batch: each case is
    scored alone and then all cases are submitted together. The aLoRA and SR
    cases additionally hold base tokens and adapter tokens in ONE sequence, which
    is guaranteed intra-forward mixing regardless of how the scheduler batches.
    Gated against the BASE case as a control — see below.

``distinct``
    Anti-vacuity: every adapter must actually change the distribution versus
    base, and no two adapters may produce the same distribution. Without it,
    adapters that did nothing would pass ``slots`` and ``batch`` trivially.
    Compared only at positions after the adapter's control token, since the
    prompt before it is identical to base by construction.

Absolute thresholds only where the quantity is exact
----------------------------------------------------
Two of the four comparisons have a real noise floor that depends on the model,
the vLLM version and the GPU, so an absolute threshold for them is a guess that
either passes everything or fails on the wrong hardware. Both are therefore
gated against a control measured in the SAME run:

* ``batch`` against the base case, where no adapter is active. vLLM is not
  batch-invariant: on granite-4.1-3b the no-adapter case alone moves by mean JSD
  0.0027 (max 0.133) between a solo and a batched submission. An adapter case is
  allowed ``BATCH_REL_FACTOR`` times that, so the check asks the only question
  that is about routing — does an adapter add anything beyond what the model
  already does — instead of measuring vLLM's scheduler.
* ``passthrough`` against the smallest adapter effect ``distinct`` measures. "No
  delta" is then a statement of scale ("under a tenth of what a real adapter
  does") rather than an absolute bit count, and it travels across base models.
  Its outlier gate is a FRACTION of positions above a JSD, not the maximum: one
  near-tie position flipping is expected, a systematic logit error would move
  many.

``slots`` and ``distinct`` keep absolute gates, because they measure quantities
that are not noise-floor-bound: ``slots`` came out bit-exact, and ``distinct``
sits two orders of magnitude above any floor.

Eager and compiled
------------------
``GraniteSwitchModel`` is wrapped in ``@support_torch_compile``, so an
``enforce_eager=True`` run skips the path a default server takes. That matters
here specifically: the per-token routing metadata is recomputed every forward and
stashed on a shared ctx object, and for SR the decoder doubles the token axis to
``[2M, H]`` INSIDE the compiled region — a mis-specialized dynamic shape or a
graph holding a stale length is exactly the multi-adapter bug this suite is for,
and eager mode cannot see it.

So both slot orders are captured twice, ``<order>`` (eager) and
``<order>_compiled``, and ``slots`` / ``batch`` gate every mode present.

Prompt scoring is prefill, so nothing here covers the graph-captured decode path.
A ``decode`` check used to: a greedy continuation, generated token by token, had
to be identical across slot orders. It was removed because exact token equality
is unsatisfiable on vLLM 0.20, and not for any reason to do with slot order --
loading ONE checkpoint twice and comparing it against itself diverges at the same
two token indices, with nothing permuted. A discrete-token check has no tolerance
to widen, so there was nothing to recalibrate. The cost is that the decode path
is now untested by this suite; the benefit is that no gate here asserts something
the runtime does not promise.
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

# Make tests.shared importable when run as a bare subprocess (cwd-independent).
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from tests.shared.logit_metrics import jaccard, jsd_bits, topk_ids
from tests.shared.synthetic_adapters import (
    PROFILES,
    adapter_facts,
    alora_invocation_ids,
    attn_modules,
    base_weight_norms,
    measure_adapter,
    mlp_modules,
    sr_activation_anchor,
    synthesize_adapter,
)

TOPK = int(os.environ.get("MULTI_ADAPTER_TOPK", "64"))
K_SWEEP = [int(x) for x in os.environ.get("MULTI_ADAPTER_KS", "1,5,10,20").split(",")]
assert max(K_SWEEP) <= TOPK, f"max(K_SWEEP)={max(K_SWEEP)} must be <= TOPK={TOPK}"

GEN_TOKENS = int(os.environ.get("MULTI_ADAPTER_GEN_TOKENS", "32"))
MAX_MODEL_LEN = int(os.environ.get("MULTI_ADAPTER_MAX_MODEL_LEN", "4096"))
#: Ceiling on vLLM's gpu_memory_utilization, which is a fraction of TOTAL memory
#: and so assumes an empty card. On a shared cluster that is wrong: a run failed to
#: start with 61.26 of 79.25 GiB free because another tenant held 18 GiB and 0.85
#: asked for 67.36. :func:`_gpu_mem_util` lowers it to fit what is actually free.
GPU_MEM_UTIL = float(os.environ.get("MULTI_ADAPTER_GPU_MEM_UTIL", "0.85"))
#: Below this, the card is too full to be worth starting on -- a 3B model plus its
#: KV cache needs roughly a tenth of an 80 GiB card, so this is generous.
GPU_MEM_UTIL_FLOOR = float(os.environ.get("MULTI_ADAPTER_GPU_MEM_UTIL_FLOOR", "0.15"))

# ── Gates ────────────────────────────────────────────────────────────
# Two of the four checks compare quantities with a real noise floor, and that
# floor depends on the model, the vLLM version and the GPU. Measured on
# granite-4.1-3b (40 layers, bf16, 1x A100, vLLM 0.19.1, 218 scored positions):
#
#   slots        mean JSD 0.000000, max 0.000000 — BIT-EXACT, both flavors.
#   distinct     mean JSD 0.095 .. 0.329 between any two cases.
#   batch        the BASE case (no adapter active at all) already diverges
#                solo-vs-batched by mean 0.0027 / max 0.133: vLLM is not
#                batch-invariant, independently of adapters. Every adapter case
#                came in at or below that on the mean; at k=1, where the compared
#                support is one or two tokens, one adapter's p99 reached 0.0111
#                against the control's 0.0028 — a tie-flip, not a routing
#                difference, which is why the budget also has an absolute floor
#                tied to the size of a real adapter effect.
#   passthrough  base-vs-upstream mean JSD 0.0027 (lora) / 0.0027 (sr), with a
#                single outlier position at 0.133 / 0.072 while k=1 mean stays
#                at 3.5e-4 — localized near-tie flips, not a systematic error.
#                The sibling test's 5e-4 / 3.5e-3 floor was measured on a
#                ZERO-adapter granite-4.0-micro, where the fused kernel takes a
#                bare base-GEMM path; it does not transfer to a populated
#                multi-adapter checkpoint 40 layers deep.
#
# So only `slots` — the actual multi-adapter claim, and the one measured exact —
# gets an absolute gate. `batch` is gated against the base case as its own
# control, and `passthrough` against the size of a real adapter effect measured
# in the same run. Both are then independent of model, version and hardware.

#: slots: same weights, same kernel, only the slot permuted. Measured exactly 0
#: under EAGER on both vLLM 0.19.1 and 0.20.2, so the eager arm keeps an almost
#: exact gate -- ~100x below every other comparison's floor rather than at 0,
#: since bit-exactness is not something a fused kernel owes across GPUs.
SLOTS_MEAN_JSD = float(os.environ.get("MULTI_ADAPTER_SLOTS_MEAN_JSD", "1e-4"))
SLOTS_MAX_JSD = float(os.environ.get("MULTI_ADAPTER_SLOTS_MAX_JSD", "1e-3"))
SLOTS_JACC = float(os.environ.get("MULTI_ADAPTER_SLOTS_JACC", "0.02"))

# The COMPILED arm cannot hold that gate, and the reason is not slot order.
# vLLM 0.20.2 does not reproduce a compiled SR run across processes: loading ONE
# checkpoint twice and comparing it against itself -- nothing permuted, no
# reversed order built -- reproduces the CI failure's numbers to every digit
# printed (base k=1 mean 0.000319 / max 0.043713 @184; syn-sr-weak k=1 mean
# 0.001251 / max 0.157874 @197), and the same two greedy continuations diverge at
# the same two token indices. Slot permutation contributes nothing measurable.
# Eager is exactly 0 in the same run, so the routing itself is fine; what varies
# is the compiled graph between processes. 0.19.1 has no such variance.
#
# Sizing, from that same-checkpoint measurement (the noise) against `distinct`'s
# inter-adapter distances (the signal a real break would produce):
#   noise   mean <= 0.0028,  1-Jaccard <= 0.060,  positions over 0.05 <= 0.46%
#   signal  an adapter moves the distribution ~0.10-0.13 bits at MOST positions
# So the compiled mean gate sits at 0.01: ~3.5x above the noise and ~10x below a
# real break. Jaccard at 0.15, ~2.5x above the noise.
SLOTS_COMPILED_MEAN_JSD = float(
    os.environ.get("MULTI_ADAPTER_SLOTS_COMPILED_MEAN_JSD", "0.01")
)
SLOTS_COMPILED_JACC = float(
    os.environ.get("MULTI_ADAPTER_SLOTS_COMPILED_JACC", "0.15")
)
#: ...and max JSD is dropped for the compiled arm rather than loosened, because it
#: CANNOT discriminate here: the noise reaches 0.157874 at one position, which is
#: larger than a whole adapter effect, so no ceiling separates the two. The
#: fraction of positions above an adapter-sized cut does separate them -- noise
#: puts <=0.46% there, a systematic routing error would put most of them -- which
#: is the same substitution `passthrough` already makes for the same reason.
SLOTS_COMPILED_OUTLIER_FRAC = float(
    os.environ.get("MULTI_ADAPTER_SLOTS_COMPILED_OUTLIER_FRAC", "0.02")
)

#: batch: an adapter case may not diverge more than this multiple of what the
#: no-adapter BASE case diverges by on the same submission...
BATCH_REL_FACTOR = float(os.environ.get("MULTI_ADAPTER_BATCH_REL_FACTOR", "1.5"))
#: p99 gets a looser factor than the mean, because the two are not comparable in
#: the same way. The control carries no delta, so its near-ties are shallower, and
#: an adapter case amplifies the SAME numeric perturbation into a larger JSD swing
#: purely because its distribution is sharper. Measured with every mean inside
#: budget: one adapter's p99 reached 2.3x the control's at roughly two positions
#: out of 218. 4x leaves headroom for that while still catching gross divergence —
#: the mutation run's routing break reached max JSD 0.82.
BATCH_P99_REL_FACTOR = float(
    os.environ.get("MULTI_ADAPTER_BATCH_P99_REL_FACTOR", "4.0")
)
#: k=1 is reported but NOT gated on the noise-floor-bound comparison. At k=1 the
#: compared support is the top-1 union -- one or two tokens -- so JSD there is not
#: measuring distribution shape at all: it measures whether the top token's mass
#: shifted, which for a near-tie is a binary event whose magnitude is set by how
#: close the tie was rather than by how large the perturbation was. It produced a
#: spurious batch failure three separate times (three different adapters, always
#: with k>=5 comfortably inside budget and the k=1 value negligible in absolute
#: terms) before this was recognised as structural rather than a threshold to
#: tune. Gates whose expectation is exact -- slots -- keep every k, because there
#: no support is too narrow for "identical" to mean something.
BATCH_MIN_GATED_K = int(os.environ.get("MULTI_ADAPTER_BATCH_MIN_GATED_K", "5"))
#: ...or more than this fraction of a real adapter effect, whichever is larger.
#: The second term is what makes the gate usable at small k. At k=1 the compared
#: support is the top-1 union — one or two tokens — so JSD is a knife-edge on
#: near-ties: measured on granite-4.1-3b the control's k=1 p99 was 0.0028, making
#: a purely relative budget 0.0042, which one tie-flip clears (the `uncertainty`
#: adapter came in at 0.0111). In absolute terms 0.011 bits at 1% of positions is
#: nothing — 3.5% of that adapter's own 0.310-bit effect — so the floor says so
#: explicitly instead of letting the ratio of two tiny numbers decide. A real
#: routing bug moves a case by an adapter-sized amount, far above this.
BATCH_EFFECT_FRACTION = float(
    os.environ.get("MULTI_ADAPTER_BATCH_EFFECT_FRACTION", "0.10")
)
#: There is deliberately NO absolute floor here. An earlier revision added one
#: (0.004 mean / 0.05 p99, from the granite-4.1-3b measurements) to survive a run
#: whose adapters were functionally weak. It worked on that model and quietly
#: broke the gate everywhere else: the CPU scenario for "an adapter diverges far
#: more than its control" started PASSING, because a constant in one model's JSD
#: scale swallows another's signal. The two relative terms -- the control and the
#: measured adapter effect -- carry the same protection without pinning the gate
#: to one model's numbers.
#: A control this large would mean the model itself is unstable, not that
#: batching is noisy — worth failing on rather than calibrating against.
BATCH_CONTROL_CEILING = float(
    os.environ.get("MULTI_ADAPTER_BATCH_CONTROL_CEILING", "0.05")
)

#: passthrough: adapter id 0 must be a no-op, measured against how much a real
#: adapter moves the same distribution — 10% of the SMALLEST adapter effect in
#: this run. Plus a localized-vs-systematic discriminator: a few near-tie
#: positions may flip (they do), but a systematic logit error would push a large
#: FRACTION of positions over the outlier threshold, not one or two.
PASS_EFFECT_FRACTION = float(
    os.environ.get("MULTI_ADAPTER_PASS_EFFECT_FRACTION", "0.10")
)
#: The floor no implementation can beat, so the budget never goes below it. The
#: fused SWITCH kernel does not reduce in the same order as vLLM's native linear
#: (CLAUDE.md gotcha 9), and on granite-4.1-3b that difference alone measures mean
#: JSD 0.0027-0.0030 over a 40-layer bf16 stack. An effect-relative budget below
#: it is unsatisfiable by correct code -- which is exactly what synthetic adapters
#: produced: 0.1 x a 0.0116-bit effect is 0.0012, under the floor.
PASS_FLOOR_MEAN = float(os.environ.get("MULTI_ADAPTER_PASS_FLOOR_MEAN", "0.004"))
#: A position counts as an outlier when its JSD reaches this multiple of the mean
#: budget — i.e. when ONE position diverges by as much as a whole real adapter
#: does on average. Scale-free for the same reason the mean gate is: on
#: granite-4.1-3b the worst position measured 0.133 (lora) / 0.072 (sr) against
#: adapter effects of 0.232 / 0.095, so a fixed 0.05 would have flagged ordinary
#: near-tie flips while missing an adapter-sized error on a stronger adapter.
PASS_OUTLIER_MULT = float(os.environ.get("MULTI_ADAPTER_PASS_OUTLIER_MULT", "10"))
PASS_OUTLIER_FRAC = float(os.environ.get("MULTI_ADAPTER_PASS_OUTLIER_FRAC", "0.02"))
#: Reporting default for every other check's ``over`` column.
PASS_OUTLIER_JSD = float(os.environ.get("MULTI_ADAPTER_PASS_OUTLIER_JSD", "0.05"))
PASS_JACC = float(os.environ.get("MULTI_ADAPTER_PASS_JACC", "0.20"))

#: Minimum mean JSD (bits) for two distributions to count as genuinely
#: different. Guards vacuity only, so it sits well below a trained adapter's
#: effect (measured >= 0.095) and well above the reduction-order floor.
DISTINCT_MIN_JSD = float(os.environ.get("MULTI_ADAPTER_DISTINCT_MIN_JSD", "0.01"))

# ── Adapter sets ─────────────────────────────────────────────────────────────
# Every adapter is generated from a profile fitted to a published granitelib
# adapter (tests/shared/synthetic_adapters.py). Nothing is downloaded, and no base
# model needs adapters published against it — which is what lets the same suite run
# on granite-4.2-3b, where none exist.
#
# The set is chosen for variety in what the routing tables are indexed by:
# technology (which decides control-token placement), rank (which kernel tier the
# adapter lands in), and module coverage (which modules have an adapter at all).
#
#   (name, profile, technology, adapts the MLP)
SYNTHETIC_LORA_FLAVOR = [
    ("syn-light", "light", "lora", False),  # r=16, fitted to citations
    ("syn-strong", "strong", "lora", False),  # r=64, fitted to requirement-check
    ("syn-medium", "medium", "alora", False),  # r=32, fitted to uncertainty
    ("syn-weak", "weak", "alora", True),  # r=16 + MLP, fitted to answerability
]

#: (name, profile, cross_stream rank, effect_scale override). All adapt the MLP:
#: SR wraps those projections unconditionally, so leaving them out would idle a
#: tier.
#:
#: SR needs its own strengths. Its control token lands at the generation-prompt
#: anchor -- position 186 of a 187-token prompt -- so ONLY the continuation is
#: adapter-active, and it does not adapt K/V either. The same profile therefore
#: moves the distribution about 3x less than on the single-stream path: measured
#: 0.0039 / 0.026 / 0.063 bits here against 0.012 / 0.042 / 0.111 there. The
#: overrides below lift the weakest into the same band as the others while keeping
#: every resulting norm inside the range trained adapters occupy
#: (||delta||_F / ||W||_F of 0.008 to 0.42).
SYNTHETIC_SR_FLAVOR = [
    ("syn-sr-weak", "weak", 16, 24.0),
    ("syn-sr-medium", "medium", 32, 2.0),
    ("syn-sr-light", "light", 64, 4.0),
]

#: A short, deterministic chat. The question needs the document to answer, so a
#: template that drops the document changes what is being scored — see
#: :func:`chat_inputs`.
MESSAGES = [
    {
        "role": "user",
        "content": "According to the document, what colour is the roof of the observatory?",
    }
]
DOCUMENTS = [
    {
        "title": "Observatory",
        "text": (
            "The Mount Herschel observatory was completed in 1974. Its dome is "
            "clad in anodised aluminium panels with a pale copper-green finish, "
            "chosen to limit thermal expansion. The visitor centre below it has "
            "a slate roof."
        ),
    }
]


def chat_inputs(tokenizer):
    """``(messages, documents, template_kwargs)`` this tokenizer actually honours.

    The granite-4.2 ChatML template takes no ``documents`` argument and silently
    drops it: passing the same call to both families rendered 187 tokens on
    granite-4.1-3b and 30 on 4.2-3b, with the document — which the question needs —
    gone. Rather than sniff the template text, render it twice and see whether the
    argument changed anything; if it did not, inline the documents into the user
    message the way the 4.2 training format does. ``enable_thinking=False`` is
    passed when the template accepts it, so a thinking-enabled default cannot add
    structure the anchors were not derived from.
    """
    extra = {}
    try:
        tokenizer.apply_chat_template(
            MESSAGES, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        extra["enable_thinking"] = False
    except (TypeError, ValueError):
        pass

    def render(**kw):
        return tokenizer.apply_chat_template(
            MESSAGES, tokenize=False, add_generation_prompt=True, **extra, **kw
        )

    honours_documents = render(documents=DOCUMENTS) != render()
    if honours_documents:
        print(f"chat template honours `documents` (kwargs={extra})")
        return MESSAGES, DOCUMENTS, extra

    inlined = "Documents:\n\n" + "\n\n".join(
        f"Document {i}\n{d['text']}" for i, d in enumerate(DOCUMENTS)
    )
    messages = [
        {**MESSAGES[0], "content": f"{inlined}\n\n{MESSAGES[0]['content']}"},
        *MESSAGES[1:],
    ]
    print(f"chat template drops `documents`; inlined into the message (kwargs={extra})")
    return messages, None, extra


# ── prepare mode ──────────────────────────────────────────────────────


def _native_dtype(config):
    """The base checkpoint's own dtype, so synthesized adapters match it."""
    import torch

    dt = getattr(config, "torch_dtype", None) or getattr(config, "dtype", None)
    if isinstance(dt, torch.dtype):
        return dt
    if isinstance(dt, str):
        return getattr(torch, dt, torch.float32)
    return torch.float32


def _local_base(base: str) -> Path:
    """Local directory for the base model, downloading it if it is a repo id."""
    if os.path.isdir(base):
        return Path(base)
    from huggingface_hub import snapshot_download

    # No allow_patterns: an incomplete snapshot is worse than a slower one. A
    # filtered list omitted chat_template.jinja, where recent transformers keeps
    # the template, and prepare died on a cold cache with "chat_template is not
    # set" while passing locally against a warm one. The compose step needs the
    # whole checkpoint anyway.
    return Path(snapshot_download(base))


def _prepare_synthetic(args, work_dir: Path, base_name: str):
    """Generate the adapter set from profiles fitted to trained adapters."""
    from transformers import AutoConfig, AutoTokenizer

    from tests.shared.base_models import dims_from_config

    base_dir = _local_base(args.base)
    config = AutoConfig.from_pretrained(str(base_dir))
    dims = dims_from_config(config)
    dtype = _native_dtype(config)
    tokenizer = AutoTokenizer.from_pretrained(str(base_dir))
    print(f"reading base weight norms from {base_dir} ...")
    norms = base_weight_norms(base_dir)
    print(f"  {len(norms)} per-layer projection norms; dims={dims}")

    # One explicit plan entry per adapter. `tech` is the DIRECTORY name, and the
    # composer reads technology from there rather than from the config keys — an
    # aLoRA adapter under lora/ would have its control token placed at the
    # sequence start, silently losing the mid-prompt activation this suite tests.
    # SR is exempt: it is detected from the weights, which overrides the label.
    if args.flavor == "sr":
        messages, documents, template_kwargs = chat_inputs(tokenizer)
        anchor = sr_activation_anchor(tokenizer, messages, documents, **template_kwargs)
        print(f"SR activation anchor for {base_name}: {anchor[0]!r} (id {anchor[1]})")
        plan = [
            {
                "name": name,
                "profile": replace(PROFILES[profile_key], effect_scale=scale),
                "profile_key": profile_key,
                "tech": "lora",
                # SR never adapts K/V (base stream) and wraps the MLP
                # unconditionally, so leaving it out would idle a tier.
                "modules": attn_modules(with_kv=False) + mlp_modules(dims),
                "cross_stream_rank": cross,
                "alora_invocation_tokens": None,
                "last_context_token": anchor,
            }
            for name, profile_key, cross, scale in SYNTHETIC_SR_FLAVOR
        ]
    else:
        messages, documents, template_kwargs = chat_inputs(tokenizer)
        invocation = alora_invocation_ids(
            tokenizer, messages, documents, **template_kwargs
        )
        print(f"aLoRA invocation ids for {base_name}: {invocation}")
        plan = [
            {
                "name": name,
                "profile": PROFILES[profile_key],
                "profile_key": profile_key,
                "tech": tech,
                # MLP coverage varies per adapter, so those modules have
                # adapters that do not apply to every slot.
                "modules": attn_modules() + (mlp_modules(dims) if with_mlp else []),
                "cross_stream_rank": None,
                "alora_invocation_tokens": invocation if tech == "alora" else None,
                "last_context_token": None,
            }
            for name, profile_key, tech, with_mlp in SYNTHETIC_LORA_FLAVOR
        ]

    entries = []
    for index, item in enumerate(plan):
        dest = lib_dir(work_dir) / f"{item['name']}/{base_name}/{item['tech']}"
        synthesize_adapter(
            dest,
            dims,
            norms,
            item["profile"],
            seed=9_000 + index,
            modules=item["modules"],
            dtype=dtype,
            cross_stream_rank=item["cross_stream_rank"],
            alora_invocation_tokens=item["alora_invocation_tokens"],
            last_context_token=item["last_context_token"],
        )
        facts = adapter_facts(dest)
        facts["profile"] = item["profile_key"]
        if item["cross_stream_rank"]:
            facts["cross_stream_rank"] = item["cross_stream_rank"]
        # Re-measure what was written, so the log shows the delta norm, spectrum
        # and asymmetry the composed checkpoint actually carries rather than what
        # the profile asked for.
        facts["measured"] = [
            {
                k: row[k]
                for k in (
                    "layer",
                    "module",
                    "rank",
                    "rel_fro",
                    "stable_rank",
                    "b_over_a",
                )
            }
            for row in measure_adapter(dest, norms, layers=(0, dims.num_layers - 1))
            if row["module"] in ("self_attn.q_proj", "cross_stream")
        ]
        entries.append({"name": item["name"], "dir": str(dest), "facts": facts})
    return entries


def lib_dir(work_dir: Path) -> Path:
    """Where both adapter sources lay out their ``<name>/<base>/<tech>/`` tree."""
    path = Path(work_dir) / "lib"
    path.mkdir(parents=True, exist_ok=True)
    return path


def cmd_prepare(args):
    """Generate the adapter set into the work dir."""
    work_dir = Path(args.work_dir)
    base_name = args.base.rstrip("/").split("/")[-1]
    entries = _prepare_synthetic(args, work_dir, base_name)
    _report_and_save(args, work_dir, entries)
    return 0


def _report_and_save(args, work_dir: Path, entries):
    """Print what was prepared, refuse a no-op adapter, and persist the manifest."""
    print(f"prepared {len(entries)} synthetic {args.flavor} adapters for {args.base}:")
    for e in entries:
        f = e["facts"]
        print(
            f"  {e['name']:<26} rank={f['rank']:<4} alpha={f['alpha']:<5} "
            f"alora={f['alora']!s:<5} cross={f.get('cross_stream_rank', '-'):<4} "
            f"A.std={f['lora_A_std']:.4g} B.std={f['lora_B_std']:.4g} "
            f"zeroB={f['num_zero_lora_B']}/{f['num_lora_B']}"
        )
        print(f"    modules: {','.join(f['target_modules'])}")
        # A trained adapter has no all-zero lora_B. One that did would be a
        # silent no-op and would make the equality gates vacuous.
        if f["num_zero_lora_B"]:
            raise SystemExit(
                f"FATAL: {e['name']} has {f['num_zero_lora_B']} all-zero lora_B "
                "tensors — it would contribute no delta."
            )
        for m in f.get("measured", ()):
            print(
                f"    L{m['layer']:<3} {m['module']:<18} r={m['rank']:<4}"
                f"rel_fro={m['rel_fro']:.5f} srank={m['stable_rank']:.2f} "
                f"b/a={m['b_over_a']:.3f}"
            )

    (work_dir / "adapters.json").write_text(
        json.dumps(
            {"base": args.base, "flavor": args.flavor, "adapters": entries}, indent=2
        )
    )


# ── build mode ────────────────────────────────────────────────────────


def cmd_build(args):
    """Compose one checkpoint via the compose CLI, in the requested slot order."""
    work_dir = Path(args.work_dir)
    spec = json.loads((work_dir / "adapters.json").read_text())
    dirs = [e["dir"] for e in spec["adapters"]]
    if args.order == "reversed":
        dirs = list(reversed(dirs))
    out_dir = work_dir / f"ckpt_{args.order}"

    cmd = [
        sys.executable,
        "-m",
        "granite_switch.composer.compose_granite_switch",
        "--base-model",
        spec["base"],
        "--adapters",
        *dirs,
        "--output",
        str(out_dir),
        # Synthetic adapters ship no io.yaml (there is no I/O contract to
        # describe); a published granitelib adapter has one, and the flag is a
        # no-op then.
        "--create-ioyaml",
    ]
    print(f"composing order={args.order}:\n  " + "\n  ".join(dirs))
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        print(f"FATAL: compose CLI failed (exit {result.returncode})")
        return 1

    config = json.loads((out_dir / "config.json").read_text())
    print(
        f"  composed: num_adapters={config['num_adapters']} "
        f"names={config['adapter_names']} ranks={config['adapter_ranks']} "
        f"control_ids={config['adapter_token_ids']} "
        f"substitutes={config['adapter_substitute_token_ids']} "
        f"dual_stream={config.get('dual_stream')} "
        f"cross_stream_rank={config.get('cross_stream_rank')}"
    )
    expected = [e["name"] for e in spec["adapters"]]
    if args.order == "reversed":
        expected = list(reversed(expected))
    if list(config["adapter_names"]) != expected:
        print(f"FATAL: slot order is {config['adapter_names']}, expected {expected}")
        return 1
    return 0


# ── cases mode ────────────────────────────────────────────────────────


def cmd_cases(args):
    """Render one prompt per adapter (plus the no-adapter prompt) with the
    composed chat template, and assert the substitute-placement invariant."""
    from transformers import AutoTokenizer

    work_dir = Path(args.work_dir)
    ckpt = work_dir / f"ckpt_{args.order}"
    config = json.loads((ckpt / "config.json").read_text())
    tok = AutoTokenizer.from_pretrained(str(ckpt))
    names = list(config["adapter_names"])
    control_ids = list(config["adapter_token_ids"])
    control_by_name = dict(zip(names, control_ids))

    messages, documents, template_kwargs = chat_inputs(tok)

    def render(**kw):
        text = tok.apply_chat_template(
            messages,
            documents=documents,
            add_generation_prompt=True,
            tokenize=False,
            **template_kwargs,
            **kw,
        )
        return list(tok(text, add_special_tokens=False).input_ids)

    base_ids = render()
    control_set = set(control_ids)
    assert not (control_set & set(base_ids)), (
        "the no-adapter render already contains a control token"
    )

    cases = {"base": {"ids": base_ids, "control_pos": -1}}
    for name in names:
        ids = render(adapter_name=name)
        if len(ids) != len(base_ids):
            raise SystemExit(
                f"FATAL: adapter '{name}' render is {len(ids)} tokens vs base "
                f"{len(base_ids)} — expected SUBSTITUTE placement (equal length). "
                "Position-aligned comparison is not valid otherwise."
            )
        diffs = [i for i, (a, b) in enumerate(zip(ids, base_ids)) if a != b]
        if len(diffs) != 1 or ids[diffs[0]] != control_by_name[name]:
            raise SystemExit(
                f"FATAL: adapter '{name}' render differs from base at "
                f"{len(diffs)} positions {diffs[:5]}; expected exactly one, "
                f"holding control id {control_by_name[name]}"
            )
        cases[name] = {"ids": ids, "control_pos": diffs[0]}
        print(
            f"  {name:<26} control id {control_by_name[name]} at position "
            f"{diffs[0]}/{len(ids)}"
        )

    out = work_dir / f"cases_{args.order}.json"
    out.write_text(
        json.dumps(
            {
                "order": args.order,
                "adapter_names": names,
                "control_ids": control_ids,
                "prompt_len": len(base_ids),
                "cases": cases,
            }
        )
    )
    print(f"  wrote {len(cases)} cases ({len(base_ids)} prompt tokens) to {out}")
    return 0


# ── run mode ──────────────────────────────────────────────────────────


def _gpu_mem_util() -> float:
    """A gpu_memory_utilization that fits in the memory actually free right now.

    vLLM measures the fraction against TOTAL device memory and refuses to start if
    that exceeds what is free, so a fixed value only works on an idle card. Take
    90% of the free fraction instead, capped at the configured ceiling.
    """
    import torch

    if not torch.cuda.is_available():
        return GPU_MEM_UTIL
    free, total = torch.cuda.mem_get_info()
    fits = 0.9 * free / total
    util = min(GPU_MEM_UTIL, fits)
    print(
        f"  GPU memory: {free / 2**30:.1f}/{total / 2**30:.1f} GiB free -> "
        f"gpu_memory_utilization={util:.3f} (ceiling {GPU_MEM_UTIL:g})"
    )
    if util < GPU_MEM_UTIL_FLOOR:
        raise SystemExit(
            f"FATAL: only {free / 2**30:.1f} GiB free on this device, which leaves "
            f"gpu_memory_utilization={util:.3f} below the {GPU_MEM_UTIL_FLOOR:g} "
            "floor. Another process is holding the card; this is not a test failure."
        )
    return util


def _dists(prompt_logprobs):
    return [
        None if d is None else {str(int(t)): float(v.logprob) for t, v in d.items()}
        for d in (prompt_logprobs or [])
    ]


def cmd_run(args):
    """Score every case solo, then all cases together in one batch."""
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    from granite_switch.vllm import register as register_granite_switch

    register_granite_switch()

    work_dir = Path(args.work_dir)
    spec = json.loads(Path(args.cases).read_text())
    cases = spec["cases"]
    if args.only:
        cases = {k: v for k, v in cases.items() if k in set(args.only)}
    names = list(cases)

    eager = args.eager == "true"
    print(f"Loading {args.model} in vLLM (enforce_eager={eager})...")
    llm = LLM(
        model=args.model,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        enforce_eager=eager,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        max_model_len=MAX_MODEL_LEN,
        max_logprobs=TOPK,
        gpu_memory_utilization=_gpu_mem_util(),
    )
    score_sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=TOPK)

    # The continuation is the BASE case's greedy output under the first model to
    # run, reused by every case and every later run so all scored sequences stay
    # position-aligned. Scoring prompt+continuation is what gives the aLoRA and SR
    # cases enough positions AFTER their control token to compare.
    cont_path = work_dir / "cont.json"
    if cont_path.exists():
        cont_ids = json.loads(cont_path.read_text())["cont_ids"]
    else:
        g = llm.generate(
            TokensPrompt(prompt_token_ids=list(cases["base"]["ids"])),
            SamplingParams(temperature=0.0, max_tokens=GEN_TOKENS, ignore_eos=True),
        )
        cont_ids = list(g[0].outputs[0].token_ids)
        # Keep the continuation inside the BASE model's vocabulary: the upstream
        # run has to score the same tokens, and a composed checkpoint's control
        # rows sit above it. Truncate at the first token it could not embed.
        # Control-token rows are appended above the base vocabulary, so the
        # lowest control id IS the base model's vocab size.
        limit = min(spec["control_ids"])
        cut = next((i for i, t in enumerate(cont_ids) if t >= limit), len(cont_ids))
        if cut != len(cont_ids):
            print(
                f"  truncating continuation at {cut}: token {cont_ids[cut]} >= {limit}"
            )
        cont_ids = cont_ids[:cut]
        cont_path.write_text(json.dumps({"cont_ids": cont_ids}))
    print(f"  continuation: {len(cont_ids)} tokens {cont_ids[:8]}")

    def full(name):
        return list(cases[name]["ids"]) + list(cont_ids)

    solo = {}
    for name in names:
        out = llm.generate(TokensPrompt(prompt_token_ids=full(name)), score_sp)
        solo[name] = _dists(out[0].prompt_logprobs)
        print(f"  solo {name}: {sum(d is not None for d in solo[name])} positions")

    batched_out = llm.generate(
        [TokensPrompt(prompt_token_ids=full(n)) for n in names], score_sp
    )
    assert len(batched_out) == len(names), (
        f"batched output count {len(batched_out)} != {len(names)}"
    )
    batched = {n: _dists(o.prompt_logprobs) for n, o in zip(names, batched_out)}
    print(f"  batched: {len(batched)} cases in one submission")

    (work_dir / f"{args.tag}.json").write_text(
        json.dumps(
            {
                "solo": solo,
                "batched": batched,
                "eager": eager,
                "prompt_len": spec["prompt_len"],
                "control_pos": {n: cases[n]["control_pos"] for n in names},
            }
        )
    )
    del llm
    return 0


# ── compare mode ──────────────────────────────────────────────────────


class _Row(NamedTuple):
    """Agreement statistics for one comparison at one k."""

    k: int
    jacc: float  # mean 1 - Jaccard(top-k)
    mean: float  # mean JSD (bits)
    p99: float  # 99th-percentile JSD
    worst: float  # max JSD
    worst_pos: int  # position of the max
    n: int  # positions compared
    over: float  # fraction of positions above `cut`
    cut: float  # the outlier threshold `over` was counted against


def _agreement(R, C, from_pos=0, outlier=None):
    """Per swept k, one :class:`_Row` of agreement statistics.

    ``p99`` and ``over`` exist because a single position is the wrong thing to
    gate on: near-tie decisions flip under any perturbation, so ``max`` is
    dominated by one outlier, while a systematic error moves a large FRACTION of
    positions. All of them are printed for every comparison so the gates can be
    recalibrated from a real run rather than guessed.
    """
    idx = [i for i in range(min(len(R), len(C))) if i >= from_pos and R[i] and C[i]]
    if not idx:
        return None
    cut = PASS_OUTLIER_JSD if outlier is None else outlier
    rows = []
    for k in K_SWEEP:
        jds, jss = [], []
        for i in idx:
            ids = list(set(topk_ids(R[i], k)) | set(topk_ids(C[i], k)))
            jds.append(1.0 - jaccard(topk_ids(R[i], k), topk_ids(C[i], k)))
            jss.append(jsd_bits(R[i], C[i], ids))
        order = sorted(range(len(jss)), key=lambda j: jss[j])
        p99_at = order[min(len(order) - 1, int(0.99 * (len(order) - 1)))]
        am = order[-1]
        rows.append(
            _Row(
                k=k,
                jacc=sum(jds) / len(jds),
                mean=sum(jss) / len(jss),
                p99=jss[p99_at],
                worst=jss[am],
                worst_pos=idx[am],
                n=len(idx),
                over=sum(1 for v in jss if v > cut) / len(jss),
                cut=cut,
            )
        )
    return rows


def _show(label, rows):
    for r in rows:
        print(
            f"    {label:<26}k={r.k:<4}1-Jacc={r.jacc:<10.6f}mean={r.mean:<11.6f}"
            f"p99={r.p99:<11.6f}max={r.worst:<11.6f}@{r.worst_pos:<5}"
            f"over{r.cut:<9.4g}={r.over:<9.4f}n={r.n}"
        )


def _load(work_dir, tag):
    return json.loads((Path(work_dir) / f"{tag}.json").read_text())


def _adapter_effects(solo, control_pos):
    """``{case: (mean JSD vs base, positions, from_pos)}`` at the widest swept k.

    This is how far a real adapter moves the distribution, measured in the same
    run: what ``passthrough`` is sized against and what ``distinct`` gates.
    """
    effects = {}
    for name in solo:
        if name == "base":
            continue
        start = max(control_pos.get(name, -1), control_pos.get("base", -1)) + 1
        rows = _agreement(solo[name], solo["base"], from_pos=start)
        effects[name] = (rows[-1].mean, rows[-1].n, start) if rows else (0.0, 0, start)
    return effects


def _modes(work_dir):
    """``[(label, tag suffix)]`` for every execution mode captured on disk.

    Eager is always present; ``*_compiled`` appears when the fixture also ran the
    checkpoints with ``--eager false``. Gating both means a routing bug that only
    shows up under torch.compile / CUDA graphs cannot hide behind the eager run.
    """
    found = [("eager", "")]
    if (Path(work_dir) / "forward_compiled.json").exists():
        found.append(("compiled", "_compiled"))
    return found


def _batch_mode(work_dir, mode, suffix):
    """Solo-vs-batched for one execution mode; returns its failures."""
    failures = []
    fwd = _load(work_dir, f"forward{suffix}")
    control = _agreement(fwd["solo"]["base"], fwd["batched"]["base"])
    if control is None:
        return [f"{mode}: the base case has no comparable positions"]
    effects = _adapter_effects(fwd["solo"], fwd["control_pos"])
    smallest = min((v[0] for v in effects.values()), default=0.0)
    # Each adapter is allowed a fraction of ITS OWN effect, not of the smallest
    # across the set. The question the gate asks -- did batching move this case
    # materially -- is naturally scaled by how much this adapter changes the output
    # at all, and using the minimum penalised the strong adapters: on
    # granite-4.2-3b, syn-strong moves the distribution 0.508 bits and its
    # solo-vs-batched p99 of 0.0085 was measured against 0.1 x syn-weak's 0.039.
    floors = {
        name: BATCH_EFFECT_FRACTION * effect
        for name, (effect, _n, _s) in effects.items()
    }
    floor = BATCH_EFFECT_FRACTION * smallest
    print(
        f"  solo vs mixed-batch submission, same checkpoint [{mode}].\n"
        "  CONTROL = the base case, where no adapter is active: whatever it\n"
        "  diverges by is vLLM's own batch non-invariance, so an adapter case\n"
        f"  may not exceed {BATCH_REL_FACTOR:g}x it on the mean or "
        f"{BATCH_P99_REL_FACTOR:g}x on p99 (k >= {BATCH_MIN_GATED_K} only), nor "
        f"{BATCH_EFFECT_FRACTION:g}x its own effect (smallest {smallest:.6f} "
        f"-> {floor:.6f}), whichever is larger."
    )
    _show(f"base (CONTROL) [{mode}]", control)
    by_k = {r.k: r for r in control}
    worst_control = max(r.mean for r in control)
    if worst_control > BATCH_CONTROL_CEILING:
        failures.append(
            f"{mode}: the no-adapter control itself diverges by "
            f"{worst_control:.6f} > {BATCH_CONTROL_CEILING} — the model is "
            "unstable under batching, so this check cannot attribute anything "
            "to adapter routing"
        )
    if all(r.worst == 0.0 for r in control):
        print(
            f"  NOTE [{mode}]: the control shows no batching effect at all, so "
            "the scheduler did not perturb this model's computation. The check "
            "is satisfied but weak here; what still holds unconditionally is "
            "intra-sequence mixing — an aLoRA or SR prompt carries base tokens "
            "and adapter tokens through the same forward pass."
        )
    for name in fwd["solo"]:
        if name == "base":
            continue
        got = _agreement(fwd["solo"][name], fwd["batched"][name])
        if got is None:
            failures.append(f"{mode}/{name}: no comparable positions")
            continue
        _show(f"{name} [{mode}]", got)
        for r in got:
            if r.k < BATCH_MIN_GATED_K:
                continue  # reported above, not gated -- see BATCH_MIN_GATED_K
            c = by_k[r.k]
            own = floors.get(name, floor)
            mean_budget = max(own, c.mean * BATCH_REL_FACTOR)
            p99_budget = max(own, c.p99 * BATCH_P99_REL_FACTOR)
            if r.mean > mean_budget:
                failures.append(
                    f"{mode}/{name} k={r.k}: mean JSD={r.mean:.6f} > "
                    f"{mean_budget:.6f} (max of {BATCH_REL_FACTOR:g}x the base "
                    f"control's {c.mean:.6f} and the {own:.6f} own-effect floor)"
                )
            if r.p99 > p99_budget:
                failures.append(
                    f"{mode}/{name} k={r.k}: p99 JSD={r.p99:.6f} > "
                    f"{p99_budget:.6f} (max of {BATCH_P99_REL_FACTOR:g}x the "
                    f"base control's {c.p99:.6f} and the {own:.6f} own-effect "
                    f"floor)"
                )
    return failures


def cmd_compare(args):
    check, work_dir = args.check, args.work_dir
    print(f"\nMULTI-ADAPTER [{check}] {args.label}")
    failures = []

    if check == "passthrough":
        ref, fwd = _load(work_dir, "upstream"), _load(work_dir, "forward")
        effects = _adapter_effects(fwd["solo"], fwd["control_pos"])
        smallest = min((v[0] for v in effects.values()), default=0.0)
        which = min(effects, key=lambda k: effects[k][0]) if effects else "-"
        budget = max(PASS_EFFECT_FRACTION * smallest, PASS_FLOOR_MEAN)
        outlier = PASS_OUTLIER_MULT * budget
        got = _agreement(
            ref["solo"]["base"], fwd["solo"]["base"], outlier=outlier or None
        )
        if got is None:
            failures.append("base vs upstream: no comparable positions")
        else:
            print(
                f"  no control token vs upstream base, {got[0].n} positions.\n"
                f"  smallest real adapter effect in this run: {smallest:.6f} bits "
                f"({which}); the no-delta budget is the larger of "
                f"{PASS_EFFECT_FRACTION:g} x that and the {PASS_FLOOR_MEAN:g} "
                f"fused-vs-native floor = {budget:.6f}, and a position "
                f"counts as an outlier at {outlier:.6f}"
            )
            _show("base vs upstream", got)
            if smallest <= 0:
                failures.append(
                    "no adapter effect could be measured, so there is nothing to "
                    "size the no-delta budget against (see the distinct check)"
                )
            for r in got:
                if r.mean > budget:
                    failures.append(
                        f"k={r.k}: mean JSD={r.mean:.6f} > {budget:.6f} "
                        f"({PASS_EFFECT_FRACTION:g}x the smallest adapter effect)"
                    )
                if r.over > PASS_OUTLIER_FRAC:
                    failures.append(
                        f"k={r.k}: {r.over:.1%} of positions exceed "
                        f"{r.cut:.6f} JSD (> {PASS_OUTLIER_FRAC:.1%}) — an "
                        "adapter-sized difference at more than a stray near-tie "
                        "position"
                    )
                if r.jacc > PASS_JACC:
                    failures.append(
                        f"k={r.k}: mean(1-Jaccard)={r.jacc:.4f} > {PASS_JACC}"
                    )

    elif check == "slots":
        # The two checkpoints render their own prompts, so verify the comparison
        # is position-aligned before trusting it: the no-adapter prompt must be
        # identical, and each adapter's control token must sit at the same index
        # (its technology decides that, and the technology did not change).
        f_cases = json.loads((Path(work_dir) / "cases_forward.json").read_text())
        r_cases = json.loads((Path(work_dir) / "cases_reversed.json").read_text())
        if f_cases["cases"]["base"]["ids"] != r_cases["cases"]["base"]["ids"]:
            failures.append(
                "the two checkpoints render different no-adapter prompts — "
                "position-aligned comparison is not valid"
            )
        for name in f_cases["adapter_names"]:
            fp = f_cases["cases"][name]["control_pos"]
            rp = r_cases["cases"][name]["control_pos"]
            if fp != rp:
                failures.append(
                    f"{name}: control token at position {fp} in forward but {rp} "
                    "in reversed — prompts are not aligned"
                )
        for mode, suffix in _modes(work_dir):
            fwd = _load(work_dir, f"forward{suffix}")
            rev = _load(work_dir, f"reversed{suffix}")
            # The compiled arm is gated differently, and NOT because slot order is
            # allowed to matter more there -- see SLOTS_COMPILED_* above. vLLM 0.20
            # does not reproduce a compiled run across processes, and this gate
            # captures its two sides in two processes, so on that version the strict
            # gate is unsatisfiable by correct code. Eager stays near-exact, which is
            # where the slot-independence claim is actually established.
            compiled = mode != "eager"
            mean_gate = SLOTS_COMPILED_MEAN_JSD if compiled else SLOTS_MEAN_JSD
            jacc_gate = SLOTS_COMPILED_JACC if compiled else SLOTS_JACC
            print(f"  same adapter, opposite slot order [{mode}]")
            print(
                f"    gates: mean {mean_gate:g}, 1-Jaccard {jacc_gate:g}, "
                + (
                    f"positions over {PASS_OUTLIER_JSD:g} JSD "
                    f"<= {SLOTS_COMPILED_OUTLIER_FRAC:.1%}"
                    if compiled
                    else f"max {SLOTS_MAX_JSD:g}"
                )
            )
            for name in fwd["solo"]:
                got = _agreement(fwd["solo"][name], rev["solo"][name])
                if got is None:
                    failures.append(f"{mode}/{name}: no comparable positions")
                    continue
                _show(f"{name} [{mode}]", got)
                for r in got:
                    if r.mean > mean_gate:
                        failures.append(
                            f"{mode}/{name} k={r.k}: mean JSD={r.mean:.8f} > "
                            f"{mean_gate}"
                        )
                    if r.jacc > jacc_gate:
                        failures.append(
                            f"{mode}/{name} k={r.k}: mean(1-Jaccard)={r.jacc:.6f} > "
                            f"{jacc_gate}"
                        )
                    if compiled:
                        # A fraction, not a ceiling: one flipped near-tie is not a
                        # routing error, a broad shift is.
                        if r.over > SLOTS_COMPILED_OUTLIER_FRAC:
                            failures.append(
                                f"{mode}/{name} k={r.k}: {r.over:.1%} of positions "
                                f"exceed {r.cut:.6f} JSD "
                                f"(> {SLOTS_COMPILED_OUTLIER_FRAC:.1%}) — an "
                                "adapter-sized difference at more than a stray "
                                "near-tie position"
                            )
                    elif r.worst > SLOTS_MAX_JSD:
                        failures.append(
                            f"{mode}/{name} k={r.k}: max JSD={r.worst:.8f} @pos "
                            f"{r.worst_pos} > {SLOTS_MAX_JSD}"
                        )

    elif check == "batch":
        for mode, suffix in _modes(work_dir):
            failures += _batch_mode(work_dir, mode, suffix)

    elif check == "distinct":
        fwd = _load(work_dir, "forward")
        solo, cpos = fwd["solo"], fwd["control_pos"]
        adapters = [n for n in solo if n != "base"]
        print("  adapters differ from base and from each other")
        pairs = [(a, "base") for a in adapters]
        pairs += [(a, b) for i, a in enumerate(adapters) for b in adapters[i + 1 :]]
        for a, b in pairs:
            # Everything before the later control token is identical by
            # construction, so comparing there would only dilute the signal.
            start = max(cpos.get(a, -1), cpos.get(b, -1)) + 1
            got = _agreement(solo[a], solo[b], from_pos=start)
            if got is None:
                failures.append(f"{a} vs {b}: no comparable positions >= {start}")
                continue
            r = got[-1]  # widest k in the sweep
            print(
                f"    {a} vs {b}: mean JSD={r.mean:.6f} over {r.n} positions "
                f"(from {start})"
            )
            if r.mean < DISTINCT_MIN_JSD:
                failures.append(
                    f"{a} vs {b}: mean JSD={r.mean:.6f} < {DISTINCT_MIN_JSD} — "
                    "indistinguishable, so the equality checks are vacuous"
                )
    else:
        print(f"unknown check {check!r}")
        return 1

    if failures:
        print(f"\nFAIL [{check}] {args.label}:\n  " + "\n  ".join(failures))
        return 1
    print(f"\nPASS [{check}] {args.label}")
    return 0


# ── CLI ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    p_prep = sub.add_parser("prepare")
    p_prep.add_argument("--flavor", required=True, choices=["lora", "sr"])
    p_prep.add_argument("--base", required=True)
    p_prep.add_argument("--work-dir", required=True)

    p_build = sub.add_parser("build")
    p_build.add_argument("--work-dir", required=True)
    p_build.add_argument("--order", required=True, choices=["forward", "reversed"])

    p_cases = sub.add_parser("cases")
    p_cases.add_argument("--work-dir", required=True)
    p_cases.add_argument("--order", required=True, choices=["forward", "reversed"])

    p_run = sub.add_parser("run")
    p_run.add_argument("--model", required=True)
    p_run.add_argument("--work-dir", required=True)
    p_run.add_argument("--tag", required=True)
    p_run.add_argument("--cases", required=True)
    p_run.add_argument("--only", nargs="*", default=None)
    p_run.add_argument(
        "--eager",
        default="true",
        choices=["true", "false"],
        help="enforce_eager. false exercises torch.compile and CUDA graphs.",
    )

    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("--work-dir", required=True)
    p_cmp.add_argument(
        "--check",
        required=True,
        choices=["passthrough", "slots", "batch", "distinct"],
    )
    p_cmp.add_argument("--label", required=True)

    args = parser.parse_args()
    return {
        "prepare": cmd_prepare,
        "build": cmd_build,
        "cases": cmd_cases,
        "run": cmd_run,
        "compare": cmd_compare,
    }[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
