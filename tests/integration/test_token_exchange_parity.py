# SPDX-License-Identifier: Apache-2.0
"""Parity eval: legacy KV hiding vs. token exchange.

Measures four metrics per token position, teacher-forced, across a list of
prompts:

1. KL(p_old || p_new)         — full-distribution divergence
2. Top-1 agreement             — headline sanity metric
3. Nucleus (top-p=0.9) Jaccard — do sampling sets agree?
4. Mass under old nucleus      — does new model put probability on tokens old
                                  model considered plausible?

Two modes:

**Synthetic mode (default, CPU-friendly):** builds two HF models with
identical base weights, one in legacy KV-hiding mode and one in token-
exchange mode. Measures only the effect of control-token handling on
logits — *not* trained-adapter behavior. Useful as a plumbing sanity
check and a regression guard.

**Real-model mode (GPU, opt-in):** set
``GRANITE_SWITCH_PARITY_MODELS='{"old":"/path","new":"/path"}'`` (JSON with
two paths) and pytest will load actual composed checkpoints. This is the
pre-merge gate described in docs/KV_CACHE_OVERHEAD_REMOVAL.md §4.

Run directly::

    python -m tests.integration.test_token_exchange_parity

Run as test::

    pytest tests/integration/test_token_exchange_parity.py -v -s --tb=short
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pytest
import torch
import torch.nn.functional as F

from granite_switch.config import GraniteSwitchConfig
from granite_switch.hf import GraniteSwitchForCausalLM


# ────────────────────────────────────────────────────────────────────
# Metric primitives
# ────────────────────────────────────────────────────────────────────


def _kl_from_logits(logits_p: torch.Tensor, logits_q: torch.Tensor) -> float:
    """KL(p || q) in nats, computed from logits to avoid softmax underflow.

    Equivalent to ``sum_i p_i * (log p_i - log q_i)`` but evaluated via
    log_softmax so that very small tail probabilities don't underflow to
    zero before the log. logits_{p,q}: 1-D [vocab].
    """
    log_p = F.log_softmax(logits_p, dim=-1)
    log_q = F.log_softmax(logits_q, dim=-1)
    p = log_p.exp()
    return float((p * (log_p - log_q)).sum())


def _nucleus_indices(p: torch.Tensor, top_p: float) -> torch.Tensor:
    """Smallest descending-sorted prefix whose cumulative sum >= top_p.

    The nucleus always contains at least one token (the argmax). k is the
    index (1-based count) of the first element where cumsum >= top_p.
    """
    sorted_p, sorted_idx = torch.sort(p, descending=True)
    cumsum = torch.cumsum(sorted_p, dim=0)
    # Smallest index where cumsum >= top_p. If cumsum never reaches top_p
    # (floating-point edge), keep everything.
    ge = cumsum >= top_p
    if ge.any():
        k = int(torch.argmax(ge.int()).item()) + 1  # +1: argmax is 0-indexed, we want count
    else:
        k = sorted_idx.numel()
    k = max(1, min(k, sorted_idx.numel()))
    return sorted_idx[:k]


def _jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    a_set = set(a.tolist())
    b_set = set(b.tolist())
    if not a_set and not b_set:
        return 1.0
    return len(a_set & b_set) / len(a_set | b_set)


# ────────────────────────────────────────────────────────────────────
# Core eval
# ────────────────────────────────────────────────────────────────────


@dataclass
class PositionResult:
    kl: float
    top1_agree: bool
    nucleus_jaccard: float
    mass_under_old_nucleus: float
    old_nucleus_size: int
    new_nucleus_size: int
    # Partition flag: True if this position is at or after a control token
    # in the causal past (adapter-activated); False if it's in the base path.
    adapter_active: bool = False


@dataclass
class AggregateResult:
    n_positions: int
    kl_mean: float
    kl_median: float
    kl_p95: float
    kl_max: float
    top1_agree_rate: float
    nucleus_jaccard_mean: float
    nucleus_jaccard_exact_match_rate: float
    mass_under_old_nucleus_mean: float
    mass_under_old_nucleus_p05: float
    # Fraction of positions where the new model places less than 0.80 of its
    # probability mass on the old model's nucleus — the actionable "how bad"
    # signal for real-model runs.
    frac_mass_under_80: float
    # Nucleus size distribution: useful for judging whether top-1 agreement
    # is a meaningful metric (confident models: median 1-5; noise: ~vocab/2).
    old_nucleus_size_median: float
    new_nucleus_size_median: float
    old_nucleus_size_p05: float
    old_nucleus_size_p95: float

    def render(self, heading: str = "") -> str:
        header = f"── {heading} ──" if heading else ""
        trusted = "(trusted)" if self.old_nucleus_size_median < 50 else "(noisy: wide nucleus)"
        lines = [
            header,
            f"n_positions                     = {self.n_positions}",
            "",
            "KL(p_old || p_new) per position:",
            f"  mean                          = {self.kl_mean:.6f}",
            f"  median                        = {self.kl_median:.6f}",
            f"  p95                           = {self.kl_p95:.6f}",
            f"  max                           = {self.kl_max:.6f}",
            "",
            f"Top-1 agreement rate            = {self.top1_agree_rate:.4f} {trusted}",
            "",
            "Nucleus (top-p=0.9):",
            f"  size — old p05/med/p95        = {self.old_nucleus_size_p05:g} / {self.old_nucleus_size_median:g} / {self.old_nucleus_size_p95:g}",
            f"  size — new median             = {self.new_nucleus_size_median:g}",
            f"  Jaccard mean                  = {self.nucleus_jaccard_mean:.4f}",
            f"  exact-match rate              = {self.nucleus_jaccard_exact_match_rate:.4f}",
            "",
            "Mass under old nucleus (new model):",
            f"  mean                          = {self.mass_under_old_nucleus_mean:.4f}",
            f"  p05 (worst 5% of positions)   = {self.mass_under_old_nucleus_p05:.4f}",
            f"  frac positions < 0.80         = {self.frac_mass_under_80:.4f}",
        ]
        return "\n".join(line for line in lines if line is not None)

    def as_dict(self) -> Dict[str, float]:
        return {
            "n_positions": self.n_positions,
            "kl_mean": self.kl_mean,
            "kl_median": self.kl_median,
            "kl_p95": self.kl_p95,
            "kl_max": self.kl_max,
            "top1_agree_rate": self.top1_agree_rate,
            "nucleus_jaccard_mean": self.nucleus_jaccard_mean,
            "nucleus_jaccard_exact_match_rate": self.nucleus_jaccard_exact_match_rate,
            "mass_under_old_nucleus_mean": self.mass_under_old_nucleus_mean,
            "mass_under_old_nucleus_p05": self.mass_under_old_nucleus_p05,
            "frac_mass_under_80": self.frac_mass_under_80,
            "old_nucleus_size_median": self.old_nucleus_size_median,
            "new_nucleus_size_median": self.new_nucleus_size_median,
            "old_nucleus_size_p05": self.old_nucleus_size_p05,
            "old_nucleus_size_p95": self.old_nucleus_size_p95,
        }


def _adapter_active_mask(input_ids: torch.Tensor, adapter_token_ids: List[int]) -> torch.Tensor:
    """[seq_len] bool: True at position s if any control token appears at
    positions <= s. Token at position s itself counts — the swap happens
    before that position's hidden state enters the decoder."""
    ctrl_set = set(adapter_token_ids)
    is_ctrl = torch.tensor(
        [int(t.item()) in ctrl_set for t in input_ids], dtype=torch.bool
    )
    # Cumulative OR along the sequence.
    return torch.cummax(is_ctrl.int(), dim=0).values.bool()


def _per_position_metrics(
    logits_old: torch.Tensor,
    logits_new: torch.Tensor,
    top_p: float,
    adapter_active: Optional[torch.Tensor] = None,
) -> List[PositionResult]:
    """logits_{old,new}: [seq_len, vocab_size]. Returns one result per position."""
    assert logits_old.shape == logits_new.shape
    results: List[PositionResult] = []
    # Promote to float32 for metric stability.
    logits_old = logits_old.to(torch.float32)
    logits_new = logits_new.to(torch.float32)
    p_old_all = F.softmax(logits_old, dim=-1)
    p_new_all = F.softmax(logits_new, dim=-1)
    for s in range(logits_old.shape[0]):
        p_old = p_old_all[s]
        p_new = p_new_all[s]
        nuc_old = _nucleus_indices(p_old, top_p)
        nuc_new = _nucleus_indices(p_new, top_p)
        results.append(
            PositionResult(
                kl=_kl_from_logits(logits_old[s], logits_new[s]),
                top1_agree=bool(p_old.argmax() == p_new.argmax()),
                nucleus_jaccard=_jaccard(nuc_old, nuc_new),
                mass_under_old_nucleus=float(p_new[nuc_old].sum()),
                old_nucleus_size=int(nuc_old.numel()),
                new_nucleus_size=int(nuc_new.numel()),
                adapter_active=bool(adapter_active[s]) if adapter_active is not None else False,
            )
        )
    return results


def _aggregate(results: List[PositionResult]) -> AggregateResult:
    if not results:
        raise ValueError("No positions measured")
    kls = sorted(r.kl for r in results)
    jaccards = [r.nucleus_jaccard for r in results]
    mass = sorted(r.mass_under_old_nucleus for r in results)
    old_sizes = sorted(r.old_nucleus_size for r in results)
    n = len(results)
    p05_idx = max(0, int(n * 0.05) - 1)
    p95_idx = min(n - 1, int(n * 0.95))
    return AggregateResult(
        n_positions=n,
        kl_mean=statistics.mean(kls),
        kl_median=statistics.median(kls),
        kl_p95=kls[p95_idx],
        kl_max=kls[-1],
        top1_agree_rate=sum(r.top1_agree for r in results) / n,
        nucleus_jaccard_mean=statistics.mean(jaccards),
        nucleus_jaccard_exact_match_rate=sum(1 for j in jaccards if j == 1.0) / n,
        mass_under_old_nucleus_mean=statistics.mean(mass),
        mass_under_old_nucleus_p05=mass[p05_idx],
        frac_mass_under_80=sum(1 for m in mass if m < 0.80) / n,
        old_nucleus_size_median=statistics.median(r.old_nucleus_size for r in results),
        new_nucleus_size_median=statistics.median(r.new_nucleus_size for r in results),
        old_nucleus_size_p05=old_sizes[p05_idx],
        old_nucleus_size_p95=old_sizes[p95_idx],
    )


# ────────────────────────────────────────────────────────────────────
# Synthetic model builder (CPU-friendly, weight-sharing pair)
# ────────────────────────────────────────────────────────────────────


_SYNTHETIC_BASE_KWARGS = dict(
    vocab_size=512,
    hidden_size=64,
    num_attention_heads=4,
    num_key_value_heads=2,
    num_hidden_layers=4,
    intermediate_size=128,
    shared_intermediate_size=128,
    max_position_embeddings=128,
    mamba_n_heads=1,
    mamba_expand=1,
    torch_dtype=torch.float32,
)


def _build_synthetic_pair(
    num_adapters: int = 2,
    seed: int = 0,
) -> Tuple[GraniteSwitchForCausalLM, GraniteSwitchForCausalLM]:
    """Build two models with identical base weights: old=hiding, new=exchange.

    Any logit difference between them is therefore purely from control-token
    handling, not from weight initialization.
    """
    adapter_token_ids = [100, 101][:num_adapters]
    substitute_token_ids = [5, 7][:num_adapters]

    torch.manual_seed(seed)
    old_config = GraniteSwitchConfig(
        **_SYNTHETIC_BASE_KWARGS,
        num_adapters=num_adapters,
        adapter_ranks=[4] * num_adapters,
        max_lora_rank=4,
        adapter_token_ids=adapter_token_ids,
        adapter_names=[f"a{i}" for i in range(num_adapters)],
        control_dims=32,
    )
    old_model = GraniteSwitchForCausalLM(old_config).eval()

    new_config = GraniteSwitchConfig(
        **_SYNTHETIC_BASE_KWARGS,
        num_adapters=num_adapters,
        adapter_ranks=[4] * num_adapters,
        max_lora_rank=4,
        adapter_token_ids=adapter_token_ids,
        adapter_substitute_token_ids=substitute_token_ids,
        adapter_names=[f"a{i}" for i in range(num_adapters)],
        control_dims=0,
    )
    new_model = GraniteSwitchForCausalLM(new_config).eval()

    # Share weights where the two configs have matching parameter shapes.
    # Non-shared: tensors whose shape depends on control_dims (e.g. switch
    # head_dim in the legacy path differs from the new native-head_dim path).
    old_sd = old_model.state_dict()
    new_sd = new_model.state_dict()
    transferred = 0
    skipped: List[str] = []
    for name, new_tensor in new_sd.items():
        if name in old_sd and old_sd[name].shape == new_tensor.shape:
            new_tensor.copy_(old_sd[name])
            transferred += 1
        else:
            skipped.append(name)
    assert transferred > 0, "no weights transferred; synthetic pair would be meaningless"
    return old_model, new_model


def _synthetic_prompts(
    num_adapters: int,
    adapter_token_ids: List[int],
    vocab_size: int,
) -> List[torch.Tensor]:
    """A small, deterministic set of prompt sequences.

    Mix of:
      - Prompts with no control token (base-path sanity).
      - Prompts with a control token at different positions (adapter-activated).
    """
    torch.manual_seed(42)
    prompts: List[torch.Tensor] = []
    seq_len = 24
    # Fill tokens are drawn from the vocab excluding control-token ids.
    safe_vocab = [t for t in range(1, vocab_size) if t not in adapter_token_ids]

    def _rand_seq() -> List[int]:
        return [safe_vocab[int(torch.randint(0, len(safe_vocab), (1,)))] for _ in range(seq_len)]

    # Base-path prompts (no control tokens).
    for _ in range(4):
        prompts.append(torch.tensor([_rand_seq()], dtype=torch.long))
    # Adapter-activated prompts (one control token at varying positions).
    for pos in (0, 2, 5, 10):
        for ctrl_id in adapter_token_ids:
            seq = _rand_seq()
            seq[pos] = ctrl_id
            prompts.append(torch.tensor([seq], dtype=torch.long))
    return prompts


def _demo_prompts(tokenizer, adapter_names: List[str]) -> List[torch.Tensor]:
    """Realistic parity prompts: render every demo from tutorials/scripts
    through the composed model's chat template, then tokenize.

    Each returned tensor is shape [1, seq_len]. Shape varies per prompt —
    the parity eval loops one at a time, so no padding is needed.
    """
    from tutorials.scripts.run_adapter_generation_direct import build_demo_prompts

    prompts: List[torch.Tensor] = []
    pairs = build_demo_prompts(tokenizer, available_adapters=set(adapter_names))
    for _demo_key, prompt_text in pairs:
        ids = tokenizer(prompt_text, return_tensors="pt").input_ids
        prompts.append(ids)
    return prompts


# ────────────────────────────────────────────────────────────────────
# Runner
# ────────────────────────────────────────────────────────────────────


@dataclass
class ParityReport:
    overall: AggregateResult
    pre_control: Optional[AggregateResult]   # positions before any control token
    adapter_active: Optional[AggregateResult]  # positions at / after control token

    def render(self) -> str:
        parts = [self.overall.render("overall")]
        if self.pre_control is not None:
            parts.append("")
            parts.append(self.pre_control.render("pre-control (base path)"))
        if self.adapter_active is not None:
            parts.append("")
            parts.append(self.adapter_active.render("adapter-active"))
        return "\n".join(parts)

    def as_dict(self) -> Dict:
        d = {"overall": self.overall.as_dict()}
        if self.pre_control is not None:
            d["pre_control"] = self.pre_control.as_dict()
        if self.adapter_active is not None:
            d["adapter_active"] = self.adapter_active.as_dict()
        return d


def run_parity_eval(
    old_model: GraniteSwitchForCausalLM,
    new_model: GraniteSwitchForCausalLM,
    prompts: List[torch.Tensor],
    adapter_token_ids: List[int],
    top_p: float = 0.9,
) -> ParityReport:
    all_results: List[PositionResult] = []
    for prompt in prompts:
        with torch.no_grad():
            out_old = old_model(input_ids=prompt)
            out_new = new_model(input_ids=prompt)
        logits_old = out_old.logits[0]  # [seq_len, vocab]
        logits_new = out_new.logits[0]
        mask = _adapter_active_mask(prompt[0], adapter_token_ids)
        all_results.extend(
            _per_position_metrics(logits_old, logits_new, top_p, adapter_active=mask)
        )

    overall = _aggregate(all_results)
    pre = [r for r in all_results if not r.adapter_active]
    active = [r for r in all_results if r.adapter_active]
    return ParityReport(
        overall=overall,
        pre_control=_aggregate(pre) if pre else None,
        adapter_active=_aggregate(active) if active else None,
    )


# ────────────────────────────────────────────────────────────────────
# pytest entry points
# ────────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_synthetic_parity_cpu():
    """Plumbing sanity check: legacy vs. token-exchange on a synthetic pair.

    With random weights, the two paths produce different logits (the swap IS
    the difference), but the *structure* of the comparison should hold: base
    positions (no control token) should agree perfectly; adapter-activated
    positions will differ and we report how much.
    """
    old_model, new_model = _build_synthetic_pair()
    prompts = _synthetic_prompts(
        num_adapters=2,
        adapter_token_ids=[100, 101],
        vocab_size=_SYNTHETIC_BASE_KWARGS["vocab_size"],
    )
    report = run_parity_eval(
        old_model, new_model, prompts, adapter_token_ids=[100, 101]
    )
    print("\n" + report.render())
    assert report.overall.n_positions > 0
    assert report.overall.kl_max >= 0.0
    assert 0.0 <= report.overall.top1_agree_rate <= 1.0
    # Pre-control positions MUST agree bit-for-bit (both paths process them
    # identically — no substitution, no hiding). Any disagreement here is a
    # bug in the swap gating, not a mode trade-off.
    if report.pre_control is not None:
        assert report.pre_control.kl_max < 1e-6, (
            f"Pre-control KL max {report.pre_control.kl_max} should be ~0"
        )
        assert report.pre_control.top1_agree_rate == 1.0


@pytest.mark.slow
@pytest.mark.requires_model
def test_real_model_parity():
    """Gate for real composed checkpoints. Opt-in via env var.

    Set ``GRANITE_SWITCH_PARITY_MODELS`` to a JSON object with two paths:
        '{"old": "/path/to/control_dims=32_build", "new": "/path/to/token_exchange_build"}'

    Both must be composed from the same base + adapter pair, differing only
    in --legacy-hiding. Acceptance thresholds are documented per-metric; the
    test fails if any is exceeded.
    """
    spec = os.environ.get("GRANITE_SWITCH_PARITY_MODELS")
    if spec is None:
        pytest.skip("GRANITE_SWITCH_PARITY_MODELS env var not set")
    paths = json.loads(spec)
    old_path, new_path = paths["old"], paths["new"]

    old_model = GraniteSwitchForCausalLM.from_pretrained(old_path).eval()
    new_model = GraniteSwitchForCausalLM.from_pretrained(new_path).eval()

    # Prompt set priority:
    #   1. GRANITE_SWITCH_PARITY_PROMPTS env var (JSON array of int lists).
    #   2. Rendered demo prompts from tutorials/scripts/run_adapter_generation_direct
    #      via the composed tokenizer — realistic adapter inputs.
    #   3. Synthetic fallback (only useful when demo prompts fail for some reason).
    prompts_spec = os.environ.get("GRANITE_SWITCH_PARITY_PROMPTS")
    if prompts_spec:
        prompt_lists = json.loads(prompts_spec)
        prompts = [torch.tensor([p], dtype=torch.long) for p in prompt_lists]
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(old_path)
        adapter_names = list(old_model.config.adapter_names or [])
        prompts = _demo_prompts(tokenizer, adapter_names)
        if not prompts:
            prompts = _synthetic_prompts(
                num_adapters=old_model.config.num_adapters,
                adapter_token_ids=list(old_model.config.adapter_token_ids),
                vocab_size=old_model.config.vocab_size,
            )

    report = run_parity_eval(
        old_model,
        new_model,
        prompts,
        adapter_token_ids=list(old_model.config.adapter_token_ids),
    )
    print("\n" + report.render())

    # Acceptance thresholds on the adapter-active partition (the comparison
    # we actually care about). See docs/KV_CACHE_OVERHEAD_REMOVAL.md §4.
    # Initial guesses — calibrate against a control_dims=32 vs control_dims=1
    # baseline run first and tighten to 2-3x the observed noise floor.
    active = report.adapter_active if report.adapter_active else report.overall
    assert active.top1_agree_rate >= 0.95, (
        f"top-1 agreement {active.top1_agree_rate:.4f} below 0.95 threshold"
    )
    assert active.kl_mean <= 0.02, (
        f"mean KL {active.kl_mean:.5f} above 0.02 threshold"
    )
    assert active.mass_under_old_nucleus_mean >= 0.88, (
        f"mean mass under old nucleus {active.mass_under_old_nucleus_mean:.4f} "
        f"below 0.88 threshold"
    )


# ────────────────────────────────────────────────────────────────────
# CLI entry point
# ────────────────────────────────────────────────────────────────────


def _cli():
    p = argparse.ArgumentParser(description="Token-exchange parity eval.")
    p.add_argument("--old", type=str, default=None, help="Path to legacy-hiding model")
    p.add_argument("--new", type=str, default=None, help="Path to token-exchange model")
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--json-out", type=str, default=None, help="Optional JSON report path")
    args = p.parse_args()

    if args.old and args.new:
        print(f"Loading old model from {args.old}...")
        old_model = GraniteSwitchForCausalLM.from_pretrained(args.old).eval()
        print(f"Loading new model from {args.new}...")
        new_model = GraniteSwitchForCausalLM.from_pretrained(args.new).eval()
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.old)
        adapter_names = list(old_model.config.adapter_names or [])
        print(f"Building demo prompts for adapters: {adapter_names}")
        prompts = _demo_prompts(tokenizer, adapter_names)
        if not prompts:
            print("No demo prompts matched; falling back to synthetic.")
            prompts = _synthetic_prompts(
                num_adapters=old_model.config.num_adapters,
                adapter_token_ids=list(old_model.config.adapter_token_ids),
                vocab_size=old_model.config.vocab_size,
            )
        print(f"Collected {len(prompts)} prompts.")
    else:
        print("Running synthetic parity (no --old/--new paths given)...")
        old_model, new_model = _build_synthetic_pair()
        prompts = _synthetic_prompts(
            num_adapters=2,
            adapter_token_ids=[100, 101],
            vocab_size=_SYNTHETIC_BASE_KWARGS["vocab_size"],
        )

    if args.old and args.new:
        adapter_token_ids = list(old_model.config.adapter_token_ids)
    else:
        adapter_token_ids = [100, 101]
    report = run_parity_eval(
        old_model, new_model, prompts, adapter_token_ids=adapter_token_ids, top_p=args.top_p,
    )
    print()
    print(report.render())
    if args.json_out:
        import json as _json
        with open(args.json_out, "w") as f:
            _json.dump(report.as_dict(), f, indent=2)
        print(f"\nWrote JSON report to {args.json_out}")


if __name__ == "__main__":
    _cli()
