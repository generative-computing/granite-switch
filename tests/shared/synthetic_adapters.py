# SPDX-License-Identifier: Apache-2.0
"""Synthetic LoRA / aLoRA / Shadow-Residual adapters fitted to trained ones.

Generating ``lora_A`` and ``lora_B`` as iid Gaussians produces an adapter that is
wrong in the one way that matters. Measured on the published granitelib adapters
for granite-4.1-3b (q_proj / o_proj / gate_proj / down_proj, layers 0/10/20/30/39,
effective delta ``(alpha/r) * B @ A``):

    adapter            r   alpha  ||d||F/||W||F      stable rank   std(B)/std(A)
    citations          16  32     0.019 -> 0.044     1.30 - 2.30   0.17 - 0.28
    requirement-check  64  64     0.302 -> 0.363     3.53 - 23.9   0.67 - 0.78
    uncertainty        32  64     0.076 -> 0.114     1.13 - 3.48   0.38 - 0.59
    answerability      16  32     0.008 -> 0.017     1.09 - 2.15   0.08 - 0.36

    an iid product      16  -     (scale-free)       13.1          1.00
                        32  -                        24.6          1.00
                        64  -                        42.1          1.00

A trained delta is nearly rank-one to rank-three **whatever its nominal rank**,
while an iid product has stable rank ~r/1.5 and an almost flat spectrum. It also
perturbs the base weight by wildly different amounts per adapter — a factor of 40
between ``answerability`` and ``requirement-check`` — and its ``B`` is usually far
smaller than its ``A``, by a ratio that says how far training went.

So a profile here fixes three things a scale knob cannot: the **singular-value
spectrum** (via a power law solved for the measured stable rank), the **delta norm
relative to the base weight** at that layer, and the **A/B asymmetry**. All three
are hit exactly, and matching the stable rank reproduces the measured spectral
decay (``sv8/sv1`` 0.13 synthesized against 0.04-0.14 real for the low-rank
adapters, 0.51 against 0.31-0.78 for requirement-check) without fitting it
separately.

Deliberately NOT modelled: alignment between the delta's row space and the base
weight's top singular directions. Measured at only 1.2-1.6x the random baseline
with a range spanning 0.83-3.2x, it would cost an SVD per module per layer to
reproduce something that weak.

Two reasons to prefer these over the real adapters the tests can also use:
they need no published adapter for the base model — so any base works, including
one whose granitelib adapters do not exist — and rank, module coverage and
strength are chosen rather than inherited, which is how a test reaches a specific
kernel rank tier. What they cannot support is any claim about adapter quality;
the equivalence tests make none.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

PEFT_PREFIX = "base_model.model.model."


@dataclass(frozen=True)
class AdapterProfile:
    """Targets for one synthetic adapter, fitted to a named real one.

    ``rel_fro_first`` / ``rel_fro_last`` are ``||delta||_F / ||W||_F`` at the
    first and last layer, interpolated linearly in between (trained adapters grow
    modestly with depth). Being expressed relative to the base weight is what
    lets a profile fitted on granite-4.1-3b carry over to another base.
    """

    name: str
    rank: int
    alpha: float
    rel_fro_first: float
    rel_fro_last: float
    stable_rank: float
    b_over_a: float
    effect_scale: float = 1.0

    @property
    def peft_scale(self) -> float:
        return self.alpha / self.rank

    def rel_fro(self, layer: int, num_layers: int) -> float:
        """Target ``||delta||_F / ||W||_F`` for this layer, after ``effect_scale``.

        The un-scaled endpoints are the trained adapter's own measured norms.
        ``effect_scale`` is the empirical correction described at
        :data:`PROFILES`: a random-direction delta needs a larger norm than a
        trained one to move the output distribution by the same amount.
        """
        t = layer / max(1, num_layers - 1)
        base = self.rel_fro_first + t * (self.rel_fro_last - self.rel_fro_first)
        return base * self.effect_scale


#: Fitted to the granitelib adapters named, on granite-4.1-3b. Ranks are on the
#: SWITCH kernel's supported tiers.
#:
#: ``effect_scale`` is where weight-space fidelity stops being enough. Matching
#: the delta norm, spectrum and asymmetry does NOT reproduce a trained adapter's
#: FUNCTIONAL effect, and the gap is large. Measured under vLLM on
#: granite-4.1-3b, mean JSD of an adapter against base:
#:
#:     profile   same norm as      synthesized   the trained adapter
#:     weak      answerability          0.012                  0.329
#:     light     citations              0.042                  (n/a)
#:     medium    uncertainty            0.111                  0.310
#:     strong    requirement-check      0.596                  0.232
#:
#: At `answerability`'s norm a random-direction delta is ~28x weaker; at
#: `requirement-check`'s it is ~2.6x stronger. So the norm-to-effect map is not
#: even monotone between the two populations, and the alignment this module
#: deliberately does not model — measured at only 1.2-1.6x the random baseline —
#: is evidently doing most of the functional work, because what matters is
#: whether the TOP singular direction lands somewhere the model is sensitive to,
#: not the average overlap across the spectrum.
#:
#: The gates live in function space: `distinct` needs effects clear of its floor,
#: and `passthrough`/`batch` size their budgets off the smallest one. So the
#: scales below lift each profile's effect into the range a trained adapter
#: occupies, while the un-scaled endpoints stay visible as what was measured.
PROFILES = {
    "weak": AdapterProfile("weak", 16, 32, 0.0076, 0.0171, 1.53, 0.140, 8.0),
    "light": AdapterProfile("light", 16, 32, 0.0190, 0.0440, 1.67, 0.223, 4.0),
    "medium": AdapterProfile("medium", 32, 64, 0.0759, 0.1137, 1.82, 0.533, 2.0),
    "strong": AdapterProfile("strong", 64, 64, 0.3022, 0.3633, 11.63, 0.719, 1.0),
}

#: cross_stream has no base weight (the shunt is W-less), so its target is sized
#: against the layer's o_proj — both inject into the residual stream at hidden
#: width. No trained SR adapter is published, so unlike every other number here
#: this one is a choice, not a measurement: same order as the adapter's own
#: o_proj delta.
CROSS_STREAM_REFERENCE = "self_attn.o_proj"


def _seeded(key: str, seed: int) -> torch.Generator:
    """Generator keyed by ``(seed, key)`` — hashlib, not the salted ``hash()``."""
    digest = hashlib.blake2b(f"{seed}:{key}".encode(), digest_size=8).digest()
    return torch.Generator().manual_seed(int.from_bytes(digest, "big") % (2**63))


def spectrum(rank: int, stable_rank: float) -> torch.Tensor:
    """Power-law singular values whose stable rank is ``stable_rank``.

    Stable rank ``(sum s^2) / s_1^2`` is the scale-free measure of how many
    directions a matrix really uses. Bisect the exponent of ``s_i = i^-a`` until
    it matches; ``a`` near 0 gives the flat iid-like spectrum, large ``a`` the
    near-rank-one one a trained adapter actually has.
    """
    target = max(1.0, min(float(stable_rank), float(rank)))
    lo, hi = 0.0, 12.0
    for _ in range(80):
        mid = (lo + hi) / 2
        s = torch.arange(1, rank + 1, dtype=torch.float64) ** (-mid)
        if float((s**2).sum() / s[0] ** 2) > target:
            lo = mid
        else:
            hi = mid
    return (torch.arange(1, rank + 1, dtype=torch.float64) ** (-(lo + hi) / 2)).float()


def factorize(
    out_features: int,
    in_features: int,
    *,
    rank: int,
    target_fro: float,
    stable_rank: float,
    b_over_a: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(lora_A, lora_B)`` whose product has the requested spectrum and norm.

    Builds ``delta = U diag(s) V^T`` from random orthonormal ``U``/``V`` and the
    prescribed ``s``, then splits it as ``B = U sqrt(s)``, ``A = sqrt(s) V^T``.
    One free scalar rescales A against B without touching their product, which is
    what sets ``std(B)/std(A)`` exactly.
    """
    if rank > min(out_features, in_features):
        raise ValueError(
            f"rank {rank} exceeds min(out={out_features}, in={in_features})"
        )
    s = spectrum(rank, stable_rank)
    s = s / s.norm() * target_fro  # ||delta||_F == ||s||_2
    u, _ = torch.linalg.qr(torch.randn(out_features, rank, generator=generator))
    v, _ = torch.linalg.qr(torch.randn(in_features, rank, generator=generator))
    root = s.sqrt()
    a = root[:, None] * v.T  # [rank, in]
    b = u * root[None, :]  # [out, rank]
    ratio = float(b.std() / a.std())
    c = (ratio / b_over_a) ** 0.5
    # contiguous(): the broadcasts above leave non-contiguous views, which
    # safetensors refuses to serialize.
    return (a * c).contiguous(), (b / c).contiguous()


# ---------------------------------------------------------------------------
# Base-model weight norms
# ---------------------------------------------------------------------------


def base_weight_norms(base_path: str | Path) -> dict[str, float]:
    """``{parameter name: Frobenius norm}`` for every per-layer projection.

    Read one tensor at a time so a 3B checkpoint costs a disk pass and not 7GB of
    memory. These norms are the denominators the profiles are expressed in.
    """
    base = Path(base_path)
    shards = sorted(base.glob("model*.safetensors")) or sorted(
        base.glob("*.safetensors")
    )
    if not shards:
        raise FileNotFoundError(f"no safetensors weights under {base}")
    norms = {}
    for shard in shards:
        with safe_open(str(shard), "pt") as handle:
            for key in handle.keys():
                if ".layers." not in key or not key.endswith(".weight"):
                    continue
                tensor = handle.get_tensor(key)
                if tensor.ndim != 2:
                    continue  # layernorm scales and the like
                norms[key] = float(torch.linalg.matrix_norm(tensor.float()))
    return norms


# ---------------------------------------------------------------------------
# Module tables
# ---------------------------------------------------------------------------


def _module_table(dims):
    """``{peft module: (base suffix, out, in)}`` for this base's MLP spelling."""
    h, i, q, kv = dims.hidden, dims.intermediate, dims.q_width, dims.kv_width
    table = {
        "self_attn.q_proj": ("self_attn.q_proj", q, h),
        "self_attn.k_proj": ("self_attn.k_proj", kv, h),
        "self_attn.v_proj": ("self_attn.v_proj", kv, h),
        "self_attn.o_proj": ("self_attn.o_proj", h, q),
    }
    if dims.mlp_style == "shared":
        table |= {
            "shared_mlp.input_linear": ("shared_mlp.input_linear", 2 * i, h),
            "shared_mlp.output_linear": ("shared_mlp.output_linear", h, i),
        }
    else:
        table |= {
            "mlp.gate_proj": ("mlp.gate_proj", i, h),
            "mlp.up_proj": ("mlp.up_proj", i, h),
            "mlp.down_proj": ("mlp.down_proj", h, i),
        }
    return table


def attn_modules(with_kv: bool = True) -> list[str]:
    mods = ["self_attn.q_proj", "self_attn.o_proj"]
    if with_kv:
        mods[1:1] = ["self_attn.k_proj", "self_attn.v_proj"]
    return mods


def mlp_modules(dims) -> list[str]:
    if dims.mlp_style == "shared":
        return ["shared_mlp.input_linear", "shared_mlp.output_linear"]
    return ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def _target_modules(peft_modules: list[str]) -> list[str]:
    """PEFT ``target_modules`` entries: the leaf name of each module path."""
    return sorted({m.rsplit(".", 1)[-1] for m in peft_modules})


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def synthesize_adapter(
    path: Path,
    dims,
    norms: dict[str, float],
    profile: AdapterProfile,
    *,
    seed: int,
    modules: list[str],
    dtype: torch.dtype = torch.float32,
    cross_stream_rank: int | None = None,
    alora_invocation_tokens: list[int] | None = None,
    last_context_token: tuple[str, int] | None = None,
) -> Path:
    """Write a synthetic PEFT adapter directory matching ``profile``.

    ``modules`` are PEFT module paths (see :func:`attn_modules` /
    :func:`mlp_modules`); a rank tier is reached by choosing the profile, and a
    module is left un-adapted simply by leaving it out — which is how a
    multi-adapter checkpoint gets modules where only some adapters apply.

    ``cross_stream_rank`` adds the Shadow-Residual layer-level shunt.
    ``alora_invocation_tokens`` marks the adapter aLoRA (control token goes
    immediately before that sequence); ``last_context_token`` marks it SR
    (control token replaces the generation-prompt anchor). Both come from the
    base tokenizer — the composer re-encodes and cross-checks them.
    """
    table = _module_table(dims)
    unknown = [m for m in modules if m not in table]
    if unknown:
        raise ValueError(f"unknown modules {unknown}; known: {sorted(table)}")

    state: dict[str, torch.Tensor] = {}
    for layer in range(dims.num_layers):
        target_rel = profile.rel_fro(layer, dims.num_layers)
        for module in modules:
            base_suffix, out_features, in_features = table[module]
            base_key = f"model.layers.{layer}.{base_suffix}.weight"
            if base_key not in norms:
                raise KeyError(
                    f"{base_key} absent from the base weight norms — the module "
                    "table does not match this checkpoint"
                )
            # The checkpoint stores the raw factors; PEFT multiplies by alpha/r at
            # apply time, so divide it out of the target here.
            target_fro = target_rel * norms[base_key] / profile.peft_scale
            a, b = factorize(
                out_features,
                in_features,
                rank=profile.rank,
                target_fro=target_fro,
                stable_rank=profile.stable_rank,
                b_over_a=profile.b_over_a,
                generator=_seeded(f"{layer}.{module}", seed),
            )
            key = f"{PEFT_PREFIX}layers.{layer}.{module}"
            state[f"{key}.lora_A.weight"] = a.to(dtype)
            state[f"{key}.lora_B.weight"] = b.to(dtype)

        if cross_stream_rank:
            ref = f"model.layers.{layer}.{CROSS_STREAM_REFERENCE}.weight"
            target_fro = target_rel * norms[ref] / profile.peft_scale
            a, b = factorize(
                dims.hidden,
                dims.hidden,
                rank=cross_stream_rank,
                target_fro=target_fro,
                stable_rank=profile.stable_rank,
                b_over_a=profile.b_over_a,
                generator=_seeded(f"{layer}.cross_stream", seed),
            )
            key = f"{PEFT_PREFIX}layers.{layer}.cross_stream"
            state[f"{key}.lora_A.weight"] = a.to(dtype)
            state[f"{key}.lora_B.weight"] = b.to(dtype)

    target_modules = _target_modules(modules)
    config = {
        "r": profile.rank,
        "lora_alpha": profile.alpha,
        "target_modules": sorted(
            target_modules + (["cross_stream"] if cross_stream_rank else [])
        ),
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "peft_type": "LORA",
    }
    if cross_stream_rank:
        config["rank_pattern"] = {"cross_stream": cross_stream_rank}
        config["alpha_pattern"] = {"cross_stream": cross_stream_rank}
    if last_context_token is not None:
        # SR activates at the generation-prompt anchor. The composer probes the
        # aLoRA key FIRST, so an SR adapter must not carry one.
        config["last_context_token"] = last_context_token[0]
        config["last_context_token_id"] = int(last_context_token[1])
    elif alora_invocation_tokens is not None:
        config["alora_invocation_tokens"] = list(alora_invocation_tokens)

    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text(json.dumps(config, indent=2))
    save_file(state, str(path / "adapter_model.safetensors"))
    return path


# ---------------------------------------------------------------------------
# Activation anchors, read off the base tokenizer
# ---------------------------------------------------------------------------


def generation_prompt_suffix(
    tokenizer, messages, documents=None, **template_kwargs
) -> str:
    """The text ``add_generation_prompt=True`` appends, e.g. the assistant marker.

    Taken as a difference of two renders rather than hardcoded, so it works for
    both Granite template families (4.0/4.1 role markers and 4.2 ChatML).
    ``template_kwargs`` pass through, so the anchor comes from the same render
    the prompts will use.
    """
    kw = dict(documents=documents, tokenize=False, **template_kwargs)
    with_gen = tokenizer.apply_chat_template(messages, add_generation_prompt=True, **kw)
    without = tokenizer.apply_chat_template(messages, add_generation_prompt=False, **kw)
    if not with_gen.startswith(without):
        raise ValueError(
            "the generation prompt is not a suffix of the plain render; cannot "
            "derive an aLoRA invocation sequence from this template"
        )
    suffix = with_gen[len(without) :]
    if not suffix:
        raise ValueError("add_generation_prompt added nothing to the render")
    return suffix


def alora_invocation_ids(
    tokenizer, messages, documents=None, **template_kwargs
) -> list[int]:
    """Invocation token ids placing the control token before the generation prompt.

    That is one of the two documented aLoRA placements, and unlike a
    task-specific invocation string it is guaranteed to occur in every rendered
    chat, so the control token always lands mid-prompt rather than at the start.
    """
    suffix = generation_prompt_suffix(tokenizer, messages, documents, **template_kwargs)
    ids = list(tokenizer(suffix, add_special_tokens=False).input_ids)
    if not ids:
        raise ValueError(f"generation prompt suffix {suffix!r} encodes to nothing")
    return ids


def sr_activation_anchor(
    tokenizer, messages, documents=None, **template_kwargs
) -> tuple[str, int]:
    """``(last_context_token, id)`` an SR adapter records for this base.

    SR is always-active during training, so instead of an invocation sequence its
    checkpoint marks the prompt/completion boundary — the final token of the
    generation prompt — and the control token REPLACES it.
    """
    text = tokenizer.apply_chat_template(
        messages,
        documents=documents,
        add_generation_prompt=True,
        tokenize=False,
        **template_kwargs,
    )
    ids = tokenizer(text, add_special_tokens=False).input_ids
    last_id = int(ids[-1])
    token = tokenizer.convert_ids_to_tokens(last_id)
    encoded = tokenizer.encode(token, add_special_tokens=False)
    if encoded != [last_id]:
        raise ValueError(
            f"generation prompt ends with id {last_id} ({token!r}) but that "
            f"literal re-encodes to {encoded}; the control-token swap replaces "
            "exactly one embedding, so an SR anchor must round-trip."
        )
    return token, last_id


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def adapter_facts(adapter_dir: Path) -> dict:
    """Rank, alpha, target modules and per-tensor magnitudes, for logging.

    Read back off disk rather than taken from the profile, so the log shows
    what the checkpoint will actually carry. The all-zero ``lora_B`` count is
    the one the caller refuses to proceed on: such a tensor contributes no
    delta whatever its ``lora_A`` holds.
    """
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    weights = load_file(str(adapter_dir / "adapter_model.safetensors"))
    a_keys = [k for k in weights if "lora_A" in k]
    b_keys = [k for k in weights if "lora_B" in k]
    zero_b = [k for k in b_keys if not weights[k].any()]
    return {
        "rank": cfg.get("r"),
        "alpha": cfg.get("lora_alpha"),
        "target_modules": sorted(cfg.get("target_modules") or []),
        "alora": bool(cfg.get("alora_invocation_tokens")),
        "num_lora_A": len(a_keys),
        "num_lora_B": len(b_keys),
        "num_zero_lora_B": len(zero_b),
        "lora_A_std": float(
            torch.cat([weights[k].float().flatten() for k in a_keys]).std()
        ),
        "lora_B_std": float(
            torch.cat([weights[k].float().flatten() for k in b_keys]).std()
        ),
    }


def measure_adapter(
    adapter_dir: Path, norms: dict[str, float], layers=(0,)
) -> list[dict]:
    """Re-measure a written adapter the way the profiles were fitted.

    Returns one row per (layer, module) with the effective delta's relative
    Frobenius norm, stable rank and A/B ratio, so a test can assert that what
    was written actually matches its profile.
    """
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    rank_pattern = cfg.get("rank_pattern") or {}
    rows = []
    with safe_open(str(adapter_dir / "adapter_model.safetensors"), "pt") as handle:
        keys = set(handle.keys())
        for layer in layers:
            prefix = f"{PEFT_PREFIX}layers.{layer}."
            modules = sorted(
                k[len(prefix) : -len(".lora_A.weight")]
                for k in keys
                if k.startswith(prefix) and k.endswith(".lora_A.weight")
            )
            for module in modules:
                a = handle.get_tensor(f"{prefix}{module}.lora_A.weight").float()
                b = handle.get_tensor(f"{prefix}{module}.lora_B.weight").float()
                delta = (b @ a) * scale
                sv = torch.linalg.svdvals(delta)
                fro = float(torch.linalg.matrix_norm(delta))
                ref = CROSS_STREAM_REFERENCE if module == "cross_stream" else module
                base_norm = norms.get(f"model.layers.{layer}.{ref}.weight")
                rows.append(
                    {
                        "layer": layer,
                        "module": module,
                        "rank": rank_pattern.get(module, cfg["r"]),
                        "rel_fro": fro / base_norm if base_norm else None,
                        "stable_rank": fro**2 / float(sv[0]) ** 2,
                        "b_over_a": float(b.std() / a.std()),
                    }
                )
    return rows


__all__ = [
    "CROSS_STREAM_REFERENCE",
    "PEFT_PREFIX",
    "PROFILES",
    "AdapterProfile",
    "adapter_facts",
    "alora_invocation_ids",
    "attn_modules",
    "base_weight_norms",
    "factorize",
    "generation_prompt_suffix",
    "measure_adapter",
    "mlp_modules",
    "spectrum",
    "sr_activation_anchor",
    "synthesize_adapter",
]
