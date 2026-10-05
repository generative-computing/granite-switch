# SPDX-License-Identifier: Apache-2.0
# Copied from the switch benchmark (granite-switch-staging, branch
# feature/switch-benchmark, commit 0359602) and kept unchanged, so this benchmark
# measures switching exactly as it does. Update it only by copying it again.
"""Build a synthetic N-adapter fleet for the SWITCH.md throughput reproduction.

Replaces the reference harness's ``make_multi_lora.py``, which cloned shapes and
keys out of one *template* PEFT directory on an internal filesystem. Here the
shapes come from the base model's own config via
:func:`tests.shared.base_models.dims_from_config`, and the weights from
:mod:`tests.shared.synthetic_adapters`. Three consequences, all of which matter:

* **No template.** Any base works, including one with no published adapters.
* **Correct key spellings by construction.** ``_module_table`` emits
  ``mlp.gate_proj`` for a dense base and ``shared_mlp.input_linear`` for a
  ``granitemoehybrid`` one. The real granitelib/When2Call adapters store MLP LoRA
  *parentless* (``layers.N.gate_proj...``), which both our composer and stock
  vLLM drop **silently** -- yielding an attention-only model with no error. A
  template-cloning builder inherits that bug; this one cannot express it.
* **A trained delta's statistics, not iid noise.** Irrelevant to latency, which
  is value-independent, but it means the same fleet is reusable by correctness
  tests, and it guarantees dense ``lora_A`` so the composer registers every
  adapter as *covering* its modules and the expand kernel actually fires.

Layout -- a composer *library* whose leaves are simultaneously valid standalone
PEFT directories, so one artifact feeds every arm and the arms are provably
weight-identical::

    <output>/adapter_00/<target-model>/<tech>/{adapter_config.json,
                                               adapter_model.safetensors,
                                               io.yaml}
    ...
    <output>/manifest.json

    switch arms : --adapters <output>            (leaf name -> control token)
    native arms : --lora-path <output>/adapter_00/<target-model>/<tech>

Two flavors, matching the two adaptation tiers the vLLM backend dispatches on:

``lora`` / ``alora``
    All seven linear projections -- attention ``q_proj``, ``k_proj``, ``v_proj``,
    ``o_proj`` **and** MLP ``gate_proj``, ``up_proj``, ``down_proj``. ("Dense"
    below distinguishes a dense base from a MoE one, i.e. which MLP *spelling*
    the keys use; it does not mean MLP-only.) ``alora`` additionally
    records ``alora_invocation_tokens``. At serve time the two are identical --
    activation is purely the control token, and the invocation sequence only
    steers template-driven token placement -- so the reproduction uses ``lora``,
    matching the reference's ``--technology lora``.

``sr``
    Shadow Residual: ``{q, o, gate, up, down}`` plus the layer-level
    ``cross_stream`` shunt, and ``last_context_token`` instead of an aLoRA
    invocation sequence. SR reads K/V from the base stream, so it covers neither
    k nor v; the composer probes the aLoRA key first, so an SR adapter must not
    carry one.

    An SR adapter **always** carries its shunt -- there is no shunt-less SR
    variant, and the same fleet feeds both the switch arm and the stock-vLLM arm.
    Stock vLLM cannot bind ``cross_stream`` to anything (its Granite model has no
    layer-level LoRA site) and its ``check_unexpected_modules`` would *raise* on
    those tensors, so the stock-vLLM arm passes
    ``lora_skip_prefixes=["cross_stream"]`` -- read by
    ``vllm/lora/worker_manager.py`` as a plain attribute on the model instance,
    and honoured *before* the expected-modules check. The artifact is therefore
    byte-identical between the two arms, and what the stock arm loses is exactly
    what stock vLLM structurally cannot express.

Usage::

    python benchmarks/make_synth_fleet.py --base ibm-granite/granite-4.1-3b \
        --flavor lora --num-adapters 12 --rank 32 --output $ROOT/fleet/lora
    python benchmarks/make_synth_fleet.py --base ibm-granite/granite-4.1-3b \
        --flavor sr   --num-adapters 12 --rank 32 --output $ROOT/fleet/sr

Add one off-rank adapter to an existing fleet (heterogeneous-rank study)::

    python benchmarks/make_synth_fleet.py ... --rank 128 --profile strong \
        --single-name adapter_r128 --seed 128
"""

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

# tests/shared is the single source of truth for the generator; see the module
# docstring in tests/shared/synthetic_adapters.py for how the profiles were fitted.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

import torch
from shared.base_models import dims_from_config
from shared.synthetic_adapters import (
    PROFILES,
    adapter_facts,
    alora_invocation_ids,
    attn_modules,
    base_weight_norms,
    mlp_modules,
    sr_activation_anchor,
    synthesize_adapter,
)

#: Rank -> the profile fitted at that rank. Ranks are on the SWITCH kernel's
#: supported tiers; a rank outside this map requires an explicit --profile.
PROFILE_BY_RANK = {16: "light", 32: "medium", 64: "strong"}

_DTYPE = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}

#: io.yaml is required by the compose CLI unless --create-ioyaml is passed. The
#: composer copies it verbatim and never parses the body, so every field is null.
IOYAML = """\
# Synthetic adapter for throughput benchmarking. Generated by
# benchmarks/make_synth_fleet.py -- do not use for any quality claim.
name: {name}
description: null
input: null
output: null
"""


def resolve_profile(rank: int, profile_name: str | None):
    """The :class:`AdapterProfile` to synthesize at ``rank``.

    Only shapes affect latency, so re-ranking a fitted profile is sound here: the
    spectrum, relative norm and A/B asymmetry targets are all scale-free or
    expressed relative to the base weight. Alpha follows the reference fleet's
    ``alpha == rank`` convention whenever the rank is overridden, which keeps the
    PEFT scale at 1.0 and so keeps the delta norm equal to the fitted target.
    """
    if profile_name is None:
        profile_name = PROFILE_BY_RANK.get(rank)
        if profile_name is None:
            raise SystemExit(
                f"FATAL: rank {rank} has no fitted profile "
                f"(have {sorted(PROFILE_BY_RANK)}); pass --profile explicitly."
            )
    if profile_name not in PROFILES:
        raise SystemExit(
            f"FATAL: unknown profile {profile_name!r}; have {sorted(PROFILES)}."
        )
    profile = PROFILES[profile_name]
    if profile.rank != rank:
        profile = dataclasses.replace(profile, rank=rank, alpha=float(rank))
    return profile


def flavor_spec(flavor: str, dims, cross_rank: int, get_tokenizer, override):
    """``(technology_dir, modules, extra_kwargs)`` for one flavor.

    ``extra_kwargs`` carries the activation marker, which is what tells the
    composer whether this is an aLoRA adapter or an SR one. ``get_tokenizer`` is
    called only by the flavors that need one -- the plain ``lora`` flavor carries
    no marker, so it can build against a base that ships no tokenizer files
    (which is how the CPU test builds against a tiny synthetic base).
    """
    probe = [{"role": "user", "content": "x"}]
    if flavor == "sr":
        anchor = override or sr_activation_anchor(get_tokenizer(), probe)
        return (
            "alora",  # --technology has no "sr" value; SR is detected from contents
            attn_modules(with_kv=False) + mlp_modules(dims),
            {"cross_stream_rank": cross_rank, "last_context_token": anchor},
        )
    modules = attn_modules(with_kv=True) + mlp_modules(dims)
    if flavor == "alora":
        ids = override or alora_invocation_ids(get_tokenizer(), probe)
        return "alora", modules, {"alora_invocation_tokens": list(ids)}
    return "lora", modules, {}


def assert_remappable(adapter_dir: Path, is_sr: bool, label: str):
    """Hard-fail unless the composer can remap every tensor we just wrote.

    The composer skips what it cannot remap (a bare ``continue`` in
    weight_transfer.py), so an unmapped tensor is weight that would vanish with
    no error and leave an arm silently measuring the base model. This is the
    single most important check in the file.
    """
    from safetensors import safe_open

    from granite_switch.composer.arch import granite_dense_arch, granite_dense_sr_arch
    from granite_switch.composer.weight_remapper import AdapterRemapper

    arch = granite_dense_sr_arch() if is_sr else granite_dense_arch()
    remapper = AdapterRemapper(arch.groups)
    with safe_open(str(adapter_dir / "adapter_model.safetensors"), "pt") as handle:
        keys = sorted(handle.keys())
    unmapped = [k for k in keys if remapper.remap_adapter_name(k) is None]
    if unmapped:
        shown = "\n  ".join(unmapped[:12])
        more = f"\n  ... and {len(unmapped) - 12} more" if len(unmapped) > 12 else ""
        raise SystemExit(
            f"FATAL [{label}]: {len(unmapped)} of {len(keys)} tensors do not remap "
            f"onto the switch model. The composer would silently DROP these:\n"
            f"  {shown}{more}"
        )
    return len(keys)


def assert_cross_stream(adapter_dir: Path, dims, expect: bool, label: str):
    """Exactly ``2 * num_layers`` cross_stream tensors, or exactly zero.

    vLLM's ``check_unexpected_modules`` keys off the safetensors **tensor keys**,
    not ``target_modules`` -- so a stray ``cross_stream`` tensor is a hard
    ValueError at adapter-add time for any stock-vLLM arm, while a *missing* one
    silently under-builds the SR shunt. Both directions are checked.
    """
    from safetensors import safe_open

    with safe_open(str(adapter_dir / "adapter_model.safetensors"), "pt") as handle:
        found = [k for k in handle.keys() if ".cross_stream." in k]
    want = 2 * dims.num_layers if expect else 0
    if len(found) != want:
        raise SystemExit(
            f"FATAL [{label}]: {len(found)} cross_stream tensors, expected {want} "
            f"(2 x {dims.num_layers} layers)"
            if expect
            else f"FATAL [{label}]: {len(found)} cross_stream tensors present; a "
            "stock-vLLM arm would die with ValueError on the first of them: "
            f"{found[:2]}"
        )


def assert_shapes(adapter_dir: Path, dims, rank: int, cross_rank: int, label: str):
    """Every lora_A is ``[r, in]`` and every lora_B is ``[out, r]`` for its module."""
    from safetensors import safe_open

    widths = {
        "self_attn.q_proj": (dims.q_width, dims.hidden),
        "self_attn.k_proj": (dims.kv_width, dims.hidden),
        "self_attn.v_proj": (dims.kv_width, dims.hidden),
        "self_attn.o_proj": (dims.hidden, dims.q_width),
        "mlp.gate_proj": (dims.intermediate, dims.hidden),
        "mlp.up_proj": (dims.intermediate, dims.hidden),
        "mlp.down_proj": (dims.hidden, dims.intermediate),
        "shared_mlp.input_linear": (2 * dims.intermediate, dims.hidden),
        "shared_mlp.output_linear": (dims.hidden, dims.intermediate),
        "cross_stream": (dims.hidden, dims.hidden),
    }
    bad = []
    with safe_open(str(adapter_dir / "adapter_model.safetensors"), "pt") as handle:
        for key in sorted(handle.keys()):
            body = key.split("layers.", 1)[1]
            module = body.split(".", 1)[1].rsplit(".lora_", 1)[0]
            if module not in widths:
                bad.append(f"{key}: unknown module {module!r}")
                continue
            out_f, in_f = widths[module]
            r = cross_rank if module == "cross_stream" else rank
            want = (r, in_f) if ".lora_A." in key else (out_f, r)
            got = tuple(handle.get_slice(key).get_shape())
            if got != want:
                bad.append(f"{key}: shape {got} != expected {want}")
    if bad:
        raise SystemExit(
            f"FATAL [{label}]: {len(bad)} tensor(s) have the wrong shape:\n  "
            + "\n  ".join(bad[:12])
        )


def compare_fleet(new_root: Path, other_root: Path, target_model: str):
    """Assert shared tensor keys are bit-identical between two fleets.

    The whole point of one generator with per-(layer, module) seeding is that a
    ``sr`` fleet and a ``lora`` fleet agree exactly on q/o/gate/up/down and
    differ only where the module sets differ. That makes ``srA`` vs ``aloraA`` a
    controlled comparison instead of two unrelated weight populations. Checked,
    not assumed.
    """
    import torch
    from safetensors.torch import load_file

    total_shared = 0
    for leaf in sorted(new_root.glob(f"*/{target_model}/*")):
        name = leaf.parent.parent.name
        others = list(other_root.glob(f"{name}/{target_model}/*"))
        if not others:
            print(f"  compare: {name} absent from {other_root}, skipped")
            continue
        a = load_file(str(leaf / "adapter_model.safetensors"))
        b = load_file(str(others[0] / "adapter_model.safetensors"))
        shared = sorted(set(a) & set(b))
        bad = [k for k in shared if not torch.equal(a[k], b[k])]
        if bad:
            raise SystemExit(
                f"FATAL: {len(bad)} of {len(shared)} shared tensors differ "
                f"between {leaf} and {others[0]}; the fleets are not a "
                f"controlled pair. First: {bad[:3]}"
            )
        total_shared += len(shared)
        print(f"  compare {name}: {len(shared)} shared tensors bit-identical")
    print(
        f"  CROSS-FLEET OK: {total_shared} shared tensors bit-identical vs {other_root}"
    )


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--base",
        required=True,
        help="Base model path or HF repo id. Its weights are read "
        "once (one streaming disk pass) for the per-layer "
        "Frobenius norms the profiles are expressed against.",
    )
    parser.add_argument("--output", required=True, help="Library root to create.")
    parser.add_argument("--flavor", required=True, choices=["lora", "alora", "sr"])
    parser.add_argument("--num-adapters", type=int, default=12)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument(
        "--cross-stream-rank",
        type=int,
        default=None,
        help="SR shunt rank (default: --rank).",
    )
    parser.add_argument(
        "--profile",
        default=None,
        choices=sorted(PROFILES),
        help="Override the profile chosen by --rank.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Override lora_alpha. The reference fleet used "
        "alpha == rank (PEFT scaling 1.0); the fitted "
        "profiles use alpha 32/64/64. Latency is "
        "value-independent so this changes no timing, but "
        "it must be recorded because absolute logits are "
        "then not comparable to the reference run.",
    )
    parser.add_argument(
        "--compare-fleet",
        default=None,
        metavar="ROOT",
        help="After building, assert every tensor key this fleet "
        "shares with the fleet at ROOT is BIT-IDENTICAL. "
        "Turns 'the arms differ only in module set' from a "
        "hope into a checked invariant.",
    )
    parser.add_argument(
        "--target-model",
        default="granite-4.1-3b",
        help="Middle path segment: adapter_NN/<target>/<tech>.",
    )
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=sorted(_DTYPE),
        help="On-disk dtype. bf16 matches the inference dtype.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer path/repo for deriving the activation "
        "marker (default: --base). Unused by --flavor lora.",
    )
    parser.add_argument(
        "--last-context-token",
        default=None,
        metavar="TOKEN:ID",
        help="SR only: supply the anchor literally instead of "
        "deriving it from the tokenizer, e.g. "
        "'<|end_of_role|>:100265'. For bases with no "
        "tokenizer files.",
    )
    parser.add_argument(
        "--alora-invocation-tokens",
        default=None,
        metavar="IDS",
        help="aLoRA only: comma-separated invocation token ids, "
        "instead of deriving them from the tokenizer.",
    )
    parser.add_argument(
        "--single-name",
        default=None,
        help="Emit exactly ONE adapter under this leaf name instead "
        "of the adapter_00..NN loop -- used to drop one "
        "off-rank adapter into an existing fleet.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for --single-name mode (numbered mode seeds by "
        "adapter index, so fleets agree wherever shapes do).",
    )
    args = parser.parse_args()

    if args.single_name and args.seed is None:
        parser.error("--single-name requires --seed")

    cross_rank = args.cross_stream_rank or args.rank
    if args.flavor != "sr" and args.cross_stream_rank:
        parser.error("--cross-stream-rank applies only to --flavor sr")

    if args.last_context_token and args.flavor != "sr":
        parser.error("--last-context-token applies only to --flavor sr")
    if args.alora_invocation_tokens and args.flavor != "alora":
        parser.error("--alora-invocation-tokens applies only to --flavor alora")

    override = None
    if args.last_context_token:
        token, _, tid = args.last_context_token.rpartition(":")
        if not token or not tid.isdigit():
            parser.error("--last-context-token must look like '<|token|>:12345'")
        override = (token, int(tid))
    elif args.alora_invocation_tokens:
        override = [
            int(x) for x in args.alora_invocation_tokens.split(",") if x.strip()
        ]

    from transformers import AutoConfig

    print(f"Base: {args.base}")
    dims = dims_from_config(AutoConfig.from_pretrained(args.base))
    print(f"  dims: {dims}")

    _tok = {}

    def get_tokenizer():
        """Loaded on first use, so --flavor lora never needs tokenizer files."""
        if "t" not in _tok:
            from transformers import AutoTokenizer

            _tok["t"] = AutoTokenizer.from_pretrained(args.tokenizer or args.base)
        return _tok["t"]

    print("  reading per-layer base weight norms (one streaming pass)...")
    norms = base_weight_norms(args.base)
    print(f"  {len(norms)} projection norms")

    profile = resolve_profile(args.rank, args.profile)
    if args.alpha is not None:
        profile = dataclasses.replace(profile, alpha=args.alpha)
    tech, modules, extra = flavor_spec(
        args.flavor, dims, cross_rank, get_tokenizer, override
    )
    is_sr = args.flavor == "sr"

    print(
        f"\nFlavor {args.flavor}: tech dir {tech!r}, r={profile.rank} "
        f"alpha={profile.alpha} profile={profile.name}"
        + (f" cross_stream_rank={cross_rank}" if is_sr else "")
    )
    print(f"  modules: {modules}" + (" + cross_stream" if is_sr else ""))
    for key, value in extra.items():
        if key != "cross_stream_rank":
            print(f"  {key}: {value}")

    root = Path(args.output)
    if args.single_name:
        plan = [(args.single_name, args.seed)]
    else:
        plan = [(f"adapter_{i:02d}", i) for i in range(args.num_adapters)]

    entries = []
    for name, seed in plan:
        leaf = root / name / args.target_model / tech
        synthesize_adapter(
            leaf,
            dims,
            norms,
            profile,
            seed=seed,
            modules=modules,
            dtype=_DTYPE[args.dtype],
            **extra,
        )
        (leaf / "io.yaml").write_text(IOYAML.format(name=name))

        n_tensors = assert_remappable(leaf, is_sr, name)
        assert_shapes(leaf, dims, profile.rank, cross_rank, name)
        assert_cross_stream(leaf, dims, is_sr, name)
        facts = adapter_facts(leaf)
        if facts["num_zero_lora_B"]:
            raise SystemExit(
                f"FATAL [{name}]: {facts['num_zero_lora_B']} lora_B tensors are "
                "all-zero; such an adapter contributes no delta and the composer "
                "may remap it to base, so the arm would measure nothing."
            )
        print(
            f"  {name}: {n_tensors} tensors, seed={seed}, "
            f"A_std={facts['lora_A_std']:.2e} B_std={facts['lora_B_std']:.2e} "
            f"-> {leaf}"
        )
        entries.append(
            {
                "name": name,
                "seed": seed,
                "path": str(leaf),
                "num_tensors": n_tensors,
                "safetensors_sha256_16": digest(leaf / "adapter_model.safetensors"),
                **{k: facts[k] for k in ("rank", "alpha", "target_modules")},
            }
        )

    manifest = {
        "generator": "benchmarks/make_synth_fleet.py",
        "base": args.base,
        "flavor": args.flavor,
        "alpha_override": args.alpha,
        "technology_dir": tech,
        "target_model": args.target_model,
        "profile": profile.name,
        "rank": profile.rank,
        "alpha": profile.alpha,
        "cross_stream_rank": cross_rank if is_sr else None,
        "peft_scale": profile.alpha / profile.rank,
        "dtype": args.dtype,
        "dims": dataclasses.asdict(dims),
        "modules": modules + (["cross_stream"] if is_sr else []),
        "adapters": entries,
    }
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and args.single_name:
        prior = json.loads(manifest_path.read_text())
        prior.setdefault("extra_adapters", []).extend(entries)
        manifest_path.write_text(json.dumps(prior, indent=2))
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2))

    if args.compare_fleet:
        compare_fleet(root, Path(args.compare_fleet), args.target_model)

    print(f"\nFLEET OK: {len(entries)} adapter(s) -> {root}")
    print(f"  switch arms: --adapters {root}")
    print(f"  native arms: --lora-path {root}/adapter_00/{args.target_model}/{tech}")


if __name__ == "__main__":
    main()
