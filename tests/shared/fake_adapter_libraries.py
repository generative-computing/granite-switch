# SPDX-License-Identifier: Apache-2.0
"""Fake granitelib adapter libraries, built from captured copies of the real ones.

The composer reads an adapter library's *layout and metadata*: the
``<adapter>/<base>/<lora|alora>/`` tree, each ``adapter_config.json`` (rank, alpha,
target modules, aLoRA invocation tokens) and each ``io.yaml``. The weights it only
stacks. So, like :mod:`tests.shared.fake_bases`, each fake library is the real
repo's small files, captured verbatim into
``tests/shared/fixtures/adapter_libraries/<repo>/`` by ``capture()``, plus
synthetic weights generated here at test time for a fake base.

The weights come from :func:`tests.shared.synthetic_adapters.synthesize_adapter`
at the captured rank and target modules, and are fitted to the fake base's own
weight norms. They carry no trained behaviour, which none of the PR-run tests
need; the release tier runs the same tests against the real library.

Refresh with ``python -m tests.shared.fake_adapter_libraries --capture``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
from pathlib import Path

from tests.shared.base_models import dims_from_config
from tests.shared.synthetic_adapters import (
    PROFILES,
    attn_modules,
    base_weight_norms,
    mlp_modules,
    synthesize_adapter,
)

FIXTURES = Path(__file__).parent / "fixtures" / "adapter_libraries"

#: Hub repo -> base flavours captured. Only the bases the fake bases cover.
LIBRARIES = {
    "ibm-granite/granite-lib-rag-r1.0": ("granite-4.0-micro", "granite-4.1-3b"),
    "ibm-granite/granitelib-rag-r1.0": ("granite-4.0-micro", "granite-4.1-3b"),
    "ibm-granite/granitelib-core-r1.0": ("granite-4.0-micro", "granite-4.1-3b"),
    "ibm-granite/granitelib-guardian-r1.0": ("granite-4.0-micro", "granite-4.1-3b"),
}

#: Copied verbatim per adapter. The weights are synthesized instead.
CAPTURED_FILES = ("adapter_config.json", "io.yaml")

#: Library-level files copied when present. None: discovery reads only the
#: per-adapter files, and a captured README links to files not captured.
LIBRARY_FILES: tuple[str, ...] = ()


def _fixture_dir(repo: str) -> Path:
    return FIXTURES / repo.replace("/", "--")


def capture(repos=None, revision=None) -> None:
    """Refresh the captured files from the Hub. Needs network; not run in tests."""
    from huggingface_hub import hf_hub_download, list_repo_files

    for repo in repos or LIBRARIES:
        bases = LIBRARIES[repo]
        dest_root = _fixture_dir(repo)
        if dest_root.exists():
            shutil.rmtree(dest_root)
        files = list_repo_files(repo, revision=revision)
        wanted = [
            f
            for f in files
            if f in LIBRARY_FILES
            or (
                f.count("/") == 3
                and f.split("/")[1] in bases
                and f.rsplit("/", 1)[1] in CAPTURED_FILES
            )
        ]
        for f in wanted:
            src = hf_hub_download(repo, f, revision=revision)
            (dest_root / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest_root / f)
        print(f"captured {len(wanted)} files from {repo}")


def _profile_for(rank: int, alpha: float):
    """A fitted profile at exactly the published ``rank`` and ``alpha``.

    The norm, spectrum and A/B asymmetry come from the fitted profile nearest in
    rank; the rank and alpha are the real adapter's, so the synthesized factors
    agree with the captured config the composer reads.
    """
    nearest = min(PROFILES.values(), key=lambda p: abs(p.rank - rank))
    return dataclasses.replace(nearest, rank=rank, alpha=float(alpha))


def _peft_modules(target_modules: list[str], dims) -> list[str]:
    """Map PEFT target names (``q_proj``, ``down_proj`` ...) to synthesizer paths."""
    available = attn_modules() + mlp_modules(dims)
    leaf = {m.rsplit(".", 1)[-1]: m for m in available}
    return [leaf[t] for t in target_modules if t in leaf]


def build_fake_library(
    repo: str,
    base_name: str,
    base_path,
    dest,
    *,
    include: list[str] | None = None,
    seed: int = 0,
) -> Path:
    """Write a fake ``repo`` library for ``base_name`` to ``dest`` and return it.

    ``base_path`` is the fake base the weights are sized and scaled for (from
    :func:`tests.shared.fake_bases.build_fake_base`). ``include`` limits the build
    to those adapter names, which keeps a test to the adapters it uses.
    """
    from transformers import AutoConfig

    src_root = _fixture_dir(repo)
    if not src_root.is_dir():
        raise FileNotFoundError(f"no captured library {repo!r} under {FIXTURES}")
    dims = dims_from_config(AutoConfig.from_pretrained(str(base_path)))
    norms = base_weight_norms(base_path)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for f in LIBRARY_FILES:
        if (src_root / f).exists():
            shutil.copyfile(src_root / f, dest / f)

    built = 0
    for config_path in sorted(src_root.glob(f"*/{base_name}/*/adapter_config.json")):
        variant_dir = config_path.parent
        adapter = variant_dir.parent.parent.name
        if include is not None and adapter not in include:
            continue
        rel = variant_dir.relative_to(src_root)
        out = dest / rel
        real_config = json.loads(config_path.read_text())
        modules = _peft_modules(real_config["target_modules"], dims)
        if not modules:
            raise ValueError(
                f"{rel}: no target module of {real_config['target_modules']} exists on the base"
            )
        profile = _profile_for(real_config["r"], real_config["lora_alpha"])
        synthesize_adapter(
            out,
            dims,
            norms,
            profile,
            seed=seed + built,
            modules=modules,
            alora_invocation_tokens=real_config.get("alora_invocation_tokens"),
        )
        # The real config, not the synthesizer's: the rank, alpha, target modules
        # and invocation tokens the composer reads are the published ones. The
        # synthesized factors already have the published rank.
        (out / "adapter_config.json").write_text(json.dumps(real_config, indent=2))
        for f in CAPTURED_FILES:
            if f != "adapter_config.json" and (variant_dir / f).exists():
                shutil.copyfile(variant_dir / f, out / f)
        built += 1
    if built == 0:
        raise ValueError(
            f"no adapters of {repo!r} for {base_name!r} (include={include})"
        )
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh the captured adapter-library files."
    )
    parser.add_argument("--capture", action="store_true", help="refresh from Hub")
    parser.add_argument("--repos", nargs="*", choices=sorted(LIBRARIES))
    args = parser.parse_args()
    if args.capture:
        capture(args.repos)


if __name__ == "__main__":
    main()


__all__ = ["LIBRARIES", "build_fake_library", "capture"]
