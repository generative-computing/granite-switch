# SPDX-License-Identifier: Apache-2.0
"""One test, two sources: a fake on every PR run, the real model in the release tier.

PR runs must not download real models; the release tier runs the same tests on the
real ones (docs/CICD.md, "Release test tier"). A test that needs a base model or an
adapter library takes it from here instead of naming a Hub id, and is parametrized
over :data:`SOURCES` with :func:`sources`:

* ``"fake"``: a tiny base built from the captured real files
  (:mod:`tests.shared.fake_bases`) and fake adapter libraries with the captured
  real configs (:mod:`tests.shared.fake_adapter_libraries`). Built once per test
  session.
* ``"real"``: the Hub id itself, marked ``release`` so PR runs deselect it.

The two run the same assertions. The fake proves our code against the formats we
captured; the real run proves the captures still match what is published.
"""

from __future__ import annotations

import tempfile
from functools import cache
from pathlib import Path

import pytest

from tests.shared.fake_adapter_libraries import build_fake_library
from tests.shared.fake_bases import (
    BASES,
    REALISTIC,
    TINY_DIMS,
    WeightPreset,
    build_fake_base,
)

SOURCES = ("fake", "real")

#: A session-wide build directory, shared across tests and xdist-safe per process.
_ROOT = Path(tempfile.mkdtemp(prefix="gs-fake-models-"))


def sources(*, real_marks=()):
    """``pytest.param`` values for :data:`SOURCES`, with the real one release-marked.

    ``real_marks`` adds marks to the real case only, e.g. ``requires_model``.
    """
    return [
        pytest.param("fake", id="fake"),
        pytest.param("real", id="real", marks=(pytest.mark.release, *real_marks)),
    ]


def _preset_tag(preset: WeightPreset, dims: tuple) -> str:
    d = dict(dims)
    shape = (
        f"h{d['hidden_size']}-q{d['num_attention_heads']}-kv{d['num_key_value_heads']}"
    )
    return f"b{preset.body:g}-e{preset.embed:g}-n{preset.final_norm:g}-{shape}"


def _dims_key(dims: dict | None) -> tuple:
    return tuple(sorted((TINY_DIMS if dims is None else dims).items()))


@cache
def _fake_base(name: str, preset: WeightPreset, dims: tuple) -> str:
    # The directory is named after the model, as a Hub id's last component is:
    # the composer derives --target-model from it, which picks the base flavour
    # inside an adapter library. So the preset and shape go in the parent.
    dest = _ROOT / "bases" / _preset_tag(preset, dims) / name
    return str(build_fake_base(name, dest, preset=preset, dims=dict(dims)))


@cache
def _fake_library(
    repo: str,
    base_name: str,
    include: tuple[str, ...] | None,
    preset: WeightPreset,
    dims: tuple,
) -> str:
    tag = "-".join(include) if include else "all"
    dest = (
        _ROOT
        / "libraries"
        / _preset_tag(preset, dims)
        / base_name
        / repo.replace("/", "--")
        / tag
    )
    return str(
        build_fake_library(
            repo,
            base_name,
            _fake_base(base_name, preset, dims),
            dest,
            include=list(include) if include else None,
        )
    )


def base_model(
    source: str,
    name: str,
    *,
    preset: WeightPreset = REALISTIC,
    dims: dict | None = None,
) -> str:
    """A path or Hub id for base ``name`` (a key of ``fake_bases.BASES``).

    ``preset`` picks the fake's random-weight scales (``fake_bases.REALISTIC`` or
    ``LIVE_LAYERS``) and ``dims`` its size (``fake_bases.TINY_DIMS`` by default,
    ``TP_DIMS`` for the tensor-parallel tests); the real source ignores both.
    """
    if source == "real":
        return BASES[name]
    return _fake_base(name, preset, _dims_key(dims))


def adapter_library(
    source: str,
    repo: str,
    base_name: str,
    *,
    include=None,
    preset: WeightPreset = REALISTIC,
    dims: dict | None = None,
) -> str:
    """A path or Hub id for adapter library ``repo``, built for ``base_name``.

    ``include`` limits a fake library to those adapters; the real library is
    passed whole, so a test that needs only some adapters should also pass the
    composer's own ``--include-adapters``.
    """
    if source == "real":
        return repo
    return _fake_library(
        repo, base_name, tuple(include) if include else None, preset, _dims_key(dims)
    )


def lora_adapter_dir(source: str, repo: str, base_name: str, adapter: str) -> str:
    """The directory of one LoRA adapter of ``repo`` for ``base_name``.

    For tests that need a single adapter directory (a YAML manifest entry, say)
    rather than a library. The real source downloads only that adapter.
    """
    rel = f"{adapter}/{base_name}/lora"
    if source == "real":
        from huggingface_hub import snapshot_download

        return str(Path(snapshot_download(repo, allow_patterns=[f"{rel}/*"])) / rel)
    return str(Path(adapter_library(source, repo, base_name, include=[adapter])) / rel)


__all__ = [
    "SOURCES",
    "adapter_library",
    "base_model",
    "lora_adapter_dir",
    "sources",
]
