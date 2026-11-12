# SPDX-License-Identifier: Apache-2.0
"""The head-type registry: how a stream type plugs in.

A head type is a :class:`HeadType` subclass, registered under its ``name``.
The model builds one module per type per backend, holding every slot of that
type, and calls it at each read layer. It never asks which type it runs:
everything type-specific lives in the type.

A type registers in one of two ways:

* built in -- :mod:`.types` registers its types when imported (``probe``);
* as a plugin -- an installed package names the class under the
  ``granite_switch.streams`` entry-point group, keyed by the type's ``name``::

      [project.entry-points."granite_switch.streams"]
      mlp_head = "my_package.mlp_head:MLPHeadType"

  Granite Switch imports it the first time a checkpoint names ``mlp_head``.
  Only that entry point is imported, so a plugin that fails to import affects
  only checkpoints that use it.

A name belongs to one class: registering a different class under a name that
is taken is refused, so a plugin can't replace the built-in ``probe``. See
docs/STREAM_API.md for a worked example.
"""

from importlib.metadata import EntryPoint, entry_points
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from granite_switch.config import GraniteSwitchConfig

    from .spec import StreamSpec

ENTRY_POINT_GROUP = "granite_switch.streams"

_REGISTRY: dict[str, type["HeadType"]] = {}


class HeadType:
    """Base class of a head stream type: subclass it to add one.

    A head reads the hidden state at its slot's marker token and returns one
    score per output token; Granite Switch writes the scores into that row's
    logits. One module holds every slot of a type, one bank row per slot.

    Required: ``name``, :meth:`hf_module` and :meth:`load_weights`. Optional:
    ``weights_file``, ``module_name``, :meth:`check`, and :meth:`vllm_module`
    (vLLM runs the HF module by default).

    The module's call, on both backends::

        scores = module(hidden, rows)
          hidden  [n, hidden_size]  the read layer's state, in the model's dtype
          rows    [n] long          1 + the slot's head_row; 0 = not this type's marker
          scores  [n, width]        any float dtype; column k lands on the slot's
                                    token_ids[k], and extra columns are ignored

    HF passes only the marker tokens. vLLM passes every token of the step and
    runs the module inside a CUDA graph, so the module must not read values back
    to Python or let a shape depend on the data (docs/STREAM_API.md).
    """

    #: The ``type`` of the stream entries this class serves (required).
    name: str = ""

    #: The file a slot's folder ships; compose reads it into the slot's row.
    weights_file: str = "head.safetensors"

    #: The model attribute holding the module, and so its checkpoint key prefix
    #: (``model.<module_name>.*``). None means ``stream_<name>``.
    module_name: str | None = None

    def check(self, spec: "StreamSpec", config: "GraniteSwitchConfig") -> None:
        """Validate one slot's ``spec.settings``; raise ``ValueError`` if invalid."""

    def hf_module(
        self, specs: "list[StreamSpec]", config: "GraniteSwitchConfig"
    ) -> nn.Module:
        """Build the HF module that holds every slot of this type.

        Size it from ``specs``: ``1 + max(s.head_row)`` bank rows, and at least
        ``max(s.num_outputs)`` score columns.
        """
        raise NotImplementedError

    def vllm_module(
        self, specs: "list[StreamSpec]", config: "GraniteSwitchConfig"
    ) -> nn.Module:
        """Build the vLLM module; by default the HF one, which must then follow
        the vLLM rules above."""
        return self.hf_module(specs, config)

    def load_weights(
        self,
        module: nn.Module,
        row: int,
        state: dict[str, torch.Tensor],
        spec: "StreamSpec",
    ) -> None:
        """Copy one slot's trained weights into ``module``; compose calls this.

        Args:
            module: this type's module, as :meth:`hf_module` built it.
            row: the slot's bank row, ``spec.head_row`` (0-based; the module's
                ``rows`` argument is this plus one).
            state: the tensors of the slot's ``weights_file``.
            spec: the slot's stream entry.
        """
        raise NotImplementedError


def _qualname(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def register_head_type(cls: type[HeadType]) -> type[HeadType]:
    """Register a head type under its ``name`` (usable as a class decorator).

    Registering the same class again is allowed (a re-imported copy replaces
    the old one); a different class under a name that is taken is refused.
    """
    if not (isinstance(cls, type) and issubclass(cls, HeadType)):
        raise TypeError(f"{cls!r} is not a HeadType subclass.")
    if not cls.name:
        raise ValueError(f"{cls.__name__} has no name.")
    if cls.name == "lora":
        raise ValueError("'lora' is the built-in adapter stream, not a head type.")
    taken = _REGISTRY.get(cls.name)
    if taken is not None and _qualname(taken) != _qualname(cls):
        raise ValueError(
            f"Head type {cls.name!r} is already registered by {_qualname(taken)}; "
            f"{_qualname(cls)} can't take its name."
        )
    _REGISTRY[cls.name] = cls
    return cls


def _register_entry_point(ep: EntryPoint) -> None:
    cls = ep.load()
    where = f"Entry point {ep.name!r} ({ep.value}) in {ENTRY_POINT_GROUP!r}"
    if not (isinstance(cls, type) and issubclass(cls, HeadType)):
        raise TypeError(f"{where} is not a HeadType subclass: {cls!r}.")
    if cls.name != ep.name:
        raise ValueError(
            f"{where} loads head type {cls.name!r}; name the entry point after "
            "the type."
        )
    register_head_type(cls)


def get_head_type(name: str) -> HeadType:
    """The head type registered as ``name``; a plugin is imported on first use."""
    from . import types  # noqa: F401  (registers the built-in types)

    if name not in _REGISTRY:
        for ep in entry_points(group=ENTRY_POINT_GROUP, name=name):
            _register_entry_point(ep)
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown stream type {name!r}. Known: {head_type_names()}. A plugin "
            f"type registers under the {ENTRY_POINT_GROUP!r} entry-point group; "
            "install its package."
        )
    return cls()


def head_type_names() -> list[str]:
    """Every head type's name: the registered ones and the installed plugins'.

    Plugins are listed by their entry points, without importing them.
    """
    from . import types  # noqa: F401  (registers the built-in types)

    installed = {ep.name for ep in entry_points(group=ENTRY_POINT_GROUP)}
    return sorted(set(_REGISTRY) | installed)
