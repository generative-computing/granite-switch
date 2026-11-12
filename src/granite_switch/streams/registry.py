# SPDX-License-Identifier: Apache-2.0
"""The head-type registry: how a stream type plugs in.

A head type is a :class:`HeadType` subclass, registered by name. The model
builds one module per type per backend, holding every slot of that type, and
calls it at each read layer on the rows of its markers. It never asks which
type it is: everything type-specific lives here, in the type.

Types register in one of two ways:

* built in -- :mod:`.types` imports and registers them (``probe``);
* as a plugin -- a package exposes a ``HeadType`` subclass under the
  ``granite_switch.streams`` entry-point group::

      [project.entry-points."granite_switch.streams"]
      mlp_head = "my_package.mlp_head:MLPHeadType"

  and Granite Switch loads it the first time the name is looked up.

The module contract (both backends)::

    module(hidden, rows) -> scores
      hidden: [n, hidden_size]  the read layer's state at n marker rows
      rows:   [n]               each marker's head_row in this type's bank
      scores: [n, width]        one score per output token, width >= num_outputs

The vLLM module is called on every token of a step (branch-free, so it runs
under CUDA graphs); rows of tokens that are not this type's markers carry
row 0 and their scores are masked out by the model.
"""

from importlib.metadata import entry_points

from torch import nn

ENTRY_POINT_GROUP = "granite_switch.streams"

_REGISTRY: dict[str, type["HeadType"]] = {}


class HeadType:
    """One head stream type. Subclass, set ``name``, implement the hooks."""

    #: Registry key, and the ``type`` of the stream entries it serves.
    name: str = ""

    #: Weight file a slot folder ships (compose loads it into ``head_row``).
    weights_file: str = "head.safetensors"

    #: The model attribute holding this type's module, and so its checkpoint key
    #: prefix (``model.<module_name>.*``). None means ``stream_<name>``.
    module_name: str | None = None

    def check(self, spec, config) -> None:
        """Validate one slot's ``settings`` (raise ``ValueError``). Default: none."""

    def hf_module(self, specs, config) -> nn.Module:
        """The HF module holding every slot of this type (``specs``)."""
        raise NotImplementedError

    def vllm_module(self, specs, config) -> nn.Module:
        """The vLLM module holding every slot of this type (``specs``).

        Defaults to the HF module, which is fine when its forward is already
        branch-free (no data-dependent Python control flow).
        """
        return self.hf_module(specs, config)

    def load_weights(self, module: nn.Module, row: int, state: dict, spec) -> None:
        """Copy one slot's trained weights (``state``, from ``weights_file``) into
        ``module``'s bank at ``row``. Used by compose."""
        raise NotImplementedError


def register_head_type(cls: type[HeadType]) -> type[HeadType]:
    """Register a head type (usable as a class decorator)."""
    if not cls.name:
        raise ValueError(f"{cls.__name__} has no name.")
    if cls.name == "lora":
        raise ValueError("'lora' is the built-in adapter stream, not a head type.")
    _REGISTRY[cls.name] = cls
    return cls


def _load_plugins() -> None:
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        if ep.name not in _REGISTRY:
            register_head_type(ep.load())


def get_head_type(name: str) -> HeadType:
    """The registered head type ``name`` (plugins are loaded on a miss)."""
    from . import types  # noqa: F401  (registers the built-in types)

    if name not in _REGISTRY:
        _load_plugins()
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown stream type {name!r}. Known: {sorted(_REGISTRY)}. A plugin "
            f"type registers under the {ENTRY_POINT_GROUP!r} entry-point group; "
            "install its package."
        )
    return _REGISTRY[name]()


def head_type_names() -> list[str]:
    """Every registered head type (built in and installed plugins)."""
    from . import types  # noqa: F401

    _load_plugins()
    return sorted(_REGISTRY)
