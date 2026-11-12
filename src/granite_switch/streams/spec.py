# SPDX-License-Identifier: Apache-2.0
"""The stream entry: one per slot, read from ``config.streams``.

A head slot's entry says everything the model needs to run it::

    {"type": "probe",            # the registered head type
     "read_layer": 20,           # decoder layer it reads; null = the final norm
     "head_row": 0,              # its row in its type's weight bank
     "adapter": null,            # an adapter slot run at the marker; null = base
     "output": {"mode": "select", "token_ids": [9891, 2201], "weight": 1.0},
     "settings": {}}             # type-specific (validated by the type)

A LoRA slot's entry is ``{"type": "lora"}``. The control token, the substitute
token and the chat-template placement stay where they are (the ``adapter_*``
config fields and the composer); a stream entry only says what runs.

An entry may carry only the fields listed in ``HEAD_FIELDS`` and
``OUTPUT_FIELDS``. A field this version doesn't know is refused, not skipped:
it may come from a newer Granite Switch, and running the slot without it would
quietly compute something else.

A checkpoint composed before streams has no ``streams`` field. Its entries are
rebuilt from the probe fields it does carry (``adapter_kinds``,
``probe_label_token_ids``, ``probe_read_layers``) with the head row equal to
the slot position, the layout those checkpoints were written in, so they load
unchanged.
"""

from dataclasses import dataclass, field

from .outputs import OUTPUT_MODES

#: The built-in adapter stream: the decoder layers' LoRA banks, no head module.
LORA = "lora"

#: The fields a head entry, and its ``output``, may carry.
HEAD_FIELDS = ("type", "read_layer", "head_row", "adapter", "output", "settings")
OUTPUT_FIELDS = ("mode", "token_ids", "weight")


def _refuse_unknown(where: str, entry: dict, known: tuple[str, ...]) -> None:
    unknown = sorted(set(entry) - set(known))
    if unknown:
        raise ValueError(
            f"{where} has fields this version of Granite Switch doesn't know: "
            f"{unknown}. Known: {list(known)}. A newer version may have "
            f"composed this checkpoint; upgrade Granite Switch to load it."
        )


@dataclass(frozen=True)
class StreamSpec:
    """One slot's stream, normalized from its config entry."""

    slot: int  # 1-based adapter index (0 is base)
    type: str
    read_layer: int | None = None
    head_row: int | None = None
    adapter: str | None = None  # an adapter slot's name, run at this marker
    mode: str = "select"
    token_ids: tuple[int, ...] = ()
    weight: float = 1.0
    settings: dict = field(default_factory=dict)

    @property
    def is_head(self) -> bool:
        return self.type != LORA

    @property
    def num_outputs(self) -> int:
        return len(self.token_ids)


def _legacy_entries(config) -> list[dict]:
    """Stream entries for a checkpoint composed before ``streams`` existed."""
    kinds = config.adapter_kinds
    labels = config.probe_label_token_ids or [None] * len(kinds)
    layers = config.probe_read_layers or [None] * len(kinds)
    entries = []
    for slot, kind in enumerate(kinds):
        if kind == LORA:
            entries.append({"type": LORA})
        else:
            entries.append(
                {
                    "type": kind,
                    "read_layer": layers[slot],
                    # The pre-streams probe bank had one row per slot.
                    "head_row": slot,
                    "output": {"mode": "select", "token_ids": list(labels[slot])},
                }
            )
    return entries


def parse_entries(entries: list[dict], num_adapters: int) -> list[StreamSpec]:
    """Validate raw entries and return one :class:`StreamSpec` per slot.

    Type-specific settings are checked by the head type (see the registry);
    this checks only what every stream shares.
    """
    if len(entries) != num_adapters:
        raise ValueError(
            f"streams length ({len(entries)}) must equal num_adapters "
            f"({num_adapters}): one entry per slot."
        )
    specs = []
    for slot, e in enumerate(entries, start=1):
        if not isinstance(e, dict) or "type" not in e:
            raise ValueError(
                f"streams[{slot - 1}] must be a dict with a type; got {e!r}."
            )
        if e["type"] == LORA:
            extra = set(e) - {"type"}
            if extra:
                raise ValueError(
                    f"streams[{slot - 1}] is a LoRA stream; it takes no {sorted(extra)}."
                )
            specs.append(StreamSpec(slot=slot, type=LORA))
            continue
        _refuse_unknown(f"streams[{slot - 1}]", e, HEAD_FIELDS)
        out = e.get("output") or {}
        _refuse_unknown(f"streams[{slot - 1}].output", out, OUTPUT_FIELDS)
        mode = out.get("mode", "select")
        if mode not in OUTPUT_MODES:
            raise ValueError(
                f"streams[{slot - 1}].output.mode must be one of {OUTPUT_MODES}; "
                f"got {mode!r}."
            )
        token_ids = tuple(out.get("token_ids") or ())
        if not token_ids or any(
            not isinstance(t, int) or isinstance(t, bool) or t < 0 for t in token_ids
        ):
            raise ValueError(
                f"streams[{slot - 1}].output.token_ids must be non-empty real token "
                f"ids; got {list(token_ids)}."
            )
        read_layer = e.get("read_layer")
        if read_layer is not None and (
            not isinstance(read_layer, int)
            or isinstance(read_layer, bool)
            or read_layer < 0
        ):
            raise ValueError(
                f"streams[{slot - 1}].read_layer must be a decoder layer (an int "
                f"from 0) or null for the final norm; got {read_layer!r}."
            )
        row = e.get("head_row")
        if not isinstance(row, int) or isinstance(row, bool) or row < 0:
            raise ValueError(
                f"streams[{slot - 1}].head_row must be a non-negative int; got {row!r}."
            )
        adapter = e.get("adapter")
        if adapter is not None and not isinstance(adapter, str):
            raise ValueError(
                f"streams[{slot - 1}].adapter must be an adapter slot's name or "
                f"null; got {adapter!r}."
            )
        specs.append(
            StreamSpec(
                slot=slot,
                type=e["type"],
                read_layer=read_layer,
                head_row=row,
                adapter=adapter,
                mode=mode,
                token_ids=token_ids,
                weight=float(out.get("weight", 1.0)),
                settings=dict(e.get("settings") or {}),
            )
        )
    return specs


def check_rows(specs: list[StreamSpec]) -> None:
    """Head rows of each type must be distinct; the bank is sized by the max."""
    by_type: dict[str, list[int]] = {}
    for s in specs:
        if s.is_head:
            by_type.setdefault(s.type, []).append(s.head_row)
    for t, rows in by_type.items():
        if len(set(rows)) != len(rows):
            raise ValueError(f"streams of type {t!r} share a head_row: {sorted(rows)}.")


def check_read_layers(specs: list[StreamSpec], config) -> None:
    """A head's read layer must be one of the model's decoder layers.

    Checked for every head type: a layer the model doesn't have is never
    reached, so the head would silently score nothing. The switch's own cache
    layers are not decoder layers a head can read.
    """
    from ..config import SWITCH_CACHE_LAYERS

    num_layers = config.num_hidden_layers - SWITCH_CACHE_LAYERS
    for s in specs:
        if s.is_head and s.read_layer is not None and s.read_layer >= num_layers:
            raise ValueError(
                f"streams[{s.slot - 1}].read_layer is {s.read_layer}, but the model "
                f"has decoder layers 0..{num_layers - 1}; use null for the final norm."
            )


def stream_specs(config) -> list[StreamSpec]:
    """The config's streams, one :class:`StreamSpec` per slot.

    Parsed on each call (a handful of entries, read at load time only). The
    records are not cached on the config, so they are never saved with it.
    """
    entries = config.streams if config.streams is not None else _legacy_entries(config)
    specs = parse_entries(entries, config.num_adapters) if config.num_adapters else []
    check_rows(specs)
    check_read_layers(specs, config)
    return specs
