#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Compose a Granite Switch model with embedded LoRA adapters.

Adapters can be provided as HuggingFace repo IDs, local paths, or built-in
(empty LoRA) slots.

Examples:
  # Build with all adapters from all libraries
  python compose_granite_switch.py \\
      --adapters ibm-granite/granitelib-rag-r1.0 \\
                 ibm-granite/granitelib-core-r1.0 \\
                 ibm-granite/granitelib-guardian-r1.0

  # Built-in adapter slots only
  python compose_granite_switch.py --built-in-adapters base

  # With a return-to-base control token (<|base_reset|>), so one request can
  # route base -> adapter -> base -> another adapter
  python compose_granite_switch.py \\
      --adapters ibm-granite/granitelib-core-r1.0 \\
      --base-reset-token

  # Include only specific adapters from a library
  python compose_granite_switch.py \\
      --adapters ibm-granite/granitelib-rag-r1.0 \\
      --include-adapters answerability citations

  # Exclude a specific adapter
  python compose_granite_switch.py \\
      --adapters ibm-granite/granitelib-guardian-r1.0 \\
      --exclude-adapters factuality-detection

  # Only lora adapters (not alora)
  python compose_granite_switch.py \\
      --adapters ibm-granite/granitelib-rag-r1.0 \\
      --technology-filter lora

  # List available adapters without building
  python compose_granite_switch.py \\
      --adapters ibm-granite/granitelib-rag-r1.0 \\
      --list-adapters

"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from granite_switch.composer.adapter_discovery import (
    discover_adapters,
    discover_adapters_from_yaml,
    filter_adapters,
    is_adapter_library,
    list_available_adapters,
    list_repo_adapters_remote,
    resolve_repo_path,
)
from granite_switch.composer.adapter_loader import is_shadow_residual_adapter
from granite_switch.composer.arch import resolve_arch
from granite_switch.composer.compose_utils import GraniteSwitchComposer
from granite_switch.composer.reporting import generate_compose_report, write_build_doc
from granite_switch.composer.tokenizer_setup import (
    ANCHOR_MODE_SR,
    add_audio_token,
    add_control_tokens,
    build_substitute_token_ids,
    configure_audio_chat_template,
    configure_chat_template,
    find_reserved_never_emitted_token_id,
    load_activation_anchor,
)
from granite_switch.composer.validator import validate_control_lut
from granite_switch.composer.weight_transfer import validate_untied_lm_head_saved
from granite_switch.config import ASR_DTYPES
from granite_switch.token_exchange import rebuild_control_to_substitute_lut

# ---------------------------------------------------------------------------
# Utility helpers (kept local — not worth a separate module)
# ---------------------------------------------------------------------------


def _load_tokenizer(model_name_or_path):
    """Load tokenizer from a Granite base model."""
    return AutoTokenizer.from_pretrained(model_name_or_path)


def build_control_token_lists(tokenizer, all_discovered, base_reset):
    """Add the control tokens and build their index-aligned substitute ids.

    Kept as one function because the two lists must agree in length and order:
    ``GraniteSwitchConfig`` validates the lengths and the token-exchange LUT zips
    them, so growing one without the other shifts every adapter's substitute by
    one.

    Args:
        tokenizer: HuggingFace tokenizer (mutated: control tokens are added).
        all_discovered: ``(adapter_path, adapter_name, technology, source)`` tuples.
        base_reset: Whether to also emit the ``<|base_reset|>`` control token.

    Returns:
        ``(adapter_token_ids, special_tokens, adapter_substitute_token_ids)``
    """
    adapter_token_ids, special_tokens = add_control_tokens(
        tokenizer, all_discovered, base_reset=base_reset
    )
    # Token-exchange substitute choice (must mirror the token that appears right
    # after the control token in the rendered chat prompt, so the swap keeps the
    # residual stream in-distribution) — see build_substitute_token_ids.
    adapter_substitute_token_ids = build_substitute_token_ids(
        all_discovered,
        _probe_lora_substitute_token_id(tokenizer),
        base_reset=base_reset,
    )
    return adapter_token_ids, special_tokens, adapter_substitute_token_ids


def _probe_lora_substitute_token_id(tokenizer) -> int:
    """Ask the tokenizer which token naturally appears at sequence position 0
    of a rendered no-adapter chat.

    The LoRA prefix insertion places the adapter control token at sequence
    position 0 of the rendered output. Whatever token would otherwise have
    occupied position 0 (in a no-adapter render) is the right substitute
    whose embedding should land at the swap site so the post-swap sequence
    is indistinguishable from a no-adapter render.

    Assumption (Granite 4.x): the chat template emits a constant
    ``input_ids[0]`` regardless of message content, system prompt presence,
    or generation-prompt flag. Empirically verified for both template
    families — every realistic render of the Granite 4.1 role-marker template
    yields ``<|start_of_role|>`` (id 100264) at position 0, and every render of
    the Granite 4.2 ChatML template (thinking on/off, with/without system
    prompt) yields ``<|im_start|>`` (id 100256) at position 0. The probe
    renders a single minimal chat to read that constant out of the template.

    A future model whose chat template branches on inputs at position 0
    (e.g. emits BOS only when no system message is present) would break
    this assumption: the probe would still return *some* valid id, but it
    might not match position 0 in another render mode at runtime, leaving
    the LoRA control token swapped to an embedding the model doesn't
    expect at that position. ``tests/composer/test_lora_substitute_probe.py``
    pins the Granite 4.x behavior; if you port to another base model with
    a more dynamic template, extend the probe to render multiple shapes
    and verify they all agree.

    By deriving the substitute from the tokenizer's own chat template at
    compose time we avoid hard-coding a Granite-specific token string.

    Raises ``ValueError`` if the template is missing, fails to render, or
    emits an unknown token.
    """
    if tokenizer.chat_template is None:
        raise ValueError(
            "Tokenizer has no chat_template; cannot probe the LoRA substitute token."
        )
    try:
        probe_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": "probe"}],
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception as e:
        raise ValueError(
            "Failed to render a probe chat via tokenizer.apply_chat_template "
            f"while detecting the LoRA substitute token: {e!r}."
        ) from e
    ids = tokenizer(probe_text, add_special_tokens=False).input_ids
    if not ids:
        raise ValueError(
            "Probe chat tokenized to an empty id list; cannot determine the "
            "LoRA substitute token."
        )
    sub_id = ids[0]
    if sub_id == tokenizer.unk_token_id:
        raise ValueError(
            "First token of the rendered probe chat is <unk>; the template "
            "appears to emit content outside the tokenizer's vocabulary."
        )
    return sub_id


def initialize_control_token_output_rows(model, token_ids, reserved_token_id):
    """Point every control token's output row at a reserved never-emitted row.

    ``resize_token_embeddings`` appends a row per control token, and since
    transformers 4.46 that row is *mean-initialized* — sampled from a Gaussian
    fitted to the trained rows, so it lands squarely inside the distribution of
    real tokens rather than looking like noise. Nothing suppresses control tokens
    at generation time (no ``bad_words``, no logit bias, ``bias=False`` on the
    head in both backends), so an uncorrected row gives each control token an
    arbitrary — and compose-run-nondeterministic — probability of being emitted.

    Policy: copy a reserved ``<|unused_N|>`` row (see
    :func:`~granite_switch.composer.tokenizer_setup.find_reserved_never_emitted_token_id`)
    into every control token's row, making each control token's logit identical
    to a token the base model was trained not to emit, for every possible hidden
    state. All control tokens share one reserved row: they are all equally
    never-emitted, so there is nothing to distinguish, and sharing keeps this
    consistent with how other added placeholder tokens are initialized.

    This replaces an earlier policy that copied each control token's
    token-exchange *substitute* row. That made a control token's output detector
    identical to its substitute's, so at any position where the substitute was
    the natural next token the two had equal logits and split the probability
    mass — for LoRA the substitute is the sequence-start role marker, for ALoRA
    the first invocation token, often something as common as ``<``.

    Applies on the **tied** path (Granite 4.0/4.1) as well as the untied one
    (4.2), even though the write then lands in the matrix shared with the input
    embedding. That is safe because a control token's input row is never read:
    the switch rewrites each control-token id to its substitute id *before* the
    embedding lookup, in both backends. Every path goes through the shared
    :func:`granite_switch.token_exchange.apply_token_exchange`: the HF and vLLM
    switches call it in ``forward`` and return ``modified_input_ids`` for the text
    path, and the vLLM model's ``embed_input_ids`` calls it against the switch's
    buffer for the multimodal path (where vLLM precomputes ``inputs_embeds``
    before ``forward`` runs).
    ``tests/vllm/_model_forward_tests.py::TestKVVisibility`` pins this on the
    vLLM side by perturbing a control token's embedding row and asserting the
    logits after it are unchanged, and
    ``tests/composer/test_control_token_output_rows.py`` does the same on CPU
    for the HF backend. Being tied is therefore not a constraint here — the shared row
    is output-only in practice, which is what makes the 4.1 fix possible at all.

    The ``<|audio|>`` marker goes through here too, for the same reason and with
    one extra consequence. The marker has no token-exchange substitute to copy
    from — it stands for a variable-length transcript, not for one token — so a
    reserved row is the only sensible source. And an emitted marker is worse than
    an emitted control token: if the reply is fed back on a later turn, the
    processor's ``_validate_marker_count`` rejects the whole request, because
    markers no longer match audio items. The tied-path argument above holds for it
    as well: the marker is replaced by the transcript's token ids before the
    decoder runs, and a marker with no matching audio item is rejected up front,
    so its input row is never read either.

    Args:
        model: A ``GraniteSwitchForCausalLM`` after ``resize_token_embeddings``.
        token_ids: Every newly added, never-trained id whose output row needs
            repointing. Control tokens — one per adapter, plus the
            ``<|base_reset|>`` slot at index 0 when MultiSwitch was composed with
            ``--base-reset-token`` — and the ``<|audio|>`` marker when audio is
            enabled. The base-reset token is newly added and never trained just
            like the rest, so it needs the same fixup and gets it by being in this
            list.
        reserved_token_id: Token id of the reserved row to copy from.
    """
    output_weight = model.get_output_embeddings().weight
    with torch.no_grad():
        for token_id in token_ids:
            output_weight[token_id].copy_(output_weight[reserved_token_id])
    # Read the norm back off a row that was WRITTEN, not the source row: the two
    # are equal by construction, so reporting the source verifies nothing.
    written_norm = output_weight[token_ids[0]].norm() if token_ids else float("nan")
    print(
        f"  Initialized {len(token_ids)} never-trained output row(s) "
        f"from reserved token id {reserved_token_id} "
        f"(row norm {written_norm:.4f})"
    )


def refresh_switch_control_lut(model) -> bool:
    """Re-derive the switch's control->substitute table to match the shipped config.

    The switch sized its table from the pre-resize ``config.vocab_size`` (copied
    from the base model), so anything that grows the vocabulary past the last
    control id leaves it short of the config this checkpoint will ship with.

    In practice that means the audio marker. The table is sized
    ``max(base_vocab_size, max_ctrl_id + 1)`` and control ids are appended, so
    after N control tokens it is exactly ``len(tokenizer)`` and agrees already.
    ``<|audio|>`` adds one more token that is NOT a control token, pushing
    ``len(tokenizer)`` one past the last control id — so a stale table is
    specific to ``--enable-audio`` rather than universal.

    Extracted from ``build()`` so the rebuild can be tested over every switch
    engine without composing a real checkpoint. It used to be an inline
    ``switch.rebuild_control_to_substitute_lut(...)`` method call, which existed
    only on ``SingleSwitch``, so ``--switch-type multi --enable-audio`` raised
    ``AttributeError`` here. Every real multi-compose test is gated behind
    ``GRANITE_SWITCH_E2E_MODELS=1`` (unset in CI) and none of them enables audio,
    so nothing caught it.

    Args:
        model: Composed model, after ``resize_token_embeddings``.

    Returns:
        True if the table was rebuilt, False if it already agreed or there is
        no switch / no token-exchange mapping.
    """
    switch = getattr(getattr(model, "model", None), "switch", None)
    if switch is None:
        return False
    lut = getattr(switch, "control_to_substitute_lut", None)
    if lut is None or lut.numel() == model.config.vocab_size:
        return False

    old_lut_size = lut.numel()
    rebuild_control_to_substitute_lut(switch, model.config)
    print(
        "Switch control LUT rebuilt: "
        f"{old_lut_size} -> {switch.control_to_substitute_lut.numel()}"
    )
    return True


def _get_directory_size(directory):
    """Return ``(total_size in GBs, file_count)`` for *directory*."""
    if Path(directory).exists():
        total_size = 0
        file_count = 0
        for dirpath, _dirnames, filenames in os.walk(directory):
            # Prune hidden directories in-place
            # This skips folders like '.git', '.cache', etc.
            _dirnames[:] = [d for d in _dirnames if not d.startswith(".")]

            for filename in filenames:
                if filename.startswith("."):
                    continue

                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                    file_count += 1
                except OSError:
                    pass

        gb_size = total_size / (1024**3)
        return gb_size, file_count
    return None, None


def _extract_hf_snapshot_commit(adapter_path):
    """Return the full 40-char commit SHA from a HuggingFace snapshot path.

    HuggingFace's ``snapshot_download`` stores adapters under
    ``<HF_HUB_CACHE>/models--<org>--<repo>/snapshots/<sha>/...``. When the
    adapter was resolved from the Hub, the SHA is baked into the path.

    Returns ``None`` when the adapter is not a HuggingFace snapshot —
    including local-path adapters, YAML-declared adapters pointing to
    arbitrary locations, and built-in slots (``adapter_path is None``). The
    gate is containment under :data:`huggingface_hub.constants.HF_HUB_CACHE`,
    which rules out paths that happen to contain a ``snapshots/<40 hex>``
    segment by coincidence.
    """
    if not adapter_path:
        return None

    from huggingface_hub.constants import HF_HUB_CACHE

    path = Path(adapter_path).resolve()
    try:
        path.relative_to(Path(HF_HUB_CACHE).resolve())
    except ValueError:
        return None

    parts = path.parts
    if "snapshots" in parts:
        idx = parts.index("snapshots")
        if idx + 1 < len(parts):
            sha = parts[idx + 1]
            if len(sha) == 40 and all(c in "0123456789abcdef" for c in sha):
                return sha
    return None


# A minimal io.yaml synthesized when --create-ioyaml is set and an adapter
# ships none. All fields null; the caller/consumer can fill them in later.
_MINIMAL_IO_YAML = "name: ~\nmodel: ~\nresponse_format: ~\ntransformations: ~\n"


def _copy_io_configs(discovered_adapters, output_path, create_ioyaml=False):
    """Copy each adapter's io.yaml to *output_path/io_configs/<adapter_name>/*.

    Skips built-in adapters (adapter_path is None).

    When an external adapter has no io.yaml:
      * ``create_ioyaml=False`` (default): raise ``FileNotFoundError`` — the
        adapter is expected to ship an io.yaml.
      * ``create_ioyaml=True``: synthesize a minimal io.yaml (all fields null)
        on the fly so composition can proceed.
    """
    print("\nCopying io.yaml configuration files...")
    io_config_paths = []

    for i, adapter_info in enumerate(discovered_adapters, 1):
        adapter_path, adapter_name = adapter_info[0], adapter_info[1]
        if adapter_path is None:
            # Built-in adapter — no io.yaml to copy
            io_config_paths.append(None)
            continue
        source = Path(adapter_path) / "io.yaml"
        io_config_dir = Path(output_path) / "io_configs" / adapter_name
        dest = io_config_dir / "io.yaml"
        if source.is_file():
            io_config_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            print(f"  [{i}] {dest.relative_to(output_path)}")
        elif create_ioyaml:
            io_config_dir.mkdir(parents=True, exist_ok=True)
            dest.write_text(_MINIMAL_IO_YAML)
            print(f"  [{i}] {dest.relative_to(output_path)} (synthesized minimal)")
        else:
            raise FileNotFoundError(
                f"Adapter '{adapter_name}' has no io.yaml at {source}. "
                f"Pass --create-ioyaml to synthesize a minimal io.yaml, or add "
                f"an io.yaml to the adapter."
            )
        io_config_paths.append(str(dest.relative_to(output_path)))

    copied = sum(1 for p in io_config_paths if p is not None)
    print(f"Copied {copied} io.yaml file(s)")
    return io_config_paths


def _create_adapter_index(
    discovered_adapters,
    io_config_paths,
    adapter_token_ids,
    output_path,
    base_model_name,
    include_debug_fields=False,
):
    """Create ``adapter_index.json``.

    Args:
        discovered_adapters: List of (path, name, technology, source) tuples.
        io_config_paths: List of io.yaml relative paths.
        adapter_token_ids: Control token IDs, in control-token order: one per
            adapter, optionally preceded by the base-reset slot.
        output_path: Output directory path.
        base_model_name: Base model name/path.
        include_debug_fields: If True, include original_path in output.
    """
    print("\nCreating adapter index file...")
    model_name_only = base_model_name.split("/")[-1]

    # adapter_token_ids may be one longer than discovered_adapters: composing with
    # --base-reset-token puts <|base_reset|> in the LEADING slot
    # (tokenizer_setup.add_control_tokens), and that slot is not an adapter. The
    # lookup below is per-adapter, so it has to skip the offset -- otherwise every
    # adapter records its predecessor's id and the last adapter's id is never
    # written, with no IndexError to reveal it.
    token_id_offset = len(adapter_token_ids) - len(discovered_adapters)
    if token_id_offset not in (0, 1):
        raise ValueError(
            f"adapter_token_ids has {len(adapter_token_ids)} entries for "
            f"{len(discovered_adapters)} adapter(s); expected the same count, or one "
            f"more when a leading base-reset slot is present. The per-adapter "
            f"control-token ids recorded in adapter_index.json cannot be aligned."
        )

    index = {
        "model_info": {
            "num_adapters": len(discovered_adapters),
            "base_model": model_name_only,
        },
        "adapters": [],
    }

    for adapter_idx, (adapter_info, io_config_path) in enumerate(
        zip(discovered_adapters, io_config_paths)
    ):
        adapter_path, adapter_name, technology = adapter_info[:3]
        source = adapter_info[3] if len(adapter_info) > 3 else None
        token_id = adapter_token_ids[adapter_idx + token_id_offset]

        entry = {
            "adapter_index": adapter_idx + 1,
            "adapter_name": adapter_name,
            "technology": technology,
            "control_token": {
                "token": f"<|{adapter_name}|>",
                "id": token_id,
            },
        }

        if adapter_path is not None:
            if include_debug_fields:
                # Use source (HF repo ID or local path) if available
                original = f"{source}/{adapter_name}" if source else adapter_path
                entry["original_path"] = original
            entry["io_config"] = io_config_path
        else:
            entry["built_in"] = True

        index["adapters"].append(entry)

    index_path = Path(output_path) / "adapter_index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print("Adapter index saved to: adapter_index.json")
    return index


def _resolve_base_model_path(base_model_name_or_path):
    """Resolve a base model identifier to a local directory path.

    If *base_model_name_or_path* is already a local directory, return the
    resolved absolute path.  Otherwise treat it as a HuggingFace Hub repo ID
    and download via ``snapshot_download``.
    """
    local = Path(base_model_name_or_path)
    if local.is_dir():
        resolved = str(local.resolve())
        print(f"Base model resolved to local path: {resolved}")
        return resolved
    print(f"Downloading base model from HuggingFace Hub: {base_model_name_or_path}")
    resolved = snapshot_download(repo_id=base_model_name_or_path, repo_type="model")
    print(f"Base model downloaded to: {resolved}")
    return resolved


# Files that are definitively wrong to copy from the upstream base model.
_UPSTREAM_EXCLUDE_PATTERNS = {
    # Weight files (replaced by save_pretrained)
    ".safetensors",
    ".bin",
    ".pt",
    ".ckpt",
    # Signature files specific to the original checkpoint
    ".sig",
}
_UPSTREAM_EXCLUDE_NAMES = {
    # Replaced with GraniteSwitchConfig
    "config.json",
    # Weight index files
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
    # Upstream README is replaced with a compose-specific BUILD.md rendered
    # by write_model_card.
    "README.md",
}


def _copy_upstream_auxiliary_files(base_model_local_path, output_path):
    """Copy non-weight files from the upstream base model to *output_path*.

    Uses a minimal exclusion list — only files that are definitively wrong to
    copy are skipped.  Everything else (``generation_config.json``,
    ``chat_template.jinja``, ``LICENSE``, etc.) is copied so that the build
    output is deployment-complete.  ``README.md`` is excluded because the
    composer writes its own compose-specific ``BUILD.md`` via
    ``write_model_card``.  ``save_pretrained()`` will overwrite files it
    manages (tokenizer, config).
    """
    src = Path(base_model_local_path)
    dst = Path(output_path)
    dst.mkdir(parents=True, exist_ok=True)

    copied = []
    skipped = []

    for entry in sorted(src.iterdir()):
        # Only top-level files
        if not entry.is_file():
            continue
        name = entry.name

        # Skip dotfiles (HF cache metadata like .gitattributes)
        if name.startswith("."):
            skipped.append(name)
            continue

        # Skip excluded extensions
        if any(name.endswith(ext) for ext in _UPSTREAM_EXCLUDE_PATTERNS):
            skipped.append(name)
            continue

        # Skip excluded filenames
        if name in _UPSTREAM_EXCLUDE_NAMES:
            skipped.append(name)
            continue

        # Skip if already present in output (from prior build steps)
        dest_file = dst / name
        if dest_file.exists():
            skipped.append(f"{name} (already exists)")
            continue

        shutil.copy2(str(entry), str(dest_file))
        copied.append(name)

    if copied:
        print(f"  Copied {len(copied)} upstream file(s):")
        for name in copied:
            print(f"    {name}")
    if skipped:
        print(f"  Skipped {len(skipped)} file(s): {', '.join(skipped)}")

    return copied


# Files that save_pretrained() is expected to write or overwrite.
_EXPECTED_SAVE_FILES = {
    # model.save_pretrained
    "config.json",
    "generation_config.json",
    # tokenizer.save_pretrained
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "tokenizer.model",
    "merges.txt",
    "vocab.json",
    "vocab.txt",
    "chat_template.jinja",
}


def _snapshot_directory(directory):
    """Return ``{filename: mtime}`` for top-level files in *directory*."""
    d = Path(directory)
    return {
        entry.name: entry.stat().st_mtime for entry in d.iterdir() if entry.is_file()
    }


def _validate_save_pretrained_writes(before, after, output_path):
    """Compare before/after directory snapshots and report what changed.

    Prints new files, overwritten files, and warnings for unexpected writes.
    """
    new_files = []
    overwritten_files = []

    for name, mtime in sorted(after.items()):
        if name not in before:
            new_files.append(name)
        elif mtime != before[name]:
            overwritten_files.append(name)

    if new_files:
        print(f"  New files from save_pretrained ({len(new_files)}):")
        for name in new_files:
            print(f"    + {name}")

    if overwritten_files:
        print(f"  Overwritten upstream files ({len(overwritten_files)}):")
        for name in overwritten_files:
            print(f"    ~ {name}")

    # Check for unexpected writes
    for name in new_files + overwritten_files:
        is_expected = (
            name in _EXPECTED_SAVE_FILES
            or name.endswith(".safetensors")
            or name == "model.safetensors.index.json"
        )
        if not is_expected:
            print(f"  WARNING: unexpected file written by save_pretrained: {name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _compose_argparser():
    parser = argparse.ArgumentParser(
        description="Compose Granite Switch model with embedded LoRA adapters",
        epilog="""
Examples:
  python compose_granite_switch.py --adapters ibm-granite/granitelib-rag-r1.0
  python compose_granite_switch.py --adapters ibm-granite/granitelib-rag-r1.0 ibm-granite/granitelib-core-r1.0
  python compose_granite_switch.py --built-in-adapters base
  python compose_granite_switch.py --adapters ibm-granite/granitelib-rag-r1.0 --include-adapters answerability citations
  python compose_granite_switch.py --adapters ibm-granite/granitelib-guardian-r1.0 --exclude-adapters factuality-detection
  python compose_granite_switch.py --adapters ibm-granite/granitelib-rag-r1.0 --technology-filter lora
  python compose_granite_switch.py --adapters ibm-granite/granitelib-rag-r1.0 --list-adapters
  python compose_granite_switch.py --adapters ibm-granite/granitelib-core-r1.0 --base-reset-token
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--adapters",
        type=str,
        nargs="*",
        default=[],
        help="Adapter HuggingFace repo IDs or local paths, or YAML manifests",
    )
    parser.add_argument(
        "--technology",
        type=str,
        default=None,
        choices=["alora", "lora"],
        help="Adapter technology (default: auto-detect from path, fallback alora)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="ibm-granite/granite-4.1-3b",
        help="Path or HF repo for base Granite model",
    )
    parser.add_argument(
        "--target-model",
        type=str,
        default=None,
        help="Target model name for adapter discovery (default: from --base-model)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./granite-with-all-aloras",
        help="Output directory",
    )
    parser.add_argument(
        "--switch-head-dim",
        type=int,
        default=None,
        help="Dimension of Q/K/V vectors in switch attention",
    )
    parser.add_argument(
        "--ms-code-m",
        type=int,
        default=None,
        choices=[6, 8],
        help="multi (coded) only: Kerdock/DG parameter m (N = 2^m). Default 6.",
    )
    parser.add_argument(
        "--ms-memory-gain",
        type=float,
        default=None,
        help="multi (coded) only: memory-head key scaling. Default 28.0.",
    )
    parser.add_argument(
        "--base-reset-token",
        action="store_true",
        help=(
            "multi only: also emit a <|base_reset|> control token that returns "
            "routing to the base model mid-sequence. Yields num_adapters + 1 "
            "control tokens with the base-reset slot first. Off by default."
        ),
    )
    parser.add_argument(
        "--built-in-adapters",
        type=str,
        nargs="*",
        default=[],
        help="Names for built-in (empty LoRA) adapter slots",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=8,
        help="LoRA rank for built-in adapters (default: 8)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=float,
        default=None,
        help="LoRA alpha for built-in adapters (default: same as --lora-rank)",
    )
    parser.add_argument(
        "--include-adapters",
        type=str,
        nargs="*",
        default=None,
        help="Only include adapters matching these names/patterns (fnmatch glob). "
        "Example: --include-adapters answerability 'query_*'",
    )
    parser.add_argument(
        "--exclude-adapters",
        type=str,
        nargs="*",
        default=None,
        help="Exclude adapters matching these names/patterns (applied after "
        "--include-adapters). "
        "Example: --exclude-adapters hallucination_detection",
    )
    parser.add_argument(
        "--technology-filter",
        type=str,
        default=None,
        choices=["alora", "lora"],
        help="Only include adapters of this technology type. "
        "Unlike --technology, this filters rather than overriding the label.",
    )
    parser.add_argument(
        "--list-adapters",
        action="store_true",
        default=False,
        help="List available adapters in the library and exit (no build).",
    )
    parser.add_argument(
        "--debug-fields",
        action="store_true",
        default=False,
        help="Include debug fields (original_path) in adapter_index.json",
    )
    parser.add_argument(
        "--create-ioyaml",
        action="store_true",
        default=False,
        help=(
            "If an adapter has no io.yaml, synthesize a minimal one "
            "(all fields null) instead of failing. Default: fail when an "
            "adapter's io.yaml is missing."
        ),
    )
    parser.add_argument(
        "--enable-audio",
        action="store_true",
        default=False,
        help="Enable the audio cascade: add the <|audio|> marker token and set "
        "asr_enabled in the config so the vLLM backend transcribes audio. This is "
        "the only flag that switches audio on — the --asr-* options configure the "
        "cascade and are ignored without it.",
    )
    parser.add_argument(
        "--asr-model",
        type=str,
        default=None,
        help="HF id of the speech-to-text model the audio preprocessor loads. "
        "Requires --enable-audio; ignored without it. Defaults to a small built-in model when unset.",
    )
    parser.add_argument(
        "--asr-device",
        type=str,
        default="cpu",
        help="Device the ASR model runs on (default: cpu). Use e.g. cuda:0 to "
        "run transcription on GPU (watch vLLM's KV-cache memory budget).",
    )
    parser.add_argument(
        "--asr-dtype",
        type=str,
        default=None,
        choices=ASR_DTYPES,
        help="Precision the ASR weights load in. Default derives it from "
        "--asr-device (float16 on CUDA, float32 on CPU); set float32 for an "
        "encoder that cannot run in half precision (e.g. one with BatchNorm). "
        "Requires --enable-audio; ignored without it.",
    )
    parser.add_argument(
        "--asr-pipeline-kwargs",
        type=json.loads,
        default=None,
        help="JSON object of extra kwargs merged into the transformers ASR "
        "pipeline() construction, e.g. '{\"chunk_length_s\": 15}'. Baked "
        "into the checkpoint config. Requires --enable-audio; ignored without it.",
    )
    parser.add_argument(
        "--asr-generate-kwargs",
        type=json.loads,
        default=None,
        help="JSON object of default decode kwargs applied on every "
        'transcription, e.g. \'{"language": "de", "task": '
        '"transcribe"}\' for multilingual Whisper. Per-request '
        "mm_processor_kwargs override these. Requires --enable-audio; ignored without it.",
    )
    parser.add_argument(
        "--asr-max-audio-clips",
        type=int,
        default=None,
        help="Max audio clips accepted per request (default 32). Implies "
        "--enable-audio.",
    )
    parser.add_argument(
        "--asr-self-chunks",
        dest="asr_self_chunks",
        action="store_true",
        default=None,
        help="Backend chunks long audio itself (Whisper default). Mutually "
        "exclusive with --asr-no-self-chunks.",
    )
    parser.add_argument(
        "--asr-no-self-chunks",
        dest="asr_self_chunks",
        action="store_false",
        help="Route long audio through the encoder-agnostic split/merge chunker "
        "(for backends with a fixed input window). Requires --enable-audio; ignored without it.",
    )
    parser.add_argument(
        "--asr-chunk-length-s",
        type=float,
        default=None,
        help="Chunker window length in seconds (default 30.0). Only used when "
        "the backend does not self-chunk. Requires --enable-audio; ignored without it.",
    )
    parser.add_argument(
        "--asr-chunk-overlap-s",
        type=float,
        default=None,
        help="Chunker window overlap in seconds (default 5.0). Only used when "
        "the backend does not self-chunk. Requires --enable-audio; ignored without it.",
    )
    return parser


def build():
    args = _compose_argparser().parse_args()

    if args.target_model is None and args.base_model:
        args.target_model = args.base_model.split("/")[-1]
        print(f"Auto-derived target model from base model: {args.target_model}")

    # ------------------------------------------------------------------ #
    # --list-adapters: preview available adapters and exit
    # ------------------------------------------------------------------ #
    if args.list_adapters:
        if not args.adapters:
            print("ERROR: --list-adapters requires --adapters")
            return 1
        for entry in args.adapters:
            # For HF repos, use metadata-only listing (no download)
            local = Path(entry)
            if "/" in entry and not local.exists():
                try:
                    available = list_repo_adapters_remote(entry, args.target_model)
                except Exception as e:
                    print(f"Failed to list adapters from {entry}: {e}")
                    return 1
            else:
                # Local path — resolve and scan
                try:
                    resolved_path = resolve_repo_path(entry)
                except Exception as e:
                    print(f"Failed to resolve {entry}: {e}")
                    return 1
                if not is_adapter_library(resolved_path):
                    print(f"\n{entry} is a single adapter, not a library.")
                    continue
                available = list_available_adapters(resolved_path, args.target_model)

            if not available:
                print(
                    f"\nNo adapters found in {entry} for target '{args.target_model}'"
                )
                continue
            max_name = max(len(a["name"]) for a in available)
            col_w = max(max_name, 4) + 2
            print(f"\nAdapters in {entry} for {args.target_model}:\n")
            print(f"  {'Name':<{col_w}}  Technologies")
            print(f"  {'-' * col_w}  {'-' * 16}")
            for a in available:
                techs = ", ".join(a["technologies"])
                print(f"  {a['name']:<{col_w}}  {techs}")
            print(f"\n{len(available)} adapter(s) found.")
        return 0

    start_time = time.time()

    print("\n" + "=" * 80)
    print("COMPOSING GRANITE SWITCH MODEL WITH EMBEDDED ADAPTERS")
    print("=" * 80)
    print(f"Base model: {args.base_model}")
    print(f"Target model for adapters: {args.target_model}")
    print(f"Output path: {args.output}")
    print()

    # Resolve base model to a local path (downloads from Hub if needed)
    base_model_local_path = _resolve_base_model_path(args.base_model)

    # Load base config early for arch resolution.
    from granite_switch.composer.arch import load_base_config

    base_config = load_base_config(base_model_local_path)
    arch = resolve_arch(base_model_local_path, base_config=base_config)

    # ------------------------------------------------------------------ #
    # Step 0: Resolve adapters
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 80)
    print("STEP 0: Resolving adapters")
    print("=" * 80)
    step_start = time.time()

    discovered_adapters = []
    if args.adapters:
        print("\n" + "=" * 80)
        print("Processing adapters")
        print("=" * 80)
        for entry in args.adapters:
            print(f"\nResolving: {entry}")
            try:
                resolved_path = resolve_repo_path(
                    entry,
                    target_model_name=args.target_model,
                    include_adapters=args.include_adapters,
                    exclude_adapters=args.exclude_adapters,
                    technology_filter=args.technology_filter,
                )
            except Exception as e:
                print(f"Failed to resolve {entry}: {e}")
                return 1

            if is_adapter_library(resolved_path):
                # Adapter library — discover individual adapters inside
                print("  Detected adapter library, scanning for adapters...")
                found = discover_adapters(
                    resolved_path,
                    args.target_model,
                    arch,
                    args.technology,
                    technology_filter=args.technology_filter,
                    source=entry,
                )
                found = filter_adapters(
                    found,
                    include=args.include_adapters,
                    exclude=args.exclude_adapters,
                )
                if not found:
                    msg = (
                        f"  WARNING: No adapters found for target '{args.target_model}'"
                    )
                    print(msg)
                discovered_adapters.extend(found)
            elif (path := Path(entry)).is_file() and path.suffix in (".yaml", ".yml"):
                found = discover_adapters_from_yaml(entry)
                discovered_adapters.extend(found)
            else:
                # Single adapter directory
                resolved = Path(resolved_path)
                dir_name = resolved.name
                if args.technology:
                    technology = args.technology
                elif dir_name in ("alora", "lora"):
                    technology = dir_name
                else:
                    technology = "alora"

                # Derive adapter name: if the directory name is a technology
                # label (alora/lora), the adapter follows the library layout
                # adapter_name/model/technology/ — use the great-grandparent.
                if dir_name in ("alora", "lora"):
                    adapter_name = resolved.parent.parent.name
                elif "/" in entry:
                    adapter_name = entry.split("/")[-1]
                else:
                    adapter_name = entry

                # 4-tuple: (path, name, technology, source)
                discovered_adapters.append(
                    (resolved_path, adapter_name, technology, entry)
                )
                print(f"  Added adapter: {adapter_name} ({technology})")

    if not discovered_adapters and not args.built_in_adapters:
        print("\nERROR: No adapters specified")
        print("Use --adapters or --built-in-adapters")
        return 1

    # Combine external + built-in adapter lists.
    # External adapters occupy slots 0..N-1, built-ins occupy N..N+M-1.
    # Tuples are 4-element: (path, name, technology, source)
    # Shadow Residual is read out of the weights, not out of the
    # adapter_name/model/technology/ path (which only ever spells alora/lora, and
    # which no SR training run produces). The label decides where the control
    # token lands, and SR has exactly one usable activation point — the anchor at
    # the end of the generation prompt — so a mislabelled SR checkpoint must not
    # be able to fall through to aLoRA or sequence-start placement.
    external_discovered = [
        (
            path,
            name,
            ANCHOR_MODE_SR if path and is_shadow_residual_adapter(path) else tech,
            source,
        )
        for path, name, tech, source in discovered_adapters
    ]
    built_in_discovered = [
        (None, name, "builtin", None) for name in (args.built_in_adapters or [])
    ]
    all_discovered = external_discovered + built_in_discovered

    has_external = len(external_discovered) > 0
    has_built_in = len(built_in_discovered) > 0

    # Mode detection (informational only — token-exchange handles both
    # native and third-party adapter builds uniformly).
    if has_built_in and not has_external:
        build_mode = "native"
    elif has_external:
        build_mode = "third-party"
    else:
        print("\nERROR: No adapters to build (should not reach here)")
        return 1

    # Extract fields from 4-tuples (path, name, tech, source)
    adapter_paths = [t[0] for t in all_discovered if t[0] is not None]
    adapter_names = [t[1] for t in all_discovered]
    built_in_names = [name for name in (args.built_in_adapters or [])]

    print(f"\nBuild mode: {build_mode}")
    if has_external:
        print(f"  External adapters: {len(external_discovered)}")
    if has_built_in:
        print(f"  Built-in adapters: {len(built_in_discovered)}")

    print(f"\nStep 0 complete in {time.time() - step_start:.2f}s")

    # ------------------------------------------------------------------ #
    # Step 1: Tokenizer + control tokens
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 80)
    print("STEP 1: Loading tokenizer and adding special tokens")
    print("=" * 80)
    step_start = time.time()
    tokenizer = _load_tokenizer(base_model_local_path)
    original_vocab_size = len(tokenizer)
    print(f"Original vocabulary size: {original_vocab_size}")

    # Control tokens + their token-exchange substitutes, built together so the two
    # lists cannot drift apart. The substitute probe renders a NO-ADAPTER chat, and
    # a no-adapter render is byte-identical before and after the template injection
    # below (asserted by test_chat_template.py::test_no_adapter_render_unchanged),
    # so probing here rather than after configure_chat_template is equivalent.
    (
        adapter_token_ids,
        # Not unused on this branch: add_audio_token re-passes these so the
        # marker call does not evict them from additional_special_tokens.
        special_tokens,
        adapter_substitute_token_ids,
    ) = build_control_token_lists(tokenizer, all_discovered, args.base_reset_token)

    # Audio cascade: add the <|audio|> marker token before the embedding resize.
    #
    # --enable-audio is the ONLY thing that turns audio on. The other --asr-*
    # flags configure the cascade; they never enable it. Setting one without
    # --enable-audio composes a text-only checkpoint and the value is simply not
    # written — deliberate, so that one explicit flag decides, rather than the
    # decision being inferred from a set of options.
    #
    # This replaced a disjunction over every --asr-* flag. Inference read well
    # until a flag was left out of it: --asr-device was, and had a non-None
    # default besides, so `--asr-device cuda:0` alone yielded a text-only
    # checkpoint with no diagnostic. A single explicit flag has no such gap, and
    # nothing to keep in step when an --asr-* option is added.
    audio_enabled = args.enable_audio
    # The control tokens are re-passed so this call doesn't drop them from the
    # tokenizer's additional-special-tokens list (add_special_tokens replaces
    # that list rather than extending it).
    audio_token_id = (
        add_audio_token(tokenizer, keep_special_tokens=special_tokens)
        if audio_enabled
        else None
    )

    # Configure chat template with adapter mappings (Granite models only).
    # Non-Granite models preserve the upstream template verbatim because
    # the injection targets Granite-specific Jinja patterns.
    normalized_type = getattr(base_config, "model_type", "").replace("_switch", "")
    sr_substitute_token_ids: dict[str, int] = {}
    if normalized_type.startswith("granite"):
        _fmt_name, sr_substitute_token_ids = configure_chat_template(
            tokenizer, all_discovered
        )
        if audio_enabled:
            # Make the chat template emit <|audio|> for audio content parts so
            # the OpenAI server / chat() path works (the ASR processor replaces
            # it). Must run after configure_chat_template: on ChatML the audio
            # flattening has to land ahead of the ALoRA Pass 2 block, which
            # rsplits `content` and so needs a string.
            configure_audio_chat_template(tokenizer)
    else:
        print("  Skipping chat template configuration (non-Granite model)")

    new_vocab_size = len(tokenizer)
    print(f"\nStep 1 complete in {time.time() - step_start:.2f}s")

    # ------------------------------------------------------------------ #
    # Step 2: Build model
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 80)
    print("STEP 2: Creating model with embedded LoRAs")
    print("=" * 80)
    step_start = time.time()

    print("\n  Adapters to embed:")
    for i, adapter_info in enumerate(all_discovered, 1):
        adapter_path, adapter_name, technology = adapter_info[:3]
        label = adapter_path if adapter_path else "(built-in)"
        print(f"    [{i}] {adapter_name}/{technology}")
        print(f"        {label}")
    print()

    optional_kwargs = {}
    if args.switch_head_dim is not None:
        optional_kwargs["switch_head_dim"] = args.switch_head_dim
    # Coded-engine (MultiSwitch) params. These flow through
    # from_base_and_adapters(**kwargs) -> config_kwargs -> GraniteSwitchConfig,
    # so they persist to config.json and from_pretrained rebuilds the switch.
    if args.ms_code_m is not None:
        optional_kwargs["ms_code_m"] = args.ms_code_m
    if args.ms_memory_gain is not None:
        optional_kwargs["ms_memory_gain"] = args.ms_memory_gain

    # adapter_substitute_token_ids was built alongside adapter_token_ids in STEP 1
    # (see build_control_token_lists), so the two lists cannot drift apart and the
    # base-reset slot (index 0, when present) stays aligned.
    #
    # Shadow Residual is resolved HERE rather than in STEP 1 because it needs
    # configure_chat_template's answer: an SR adapter's substitute is the token the
    # control token replaced in the generation prompt -- usually its
    # last_context_token, but on a template whose branches do not all emit it
    # (Granite 4.2's enable_thinking) the resolver picks an earlier token common to
    # every branch, and that answer wins over the adapter config's. Overwriting in
    # place preserves the STEP 1 length/order invariant.
    _sub_offset = len(adapter_substitute_token_ids) - len(all_discovered)
    if _sub_offset not in (0, 1):
        raise ValueError(
            f"adapter_substitute_token_ids ({len(adapter_substitute_token_ids)}) must be "
            f"len(all_discovered) ({len(all_discovered)}) or one more (base-reset slot); "
            f"got offset {_sub_offset}."
        )
    for _i, (_path, _name, _tech, _src) in enumerate(all_discovered):
        if _tech == ANCHOR_MODE_SR:
            if _name in sr_substitute_token_ids:
                _sub = sr_substitute_token_ids[_name]
            else:
                _anchor_text, _sub, _mode = load_activation_anchor(_path, tokenizer)
            adapter_substitute_token_ids[_sub_offset + _i] = _sub

    model = GraniteSwitchComposer.from_base_and_adapters(
        base_model_name_or_path=base_model_local_path,
        adapter_paths=adapter_paths,
        adapter_token_ids=adapter_token_ids,
        adapter_substitute_token_ids=adapter_substitute_token_ids,
        adapter_names=adapter_names,
        built_in_adapter_names=built_in_names,
        built_in_lora_rank=args.lora_rank,
        built_in_lora_alpha=args.lora_alpha
        if args.lora_alpha is not None
        else float(args.lora_rank),
        **optional_kwargs,
    )

    # Record audio-cascade settings in the config so the checkpoint is
    # self-describing and the vLLM backend gates audio on asr_enabled.
    if audio_enabled:
        model.config.asr_enabled = True
        model.config.asr_model_id = args.asr_model
        model.config.asr_device = args.asr_device
        model.config.asr_dtype = args.asr_dtype
        # Optional pipeline-construction extras and default decode kwargs. Only
        # set when provided so the config stays minimal for the common case.
        if args.asr_pipeline_kwargs is not None:
            model.config.asr_pipeline_kwargs = args.asr_pipeline_kwargs
        if args.asr_generate_kwargs is not None:
            model.config.asr_generate_kwargs = args.asr_generate_kwargs
        # Long-audio / multi-clip knobs. Only set when explicitly given so the
        # config keeps the constructor defaults otherwise.
        if args.asr_max_audio_clips is not None:
            model.config.asr_max_audio_clips = args.asr_max_audio_clips
        if args.asr_self_chunks is not None:
            model.config.asr_self_chunks = args.asr_self_chunks
        if args.asr_chunk_length_s is not None:
            model.config.asr_chunk_length_s = args.asr_chunk_length_s
        if args.asr_chunk_overlap_s is not None:
            model.config.asr_chunk_overlap_s = args.asr_chunk_overlap_s
        print(
            f"  Audio cascade enabled "
            f"(asr_model_id={args.asr_model or 'default'}, "
            f"asr_device={args.asr_device}, "
            f"asr_dtype={args.asr_dtype or 'auto'}, "
            f"audio_token_id={audio_token_id}, "
            f"pipeline_kwargs={args.asr_pipeline_kwargs or {}}, "
            f"generate_kwargs={args.asr_generate_kwargs or {}}, "
            f"max_audio_clips={args.asr_max_audio_clips or 'default'}, "
            f"self_chunks={args.asr_self_chunks})"
        )

    # Base model size (best effort)
    base_model_size_gb, _ = _get_directory_size(base_model_local_path)
    if base_model_size_gb is not None:
        print(f"  Base model size: {base_model_size_gb:.2f} GB")

    print(f"\nStep 2 complete in {time.time() - step_start:.2f}s")

    # Compose report
    print("\n" + "=" * 80)
    print("Generating compose report...")
    print("=" * 80)
    if hasattr(model, "_build_mappings"):
        generate_compose_report(
            base_mapping=model._build_mappings["base"],
            adapter_mapping=model._build_mappings["adapter"],
            output_path=args.output,
            model=model,
            adapter_paths=adapter_paths,
            adapter_names=adapter_names,
            arch=arch,
            source_analysis=model._build_mappings.get("source_analysis"),
        )
        print(f"Compose report saved to {args.output}/compose_report.json")

    # ------------------------------------------------------------------ #
    # Step 3: Resize embeddings
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 80)
    print("STEP 3: Resizing model embeddings for new vocabulary")
    print("=" * 80)
    step_start = time.time()
    old_embed_size = model.model.embed_tokens.weight.shape[0]
    model.resize_token_embeddings(new_vocab_size)
    new_embed_size = model.model.embed_tokens.weight.shape[0]
    print(f"Embeddings resized: {old_embed_size} -> {new_embed_size}")

    # resize_token_embeddings appended a mean-initialized row per token added
    # above — every control token, plus the <|audio|> marker when audio is
    # enabled. None of them was ever a training target, so each is left with an
    # arbitrary, compose-run-nondeterministic emission probability (nothing
    # suppresses them at generation time). Repoint every such row at one reserved
    # never-emitted row. Runs on the tied path (4.0/4.1) as well as the untied one
    # (4.2): none of these ids has its *input* row read, so the shared-matrix
    # write is inert on the input side — see
    # initialize_control_token_output_rows for both halves of that argument.
    #
    # One list and one lookup rather than a block per token kind: the policy is
    # identical, and find_reserved_never_emitted_token_id materializes the whole
    # vocabulary, so calling it twice bought nothing.
    never_trained_token_ids = list(adapter_token_ids or [])
    if audio_token_id is not None:
        never_trained_token_ids.append(audio_token_id)

    if never_trained_token_ids:
        reserved_token_id = find_reserved_never_emitted_token_id(tokenizer)
        if reserved_token_id is None:
            # Warn rather than fail: the compose is still usable, just with these
            # rows as emittable as an average token's. Name the audio consequence
            # separately — an emitted marker does not merely look odd, it makes
            # _validate_marker_count reject a later turn outright.
            print(
                "  Warning: no reserved <|unused_N|> token in this vocabulary, so "
                f"{len(never_trained_token_ids)} newly added token(s) keep the "
                "rows resize_token_embeddings generated. Each is then as emittable "
                "as an average token, with a probability that varies between "
                "compose runs."
            )
            if audio_token_id is not None:
                print(
                    "  Warning: that includes the <|audio|> marker — a generated "
                    "marker breaks a later turn's marker/audio-item count."
                )
        else:
            initialize_control_token_output_rows(
                model, never_trained_token_ids, reserved_token_id
            )

    refresh_switch_control_lut(model)
    validate_control_lut(model)

    print(f"\nStep 3 complete in {time.time() - step_start:.2f}s")

    return (
        model,
        tokenizer,
        args,
        base_model_local_path,
        base_model_size_gb,
        adapter_paths,
        all_discovered,
        adapter_token_ids,
        start_time,
        new_vocab_size,
        original_vocab_size,
    )


def save_and_validate_model_artifacts(
    model,
    tokenizer,
    args,
    base_model_local_path,
    all_discovered,
    adapter_token_ids,
    base_model_size_gb=None,
    adapter_paths=None,
    start_time=None,
    new_vocab_size=None,
    original_vocab_size=None,
):
    # ------------------------------------------------------------------ #
    # Step 4: io.yaml + adapter index
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 80)
    print("STEP 4: Collecting io.yaml configurations")
    print("=" * 80)
    step_start = time.time()
    os.makedirs(args.output, exist_ok=True)
    io_config_paths = _copy_io_configs(
        all_discovered, args.output, create_ioyaml=args.create_ioyaml
    )
    adapter_index = _create_adapter_index(
        all_discovered,
        io_config_paths,
        adapter_token_ids,
        args.output,
        args.base_model,
        include_debug_fields=args.debug_fields,
    )
    print(f"\nStep 4 complete in {time.time() - step_start:.2f}s")
    # ------------------------------------------------------------------ #
    # Step 5: Save
    # ------------------------------------------------------------------ #

    print("\n" + "=" * 80)
    print("STEP 5: Saving model and tokenizer")
    print("=" * 80)
    step_start = time.time()
    print(f"Output directory: {args.output}")

    # Copy upstream auxiliary files first (generation_config, chat_template, etc.)
    print("\nCopying upstream auxiliary files...")
    _copy_upstream_auxiliary_files(base_model_local_path, args.output)

    # Snapshot directory state before save_pretrained
    before_snapshot = _snapshot_directory(args.output)

    model.save_pretrained(args.output, max_shard_size="5GB")
    print("Model saved")
    tokenizer.save_pretrained(args.output)
    print("Tokenizer saved")

    # Validate what save_pretrained wrote/overwrote
    after_snapshot = _snapshot_directory(args.output)
    _validate_save_pretrained_writes(before_snapshot, after_snapshot, args.output)

    # Untied bases (e.g. Granite 4.2) must keep a distinct lm_head in the
    # checkpoint; confirm it was not dropped by any tied-alias dedupe.
    if not getattr(model.config, "tie_word_embeddings", True):
        validate_untied_lm_head_saved(
            args.output,
            expected_vocab_size=model.config.vocab_size,
            hidden_size=model.config.hidden_size,
        )

    # Write compose-specific BUILD.md. The upstream README.md is excluded
    # from _copy_upstream_auxiliary_files so the composed output describes
    # itself rather than shadowing base-model documentation.
    write_build_doc(
        model=model,
        args=args,
        all_discovered=all_discovered,
        output_path=args.output,
        base_model_local_path=base_model_local_path,
        adapter_index=adapter_index,
        extract_hf_snapshot_commit=_extract_hf_snapshot_commit,
    )

    total_size_gb, file_count = _get_directory_size(args.output)

    print(f"\nStep 5 complete in {time.time() - step_start:.2f}s")
    print(f"  Total files: {file_count}")
    print(f"  Final model size: {total_size_gb:.2f} GB")

    if base_model_size_gb is not None:
        size_increase_gb = total_size_gb - base_model_size_gb
        size_increase_pct = (size_increase_gb / base_model_size_gb) * 100
        print(f"  Base model size: {base_model_size_gb:.2f} GB")
        if size_increase_gb >= 0:
            print(
                f"  Size increase: +{size_increase_gb:.3f} GB (+{size_increase_pct:.1f}%)"
            )
        else:
            print(
                f"  Size difference: {size_increase_gb:.3f} GB ({size_increase_pct:.1f}%)"
            )

    # ------------------------------------------------------------------ #
    # Final summary
    # ------------------------------------------------------------------ #
    total_time = time.time() - start_time
    num_adapters = len(adapter_paths)
    num_added = new_vocab_size - original_vocab_size

    print("\n" + "=" * 80)
    print("MODEL COMPOSITION COMPLETE!")
    print("=" * 80)
    print(f"\nTotal time: {total_time:.2f}s ({total_time / 60:.2f} minutes)")
    print(f"Output location: {args.output}")
    print(f"Vocabulary size: {new_vocab_size} (+{num_added} new tokens)")
    print(f"Number of adapters: {num_adapters}")
    print("\nAdapter summary:")
    for i, adapter_info in enumerate(adapter_index["adapters"], 1):
        adapter_name = adapter_info["adapter_name"]
        ctrl = adapter_info["control_token"]
        io_config = adapter_info.get("io_config")
        print(f"  [{i}] {adapter_name}")
        print(f"      Control: {ctrl['token']} (ID {ctrl['id']})")
        if io_config:
            print(f"      Config: {io_config}")
        if adapter_info.get("built_in"):
            print("      (built-in adapter)")
    print(f"\nAdapter index: {args.output}/adapter_index.json")
    print(f"IO configs: {args.output}/io_configs/")
    print("\n" + "=" * 80)
    print()


def main():
    result = build()

    # Early exit (e.g. --list-adapters) returns an int exit code
    if isinstance(result, int):
        return result

    (
        model,
        tokenizer,
        args,
        base_model_local_path,
        base_model_size_gb,
        adapter_paths,
        all_discovered,
        adapter_token_ids,
        start_time,
        new_vocab_size,
        original_vocab_size,
    ) = result

    save_and_validate_model_artifacts(
        model=model,
        tokenizer=tokenizer,
        args=args,
        base_model_local_path=base_model_local_path,
        all_discovered=all_discovered,
        adapter_token_ids=adapter_token_ids,
        base_model_size_gb=base_model_size_gb,
        adapter_paths=adapter_paths,
        start_time=start_time,
        new_vocab_size=new_vocab_size,
        original_vocab_size=original_vocab_size,
    )
    return 0


if __name__ == "__main__":
    exit(main())
