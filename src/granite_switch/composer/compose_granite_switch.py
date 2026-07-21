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

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from granite_switch.composer.arch import resolve_arch
from granite_switch.composer.compose_utils import GraniteSwitchComposer
from granite_switch.composer.adapter_discovery import (
    discover_adapters,
    discover_adapters_from_yaml,
    filter_adapters,
    is_adapter_library,
    list_available_adapters,
    list_repo_adapters_remote,
    resolve_repo_path,
)
from granite_switch.composer.tokenizer_setup import (
    add_audio_token,
    add_control_tokens,
    configure_audio_chat_template,
    configure_chat_template,
    get_alora_first_invocation_token_id,
)
from granite_switch.composer.reporting import generate_compose_report, write_build_doc


# ---------------------------------------------------------------------------
# Utility helpers (kept local — not worth a separate module)
# ---------------------------------------------------------------------------


def _load_tokenizer(model_name_or_path):
    """Load tokenizer from a Granite base model."""
    return AutoTokenizer.from_pretrained(model_name_or_path)


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
    or generation-prompt flag. Empirically verified — every realistic render
    of the Granite 4.1 template yields ``<|start_of_role|>`` (id 100264) at
    position 0. The probe renders a single minimal chat to read that
    constant out of the template.

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
            "Tokenizer has no chat_template; cannot probe the LoRA "
            "substitute token."
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


def _get_directory_size(directory):
    """Return ``(total_size in GBs, file_count)`` for *directory*."""
    if Path(directory).exists():
        total_size = 0
        file_count = 0
        for dirpath, _dirnames, filenames in os.walk(directory):
            # Prune hidden directories in-place
            # This skips folders like '.git', '.cache', etc.
            _dirnames[:] = [d for d in _dirnames if not d.startswith('.')]
            
            for filename in filenames:
                if filename.startswith('.'):
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


def _copy_io_configs(discovered_adapters, output_path):
    """Copy io.yaml files to *output_path/io_configs/<adapter_name>/*.

    Skips built-in adapters (adapter_path is None).
    """
    print("\nCopying io.yaml configuration files...")
    io_config_paths = []

    for i, adapter_info in enumerate(discovered_adapters, 1):
        adapter_path, adapter_name = adapter_info[0], adapter_info[1]
        if adapter_path is None:
            # Built-in adapter — no io.yaml to copy
            io_config_paths.append(None)
            continue
        io_config_dir = Path(output_path) / "io_configs" / adapter_name
        io_config_dir.mkdir(parents=True, exist_ok=True)
        source = Path(adapter_path) / "io.yaml"
        dest = io_config_dir / "io.yaml"
        shutil.copy2(source, dest)
        rel_path = dest.relative_to(output_path)
        io_config_paths.append(str(rel_path))
        print(f"  [{i}] {rel_path}")

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
        adapter_token_ids: List of control token IDs.
        output_path: Output directory path.
        base_model_name: Base model name/path.
        include_debug_fields: If True, include original_path in output.
    """
    print("\nCreating adapter index file...")
    model_name_only = base_model_name.split("/")[-1]

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
        token_id = adapter_token_ids[adapter_idx]

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
        entry.name: entry.stat().st_mtime
        for entry in d.iterdir()
        if entry.is_file()
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
        "--enable-audio",
        action="store_true",
        default=False,
        help="Enable the audio cascade: add the <|audio|> marker token and set "
             "asr_enabled in the config so the vLLM backend transcribes audio.",
    )
    parser.add_argument(
        "--asr-model",
        type=str,
        default=None,
        help="HF id of the speech-to-text model the audio preprocessor loads. "
             "Implies --enable-audio. Defaults to a small built-in model when unset.",
    )
    parser.add_argument(
        "--asr-device",
        type=str,
        default="cpu",
        help="Device the ASR model runs on (default: cpu). Use e.g. cuda:0 to "
             "run transcription on GPU (watch vLLM's KV-cache memory budget).",
    )
    parser.add_argument(
        "--asr-pipeline-kwargs",
        type=json.loads,
        default=None,
        help="JSON object of extra kwargs merged into the transformers ASR "
             "pipeline() construction, e.g. '{\"chunk_length_s\": 15}'. Baked "
             "into the checkpoint config. Implies --enable-audio.",
    )
    parser.add_argument(
        "--asr-generate-kwargs",
        type=json.loads,
        default=None,
        help="JSON object of default decode kwargs applied on every "
             "transcription, e.g. '{\"language\": \"de\", \"task\": "
             "\"transcribe\"}' for multilingual Whisper. Per-request "
             "mm_processor_kwargs override these. Implies --enable-audio.",
    )
    parser.add_argument(
        "--asr-max-audio-clips",
        type=int,
        default=None,
        help="Max audio clips accepted per request (default 32). Implies "
             "--enable-audio.",
    )
    parser.add_argument(
        "--asr-generation-reserve-tokens",
        type=int,
        default=None,
        help="Context tokens held back for the generated answer when sizing the "
             "per-clip transcript budget (default 8192). Implies --enable-audio.",
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
             "(for backends with a fixed input window). Implies --enable-audio.",
    )
    parser.add_argument(
        "--asr-chunk-length-s",
        type=float,
        default=None,
        help="Chunker window length in seconds (default 30.0). Only used when "
             "the backend does not self-chunk. Implies --enable-audio.",
    )
    parser.add_argument(
        "--asr-chunk-overlap-s",
        type=float,
        default=None,
        help="Chunker window overlap in seconds (default 5.0). Only used when "
             "the backend does not self-chunk. Implies --enable-audio.",
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
                    available = list_repo_adapters_remote(
                        entry, args.target_model
                    )
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
                available = list_available_adapters(
                    resolved_path, args.target_model
                )

            if not available:
                print(f"\nNo adapters found in {entry} for target '{args.target_model}'")
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
                    resolved_path, args.target_model, arch, args.technology,
                    technology_filter=args.technology_filter,
                    source=entry,
                )
                found = filter_adapters(
                    found,
                    include=args.include_adapters,
                    exclude=args.exclude_adapters,
                )
                if not found:
                    msg = f"  WARNING: No adapters found for target '{args.target_model}'"
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
    external_discovered = list(discovered_adapters)
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

    adapter_token_ids, special_tokens = add_control_tokens(tokenizer, all_discovered)

    # Audio cascade: add the <|audio|> marker token before the embedding resize.
    # Enabled by --enable-audio or by naming an --asr-model.
    audio_enabled = (
        args.enable_audio
        or args.asr_model is not None
        or args.asr_pipeline_kwargs is not None
        or args.asr_generate_kwargs is not None
        or args.asr_max_audio_clips is not None
        or args.asr_generation_reserve_tokens is not None
        or args.asr_self_chunks is not None
        or args.asr_chunk_length_s is not None
        or args.asr_chunk_overlap_s is not None
    )
    audio_token_id = add_audio_token(tokenizer) if audio_enabled else None

    # Configure chat template with adapter mappings (Granite models only).
    # Non-Granite models preserve the upstream template verbatim because
    # the injection targets Granite-specific Jinja patterns.
    normalized_type = getattr(base_config, "model_type", "").replace("_switch", "")
    if normalized_type.startswith("granite"):
        configure_chat_template(tokenizer, all_discovered)
        if audio_enabled:
            # Make the chat template emit <|audio|> for audio content parts so
            # the OpenAI server / chat() path works (the ASR processor replaces it).
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

    # Token-exchange substitute choice (must mirror the token that appears
    # right after the control token in the rendered chat prompt, so the
    # swap keeps the residual stream in-distribution):
    #   - ALoRA: first token of the adapter's alora_invocation_tokens.
    #   - LoRA/builtin: whatever the tokenizer's chat template emits at
    #     the very start of a no-adapter user turn. For Granite 4.x that's
    #     <|start_of_role|>; the probe derives this from the template at
    #     compose time so other base models work by construction.
    lora_sub_id = _probe_lora_substitute_token_id(tokenizer)
    adapter_substitute_token_ids = []
    for adapter_path, _name, technology, _source in all_discovered:
        if technology == "alora":
            sub_id = get_alora_first_invocation_token_id(adapter_path)
        else:
            sub_id = lora_sub_id
        adapter_substitute_token_ids.append(sub_id)

    model = GraniteSwitchComposer.from_base_and_adapters(
        base_model_name_or_path=base_model_local_path,
        adapter_paths=adapter_paths,
        adapter_token_ids=adapter_token_ids,
        adapter_substitute_token_ids=adapter_substitute_token_ids,
        adapter_names=adapter_names,
        built_in_adapter_names=built_in_names,
        built_in_lora_rank=args.lora_rank,
        built_in_lora_alpha=args.lora_alpha if args.lora_alpha is not None else float(args.lora_rank),
        **optional_kwargs,
    )

    # Record audio-cascade settings in the config so the checkpoint is
    # self-describing and the vLLM backend gates audio on asr_enabled.
    if audio_enabled:
        model.config.asr_enabled = True
        model.config.asr_model_id = args.asr_model
        model.config.asr_device = args.asr_device
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
        if args.asr_generation_reserve_tokens is not None:
            model.config.asr_generation_reserve_tokens = (
                args.asr_generation_reserve_tokens
            )
        if args.asr_self_chunks is not None:
            model.config.asr_self_chunks = args.asr_self_chunks
        if args.asr_chunk_length_s is not None:
            model.config.asr_chunk_length_s = args.asr_chunk_length_s
        if args.asr_chunk_overlap_s is not None:
            model.config.asr_chunk_overlap_s = args.asr_chunk_overlap_s
        print(
            f"  Audio cascade enabled "
            f"(asr_model_id={args.asr_model or 'default'}, "
            f"asr_device={args.asr_device}, audio_token_id={audio_token_id}, "
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

    print(f"\nStep 3 complete in {time.time() - step_start:.2f}s")

    return (
        model, tokenizer, args, base_model_local_path, base_model_size_gb,
        adapter_paths, all_discovered, adapter_token_ids,
        start_time, new_vocab_size, original_vocab_size,
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
    io_config_paths = _copy_io_configs(all_discovered, args.output)
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
            print(f"  Size increase: +{size_increase_gb:.3f} GB (+{size_increase_pct:.1f}%)")
        else:
            print(f"  Size difference: {size_increase_gb:.3f} GB ({size_increase_pct:.1f}%)")
    
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
    print(f"\nAdapter summary:")
    for i, adapter_info in enumerate(adapter_index["adapters"], 1):
        adapter_name = adapter_info["adapter_name"]
        ctrl = adapter_info["control_token"]
        io_config = adapter_info.get("io_config")
        print(f"  [{i}] {adapter_name}")
        print(f"      Control: {ctrl['token']} (ID {ctrl['id']})")
        if io_config:
            print(f"      Config: {io_config}")
        if adapter_info.get("built_in"):
            print(f"      (built-in adapter)")
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
        model, tokenizer, args, base_model_local_path, base_model_size_gb,
        adapter_paths, all_discovered, adapter_token_ids,
        start_time, new_vocab_size, original_vocab_size,
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
