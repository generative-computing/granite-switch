# SPDX-License-Identifier: Apache-2.0
"""Adapter discovery and resolution utilities.

Provides helpers for resolving adapter paths (local or HuggingFace Hub),
discovering adapters within structured adapter libraries, and filtering
discovered adapters by name patterns or technology type.
"""

from fnmatch import fnmatch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from .arch import ArchDescriptor
from .adapter_loader import load_adapter_target_modules


def discover_adapters(
    root_dir: str,
    target_model_name: str,
    arch: ArchDescriptor,
    technology_fallback: Optional[str] = None,
    technology_filter: Optional[str] = None,
    source: Optional[str] = None,
) -> List[Tuple[str, str, str, Optional[str]]]:
    """Discover adapters for a target model in an adapter library directory.

    Scans *root_dir* for the ``adapter_name/model/technology/`` layout and
    returns every adapter whose *model* directory matches *target_model_name*.
    Both ``alora`` and ``lora`` technologies are recognized; when both exist
    for the same adapter, ``alora`` wins.

    Args:
        root_dir: Root directory to search.
        target_model_name: Target model name (e.g., ``ibm-granite/granite-4.1-3b``).
        arch: Architecture descriptor (for module group lookup).
        technology_fallback: If set, use this technology as fallback for all adapters
            if not detected from the directory name.
        technology_filter: If set, only discover adapters of this technology
            type (``"alora"`` or ``"lora"``).
        source: Original source identifier (e.g., HF repo ID or local path)
            to associate with discovered adapters.

    Returns:
        List of ``(adapter_path, adapter_name, technology, source)`` tuples.
    """
    print("=" * 80)
    print("DISCOVERY PHASE: Finding adapters")
    print("=" * 80)
    print(f"Root directory: {root_dir}")
    print(f"Target model: {target_model_name}")
    print()

    discovered_by_name = {}
    root_path = Path(root_dir)

    print("Searching for LoRA adapters (both alora and lora)...")
    print("  Preference: alora > lora (alora preferred when both exist)")
    print()

    for io_yaml_path in root_path.rglob("*/*/*/io.yaml"):
        adapter_dir = io_yaml_path.parent
        tech = adapter_dir.name

        if tech not in ("alora", "lora"):
            tech = technology_fallback
        if not tech or (technology_filter and tech != technology_filter):
            continue

        model_name = adapter_dir.parent.name
        adapter_name = adapter_dir.parent.parent.name

        if model_name == target_model_name:
            adapter_model = adapter_dir / "adapter_model.safetensors"
            adapter_config_file = adapter_dir / "adapter_config.json"

            if adapter_model.exists() and adapter_config_file.exists():
                if adapter_name in discovered_by_name:
                    existing_tech = discovered_by_name[adapter_name][2]
                    if tech == "alora" and existing_tech == "lora":
                        discovered_by_name[adapter_name] = (
                            str(adapter_dir), adapter_name, tech, source,
                        )
                        print(f"  Found: {adapter_name}/{tech} - replacing lora")
                    else:
                        print(
                            f"  Skipping: {adapter_name}/{tech} "
                            f"(duplicate, keeping {existing_tech})"
                        )
                else:
                    discovered_by_name[adapter_name] = (
                        str(adapter_dir), adapter_name, tech, source,
                    )
                    print(f"  Found: {adapter_name}/{tech}")

    discovered = list(discovered_by_name.values())
    print(f"\nDiscovery complete: {len(discovered)} adapter(s) found")

    # Report per-adapter module contributions
    _report_module_contributions(discovered, arch)

    return discovered


def discover_adapters_from_yaml(
    manifest_path: str
) -> List[Tuple[str, str, str, Optional[str]]]:
    """Discover adapters from a YAML manifest file.

    Reads a YAML manifest that maps adapter names to their paths and types.

    Args:
        manifest_path: Path to the YAML manifest file.

    Returns:
        List of ``(adapter_path, adapter_name, technology, source)`` tuples.
        The source is set to the manifest path for traceability.
    """
    import yaml
    path = Path(manifest_path)
    print(f"  Loading adapters manifest: {path.name}")

    found = []
    if path.is_file() and path.suffix in (".yaml", ".yml"):
        with open(path, 'r') as f:
            adapters_config = yaml.safe_load(f)

        if adapters_config:
            for name, info in adapters_config.items():
                adapter_path = info.get("path")
                tech = info.get("type")
                # Use manifest path as source for traceability
                found.append((adapter_path, name, tech, manifest_path))

    return found



def _report_module_contributions(
    discovered: List[Tuple[str, str, str, Optional[str]]],
    arch: ArchDescriptor,
):
    """Report which module groups each adapter contributes to."""
    print("\nAdapter module contributions:")
    print(f"  Analyzing target modules for {len(discovered)} adapter(s)...")

    adapter_paths_only = [path for path, _, _, *_ in discovered]
    target_module_sets = load_adapter_target_modules(adapter_paths_only)

    adapter_modules = {}
    for idx, (_, adapter_name, _, *_) in enumerate(discovered):
        modules = target_module_sets[idx]

        groups = set()
        for g in arch.groups:
            if any(mod in modules for mod in g.peft_modules):
                groups.add(g.name)

        adapter_modules[adapter_name] = {
            "modules": sorted(modules),
            "groups": sorted(groups),
        }

    all_groups: set = set()
    for info in adapter_modules.values():
        all_groups.update(info["groups"])

    print(f"\n  Union of Switch module groups: {sorted(all_groups)}")

    for g in sorted(arch.groups, key=lambda g: g.name):
        contributors = [
            name
            for name, info in adapter_modules.items()
            if g.name in info["groups"]
        ]
        if contributors:
            print(
                f"    - {g.name}: {len(contributors)}/{len(discovered)} adapters "
                f"({', '.join(contributors[:3])}"
                f"{'...' if len(contributors) > 3 else ''})"
            )
        else:
            print(f"    - {g.name}: NOT PRESENT (will not be instantiated)")


def filter_adapters(
    discovered: List[Tuple[str, str, str, Optional[str]]],
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
) -> List[Tuple[str, str, str, Optional[str]]]:
    """Filter a list of discovered adapters by name patterns.

    Args:
        discovered: Output of :func:`discover_adapters` — list of
            ``(adapter_path, adapter_name, technology, source)`` tuples.
        include: If provided, keep only adapters whose name matches at
            least one pattern (fnmatch glob — ``*``, ``?``, ``[seq]``).
            Exact names work since ``fnmatch("x", "x")`` is ``True``.
        exclude: Drop adapters whose name matches any pattern.
            Applied **after** *include*.

    Returns:
        Filtered list (same tuple format).
    """
    if not include and not exclude:
        return discovered

    result = list(discovered)

    if include:
        result = [
            t for t in result
            if any(fnmatch(t[1], pat) for pat in include)
        ]
        for pat in include:
            if not any(fnmatch(t[1], pat) for t in discovered):
                msg = f"  WARNING: --include-adapters pattern '{pat}' matched nothing"
                print(msg)

    if exclude:
        before = len(result)
        result = [
            t for t in result
            if not any(fnmatch(t[1], pat) for pat in exclude)
        ]
        dropped = before - len(result)
        if dropped:
            print(f"  Excluded {dropped} adapter(s) via --exclude-adapters")

    if include or exclude:
        print(f"  After filtering: {len(result)} adapter(s) remaining")

    return result


def list_available_adapters(
    root_dir: str,
    target_model_name: str,
) -> List[Dict[str, object]]:
    """List all adapters available in an adapter library.

    Unlike :func:`discover_adapters`, this returns **all** technology
    variants (alora *and* lora) without deduplication, so the user can
    see what's available before choosing.

    Args:
        root_dir: Root directory of the adapter library.
        target_model_name: Target model name to filter by.

    Returns:
        List of dicts ``{"name": str, "technologies": [str]}``, sorted
        by adapter name.
    """
    by_name: Dict[str, list] = {}
    root_path = Path(root_dir)

    for io_yaml_path in root_path.rglob("*/*/*/io.yaml"):
        adapter_dir = io_yaml_path.parent
        tech = adapter_dir.name
        if tech not in ("alora", "lora"):
            continue
        model_name = adapter_dir.parent.name
        adapter_name = adapter_dir.parent.parent.name
        if model_name != target_model_name:
            continue
        adapter_model = adapter_dir / "adapter_model.safetensors"
        adapter_config_file = adapter_dir / "adapter_config.json"
        if not (adapter_model.exists() and adapter_config_file.exists()):
            continue
        by_name.setdefault(adapter_name, set()).add(tech)

    return [
        {"name": name, "technologies": sorted(techs)}
        for name, techs in sorted(by_name.items())
    ]


def is_adapter_library(path: str) -> bool:
    """Check if a path is an adapter library (no adapter_config.json at root)."""
    p = Path(path)
    return p.is_dir() and not (p / "adapter_config.json").exists()


# ------------------------------------------------------------------ #
# HuggingFace Hub metadata helpers (no file downloads)
# ------------------------------------------------------------------ #


def _list_repo_adapter_names(repo_id: str) -> List[str]:
    """Get adapter folder names from a HF repo using metadata-only API calls.

    Returns top-level directory names, skipping entries that start with ``_``
    (e.g. ``_ollama``).
    """
    from huggingface_hub import list_repo_tree
    from huggingface_hub.hf_api import RepoFolder

    tree = list_repo_tree(repo_id, repo_type="model")
    return [
        item.path for item in tree
        if isinstance(item, RepoFolder) and not item.path.startswith("_")
    ]


def _resolve_technology(
    repo_id: str,
    adapter_name: str,
    target_model_name: str,
) -> Optional[str]:
    """Resolve preferred technology for an adapter via Hub metadata.

    Prefers ``alora`` over ``lora``. Returns ``None`` if neither exists
    for this adapter/model combination.
    """
    from huggingface_hub import list_repo_tree
    from huggingface_hub.hf_api import RepoFolder
    from huggingface_hub.errors import EntryNotFoundError

    try:
        subtree = list_repo_tree(
            repo_id, repo_type="model",
            path_in_repo=f"{adapter_name}/{target_model_name}",
        )
        technologies = {
            item.path.split("/")[-1] for item in subtree
            if isinstance(item, RepoFolder)
            and item.path.split("/")[-1] in ("alora", "lora")
        }
    except EntryNotFoundError:
        return None

    if "alora" in technologies:
        return "alora"
    elif "lora" in technologies:
        return "lora"
    return None


def _build_allow_patterns(
    repo_id: str,
    target_model_name: Optional[str] = None,
    include_adapters: Optional[List[str]] = None,
    exclude_adapters: Optional[List[str]] = None,
) -> Optional[List[str]]:
    """Build ``allow_patterns`` for selective ``snapshot_download``.

    Uses lightweight Hub API calls to discover adapter names, then applies
    fnmatch-based include/exclude filtering and target model constraints to
    construct download patterns.  When both adapter names and target model
    are known, also resolves the preferred technology (alora > lora) so
    only the needed technology variant is downloaded.

    Returns:
        List of glob patterns, or ``None`` if no filtering is possible.
    """
    adapter_names = _list_repo_adapter_names(repo_id)

    # Apply include filter
    if include_adapters:
        adapter_names = [
            name for name in adapter_names
            if any(fnmatch(name, pat) for pat in include_adapters)
        ]

    # Apply exclude filter
    if exclude_adapters:
        adapter_names = [
            name for name in adapter_names
            if not any(fnmatch(name, pat) for pat in exclude_adapters)
        ]

    # Construct patterns with technology resolution
    if adapter_names and target_model_name:
        patterns = []
        for name in adapter_names:
            tech = _resolve_technology(repo_id, name, target_model_name)
            if tech:
                patterns.append(f"{name}/{target_model_name}/{tech}/**")
            else:
                # Model not found for this adapter — include anyway so
                # discover_adapters can report it as missing downstream
                patterns.append(f"{name}/{target_model_name}/**")
        return patterns if patterns else None
    elif target_model_name:
        return [f"*/{target_model_name}/**"]
    elif adapter_names:
        return [f"{name}/**" for name in adapter_names]
    else:
        return None


def list_repo_adapters_remote(
    repo_id: str,
    target_model_name: str,
) -> List[Dict[str, object]]:
    """List adapters available in a remote HF repo without downloading.

    Uses Hub metadata API calls to discover adapter names and their
    available technologies for the given target model.

    Args:
        repo_id: HuggingFace repo ID (e.g., ``"ibm-granite/granitelib-rag-r1.0"``).
        target_model_name: Target model name (e.g., ``"granite-4.1-3b"``).

    Returns:
        List of dicts ``{"name": str, "technologies": [str]}``, sorted
        by adapter name.
    """
    from huggingface_hub import list_repo_tree
    from huggingface_hub.hf_api import RepoFolder
    from huggingface_hub.errors import EntryNotFoundError

    adapter_names = _list_repo_adapter_names(repo_id)
    results = []

    for name in adapter_names:
        try:
            subtree = list_repo_tree(
                repo_id, repo_type="model",
                path_in_repo=f"{name}/{target_model_name}",
            )
            technologies = sorted(
                item.path.split("/")[-1] for item in subtree
                if isinstance(item, RepoFolder)
                and item.path.split("/")[-1] in ("alora", "lora")
            )
            if technologies:
                results.append({"name": name, "technologies": technologies})
        except EntryNotFoundError:
            # Adapter doesn't have this target model — skip
            continue

    return sorted(results, key=lambda x: x["name"])


# ------------------------------------------------------------------ #
# Path resolution
# ------------------------------------------------------------------ #


def resolve_repo_path(
    path_or_repo: str,
    target_model_name: Optional[str] = None,
    include_adapters: Optional[List[str]] = None,
    exclude_adapters: Optional[List[str]] = None,
) -> str:
    """Resolve a local path or HuggingFace repo ID to a local directory.

    For HuggingFace repos, applies selective downloading using
    ``allow_patterns`` constructed from the provided filters.

    Args:
        path_or_repo: Either a local directory path or a HuggingFace repo ID
            (e.g., ``"ibm-granite/granite-lib-rag-r1.0"``).
        target_model_name: Target model name to filter by (e.g.,
            ``"granite-4.1-3b"``).
        include_adapters: Only download adapters matching these fnmatch
            patterns.
        exclude_adapters: Skip adapters matching these fnmatch patterns.

    Returns:
        Absolute local path to the directory.
    """
    from huggingface_hub import snapshot_download

    local = Path(path_or_repo)

    if local.exists() and (local.is_dir() or local.is_file()):
        return str(local)

    if "/" in path_or_repo and not local.exists():
        print(f"  Detected HuggingFace repo: {path_or_repo}")

        # Build selective download patterns
        allow_patterns = None
        if target_model_name or include_adapters or exclude_adapters:
            try:
                allow_patterns = _build_allow_patterns(
                    path_or_repo,
                    target_model_name=target_model_name,
                    include_adapters=include_adapters,
                    exclude_adapters=exclude_adapters,
                )
                if allow_patterns:
                    print(f"  Selective download patterns: {allow_patterns}")
            except Exception as e:
                print(f"  WARNING: Failed to build download filters ({e}), "
                      f"downloading full repo")
                allow_patterns = None

        print(f"  Downloading from HuggingFace Hub...")
        try:
            kwargs = {"repo_id": path_or_repo, "repo_type": "model"}
            if allow_patterns:
                kwargs["allow_patterns"] = allow_patterns
            cache_dir = snapshot_download(**kwargs)
            print(f"  Downloaded to: {cache_dir}")
            return cache_dir
        except Exception as e:
            raise ValueError(
                f"Failed to download from HuggingFace Hub: "
                f"{path_or_repo}\nError: {e}\n"
                f"Make sure the repository exists and you have access to it."
            )

    raise ValueError(
        f"Path not found and doesn't appear to be a HuggingFace repo: "
        f"{path_or_repo}\n"
        f"Please provide either:\n"
        f"  - A valid local directory path\n"
        f"  - A HuggingFace repo ID (e.g., 'org/repo-name')"
    )


