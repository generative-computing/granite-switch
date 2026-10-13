# SPDX-License-Identifier: Apache-2.0
"""Stage a Hugging Face release for the benchmark (``releases`` in adapters.yaml).

Runs on a pod, with ``huggingface_hub`` (``vela/pod_entry.sh``)::

    python -m benchmarks.adapter_eval.release stage --release 7a3ac02e \\
        --root <release folder> --eval-from <the model's bench root> [--replace]

The release's folder (``RELEASE_ROOT/<model>/<repo name>/<revision>``)::

    checkpoint/                   the composed checkpoint at the release's revision
    native/<adapter>/             each of its adapters from its library, at the
                                  revision the release pins: what stock vLLM serves
    adapters/<intrinsic>/<tech>/  the benchmark's intrinsics among them, as a bench
                                  root holds them, for HF + PEFT (reference.py)
    eval/<intrinsic>/evaluation.jsonl   the model's eval sets, copied from its root
    model.json, release.json      the model; what was staged, from which Hub commits

An adapter's library folder is ``<name>/<base>/<technology>``, the granitelib
repositories' layout, with its name and technology from the checkpoint's
``adapter_index.json``. Every library revision the release pins must appear in
the checkpoint's BUILD.md, its record of what it was composed from.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
from pathlib import Path

from . import stage, staged
from .common import NOT_IN_RELEASE, load_spec

CHECKPOINT = "checkpoint"
NATIVE = "native"
RELEASE_FILE = "release.json"
INDEX_FILE = "adapter_index.json"
BUILD_FILE = "BUILD.md"
# What the benchmark uses of an adapter's library folder.
LEAF_FILES = (staged.CONFIG_FILE, staged.WEIGHTS_FILE, "io.yaml")


def read_release(root: Path, release) -> dict:
    """The record of a staged release; raises if ``root`` holds another revision."""
    record = json.loads((root / RELEASE_FILE).read_text())
    if record.get("revision") != release.revision:
        raise ValueError(
            f"{root} holds {record.get('revision')}, not {release.revision}"
        )
    return record


def discover(root: Path, spec, release, only=None) -> list[staged.StagedCell]:
    """``staged.discover`` on a release's folder, where a cell the release has
    no adapter for is skipped as not in it."""
    cells = []
    for c in staged.discover(root, spec, only):
        absent = (
            c.intrinsic not in release.adapters
            or not staged.adapter_dir(root, c.intrinsic, c.tech).is_dir()
        )
        if c.skip_reason and absent:
            c = dataclasses.replace(c, skip_reason=NOT_IN_RELEASE)
        cells.append(c)
    return cells


def native_copies(root: Path, record: dict, dest: Path) -> list[Path]:
    """The release's adapters as stock vLLM serves them, in the checkpoint's
    order and named as the switch benchmark's drivers look for them
    (``adapter_NN``); an aLoRA without its invocation tokens, which would keep
    it off after a one-token prompt (``staged.peft_sr_copy``), and none asking
    for bias weights it does not have (``unused_bias_dropped``)."""
    leaves = []
    for k, adapter in enumerate(record["adapters"]):
        src, leaf = root / NATIVE / adapter["name"], dest / f"adapter_{k:02d}"
        if not staged.peft_sr_copy(src, leaf):
            staged.link_others(src, leaf, "")
        unused_bias_dropped(leaf)
        leaves.append(leaf)
    return leaves


def unused_bias_dropped(leaf: Path) -> bool:
    """Make a config that asks for bias weights its checkpoint never saved say
    ``bias: none``; True if it did. Stock vLLM refuses an adapter whose config
    asks for bias, and some library aLoRAs say ``lora_only`` with no bias
    tensor, so the weights it serves stay the same."""
    path = leaf / staged.CONFIG_FILE
    config = json.loads(path.read_text())
    keys = staged.safetensors_keys(leaf / staged.WEIGHTS_FILE)
    if config.get("bias", "none") == "none" or any("bias" in k for k in keys):
        return False
    config["bias"] = "none"
    path.unlink()  # a link to the staged config, or peft_sr_copy's own copy
    path.write_text(json.dumps(config, indent=2))
    return True


def switching_settings(spec, record: dict) -> dict:
    """The switching cell over the release's own adapters: the benchmark's
    settings, with its adapters, as many as it has, at their largest rank."""
    return {
        **spec.switching.settings(),
        "adapters": len(record["adapters"]),
        "rank": max(a["rank"] for a in record["adapters"]),
    }


def stage_release(
    spec, release, root: Path, eval_from: Path, replace: bool, *, downloads: Path
) -> dict:
    """Fetch ``release`` into ``root``, through ``downloads``: the Hub's downloads
    go to local disk first (its locks need one), then onto the volume."""
    from huggingface_hub import HfApi, snapshot_download

    if (root / RELEASE_FILE).is_file() and not replace:
        print(f"[release] {root} already holds it (--replace to fetch it again)")
        return read_release(root, release)
    api = HfApi()
    root.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(downloads, ignore_errors=True)
    local = downloads / CHECKPOINT
    snapshot_download(release.repo, revision=release.revision, local_dir=local)
    index = json.loads((local / INDEX_FILE).read_text())
    build = (local / BUILD_FILE).read_text()
    unnamed = [
        f"{lib}@{rev[:8]}" for lib, rev in release.sources.items() if rev not in build
    ]
    if unnamed:
        raise ValueError(f"its {BUILD_FILE} does not name {unnamed}")
    adapters = sorted(index["adapters"], key=lambda a: a["adapter_index"])
    order = json.loads((local / "config.json").read_text()).get("adapter_names")
    if order != [a["adapter_name"] for a in adapters]:
        raise ValueError(f"{INDEX_FILE} and config.json list different adapters")
    absent = sorted(set(release.adapters.values()) - set(order))
    if absent:
        raise ValueError(f"the release has no adapter {absent}")
    base = index["model_info"]["base_model"]

    # Each adapter from the one pinned library that has its folder.
    listed = {
        lib: set(api.list_repo_files(lib, revision=rev))
        for lib, rev in release.sources.items()
    }
    leaves = {}
    for a in adapters:
        path = f"{a['adapter_name']}/{base}/{a['technology']}"
        libs = [
            lib
            for lib, files in listed.items()
            if f"{path}/{staged.CONFIG_FILE}" in files
        ]
        if len(libs) != 1:
            raise ValueError(f"{path} is in {len(libs)} of the pinned libraries")
        leaves[a["adapter_name"]] = (libs[0], path)
    for lib, rev in release.sources.items():
        wanted = [
            f"{p}/{f}" for (src, p) in leaves.values() if src == lib for f in LEAF_FILES
        ]
        if wanted:
            snapshot_download(
                lib, revision=rev, allow_patterns=wanted, local_dir=downloads / lib
            )

    record = {
        "model": spec.model_id,
        **{k: getattr(release, k) for k in ("repo", "revision", "base_revision")},
        "base": base,
        "adapters": [],
    }
    commit = next(iter(api.list_repo_commits(release.repo, revision=release.revision)))
    record.update(date=commit.created_at.isoformat(), title=commit.title)
    by_name = {}
    for a in adapters:
        lib, path = leaves[a["adapter_name"]]
        src = downloads / lib / path
        files = {f: src / f for f in LEAF_FILES if (src / f).is_file()}
        hub = {"repo": lib, "revision": release.sources[lib], "path": path}
        config = json.loads((src / staged.CONFIG_FILE).read_text())
        prov = {
            "source": f"hf://{lib}@{hub['revision']}/{path}",
            "hub": hub,
            "technology": a["technology"],
            "rank": config.get("r"),
        }
        dest = root / NATIVE / a["adapter_name"]
        prov = stage.copy_into(files, dest, prov, replace=True)
        reason = staged.check_adapter(dest, a["technology"])
        if reason:
            raise ValueError(f"{a['adapter_name']}: {reason}")
        by_name[a["adapter_name"]] = (dest, prov)
        record["adapters"].append(
            {
                "name": a["adapter_name"],
                "technology": a["technology"],
                "token_id": a["control_token"]["id"],
                **hub,
                "rank": prov["rank"],
                "weights_sha256": prov["files"][staged.WEIGHTS_FILE],
            }
        )

    # The benchmark's intrinsics: a bench root's layout, and the model's eval sets.
    techs = {a["adapter_name"]: a["technology"] for a in adapters}
    for intrinsic_id, name in release.adapters.items():
        src, prov = by_name[name]
        files = {f: src / f for f in LEAF_FILES if (src / f).is_file()}
        stage.copy_into(
            files,
            staged.adapter_dir(root, intrinsic_id, techs[name]),
            {k: v for k, v in prov.items() if k not in ("files", "staged_at")},
            replace=True,
        )
        ev = staged.eval_path(eval_from, intrinsic_id)
        stage.copy_into(
            {staged.EVAL_FILE: ev},
            staged.eval_path(root, intrinsic_id).parent,
            {"source": str(ev), "source_sha256": stage.sha256(ev)},
            replace=True,
        )
    record["intrinsics"] = {
        i: [name, techs[name]] for i, name in release.adapters.items()
    }

    # The checkpoint onto the volume, every file checked for its size.
    tmp = root / f"{CHECKPOINT}.staging"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(local, tmp, ignore=shutil.ignore_patterns(".cache"))
    for f in local.rglob("*"):
        rel = f.relative_to(local)
        if f.is_file() and ".cache" not in rel.parts:
            if (tmp / rel).stat().st_size != f.stat().st_size:
                raise OSError(f"size mismatch copying {rel}")
    shutil.rmtree(root / CHECKPOINT, ignore_errors=True)
    tmp.rename(root / CHECKPOINT)
    shutil.rmtree(downloads, ignore_errors=True)
    (root / staged.MODEL_FILE).write_text(
        json.dumps({"model": spec.model_id, "name": spec.base_model}, indent=2)
    )
    record["staged_at"] = stage.now()
    (root / RELEASE_FILE).write_text(json.dumps(record, indent=2, sort_keys=True))
    return record


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage", help="fetch a release into its folder")
    s.add_argument("--release", required=True, help="its revision (8+ characters)")
    s.add_argument("--root", required=True, type=Path, help="the release's folder")
    s.add_argument(
        "--eval-from", required=True, type=Path, help="the model's bench root"
    )
    s.add_argument("--replace", action="store_true")
    s.add_argument(
        "--downloads",
        type=Path,
        default=Path("/workspace/release-downloads"),
        help="local disk for the Hub's downloads (default: the pod's)",
    )
    args = p.parse_args(argv)

    spec = load_spec()
    release = spec.release(args.release)
    spec = spec.for_model(release.model)
    staged.check_bench_root(args.eval_from, spec)
    record = stage_release(
        spec, release, args.root, args.eval_from, args.replace, downloads=args.downloads
    )
    print("\n[release] the folder now holds:")
    for cell in discover(args.root, spec, release):
        print(f"  {cell.intrinsic}/{cell.tech}: {cell.skip_reason or 'ok'}")
    print(stage.STAGE_BEGIN)
    print(json.dumps(record, sort_keys=True))
    print(stage.STAGE_END, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
