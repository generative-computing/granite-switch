#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Record (or refresh) the committed eval baselines ON A DEV BOX.

NOT THE SANCTIONED PATH. Record on the GPU runner instead, with

    gh workflow run gpu-tests.yaml --ref <branch> -f suite=eval-record -f sha=<sha>

(see `tests/eval/test_record_baselines.py`), because a baseline is only a
reference if the environment that checks it can reproduce it. A baseline recorded
on a Vela pod was off by a systematic ~0.004 mean JSD on the CI runner --
identically on the 0.26 and 0.30 legs, so neither version drift nor torch. The
difference in the vLLM capture path between those machines was never isolated;
recording where the check runs makes it moot.

This script survives for the case where that is not available: a dev box with the
checkpoints staged, comparing two locally-installed vLLM lines against each other.
Run it from a KNOWN-GOOD line, BEFORE porting:

    uv run --group dev-vllm30 \
      .claude/skills/upgrade-vllm/scripts/record_eval_baseline.py --config all

A baseline captured on the version under test proves nothing, which is why this
prints the version it recorded against and refuses to overwrite an existing
baseline without --force.

Uses tests/eval/_eval_core.py -- the same code the check and the runner recorder
both run -- so the baseline cannot describe something different from what is later
measured, and a file written here is byte-comparable with one printed there.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def repo_root() -> Path:
    for d in Path(__file__).resolve().parents:
        p = d / "pyproject.toml"
        if p.is_file() and 'name = "granite-switch"' in p.read_text():
            return d
    sys.exit("could not locate the granite-switch repo root")


ROOT = repo_root()
sys.path.insert(0, str(ROOT))

from tests.eval import _eval_core as core  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="all",
        help=f"one of {', '.join(core.EVAL_CONFIGS)}, or 'all'",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing baseline (adopts whatever this run produces)",
    )
    args = ap.parse_args()

    if args.config == "all":
        names = list(core.EVAL_CONFIGS)
    elif args.config in core.EVAL_CONFIGS:
        names = [args.config]
    else:
        sys.exit(
            f"unknown config {args.config!r}; choose from {', '.join(core.EVAL_CONFIGS)}"
        )

    vers = core.versions()
    print(
        f"recording against vllm {vers['vllm']}, torch {vers['torch']}, "
        f"transformers {vers['transformers']}"
    )
    if vers["vllm"] == "absent":
        sys.exit(
            "vLLM is not installed in this environment. Record with an explicit "
            "line, e.g.  uv run --group dev-vllm30 <this script>"
        )

    recorded, skipped = [], []
    for name in names:
        cfg = core.EVAL_CONFIGS[name]
        out = core.baseline_path(name)

        model_dir = core.checkpoint_dir(cfg)
        if model_dir is None:
            envs = " or ".join(filter(None, (cfg.env, cfg.fallback_env)))
            print(f"  skip   {name}: {envs} is unset")
            skipped.append(name)
            continue

        if out.is_file() and not args.force:
            existing = json.loads(out.read_text())
            print(
                f"  skip   {name}: baseline already exists (recorded on vllm "
                f"{existing['recorded_on'].get('vllm')}). Pass --force only if a "
                "human has accepted the change."
            )
            skipped.append(name)
            continue

        print(f"  run    {name}: {cfg.description}")
        print(f"         checkpoint {model_dir}")
        data = core.capture(cfg, model_dir)
        out.parent.mkdir(parents=True, exist_ok=True)
        # Deterministic formatting: these files are committed and reviewed, and a
        # reordered dict would show up as a diff that is not a change.
        out.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
        size_kb = out.stat().st_size / 1024
        print(
            f"         wrote {out.relative_to(ROOT)} ({size_kb:.0f} KB), "
            f"adapter {data['adapter']['name']!r}, "
            f"fingerprint {data['checkpoint_fingerprint']}"
        )
        # check-added-large-files --maxkb=500 is a pre-commit hook; a baseline over
        # it cannot be committed, and the fix is fewer prompts or a smaller TOP_K.
        if size_kb > 450:
            print(
                f"         WARNING: {size_kb:.0f} KB is near the 500 KB pre-commit "
                "ceiling -- reduce GSW_EVAL_TOP_K or the prompt count"
            )
        recorded.append(name)

    print()
    print(f"recorded: {', '.join(recorded) or '(none)'}")
    print(f"skipped : {', '.join(skipped) or '(none)'}")
    if recorded:
        print("\nCommit these. They are the eval gate -- without them the eval suite")
        print("fails for any config whose checkpoint is staged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
