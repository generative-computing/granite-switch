#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render the four scored accuracy evals as AppWrapper jobs for a vLLM line.

The source of truth is `evals.yaml` next to this script. One config there becomes
one job here, with the vLLM line, the repo sha and the result paths substituted
in. Previously these were four 34 KB YAMLs maintained by copy-paste; the body is
identical across them and only CKPT / BAR / EVAL_MODE ever differed, so the body
is generated and the differences are data.

SECRETS. The rendered jobs need an HF token (gated base models) and a GitHub
token (cloning the repo at a sha). Neither is in `evals.yaml` and neither is
baked in here. Both are read from the environment at render time and appear only
in the rendered output, which must stay gitignored:

    export HF_TOKEN=hf_...
    export GITHUB_COM_TOKEN=ghp_...
    render_eval_jobs.py --vllm 0.31 --sha "$(git rev-parse HEAD)"

Without them, rendering still works but emits `${HF_TOKEN}` placeholders and
prints a warning -- useful for reviewing the shape, not for submitting.

Usage:
    render_eval_jobs.py --vllm 0.31 --sha abc1234
    render_eval_jobs.py --vllm 0.31 --sha abc1234 --config granite5_sr
    render_eval_jobs.py --vllm 0.31 --sha abc1234 --out /tmp/jobs
    render_eval_jobs.py --list
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    sys.exit(
        "PyYAML is required: uvx --with pyyaml python <this script>, or "
        "uv run --with pyyaml <this script>"
    )

HERE = Path(__file__).resolve().parent
CONFIG = HERE.parent / "evals.yaml"
#: Rendered jobs carry live tokens, so they go somewhere gitignored. `scratch/`
#: already is, repo-wide, and is the documented home for throwaway artifacts.
DEFAULT_OUT = "scratch/eval-jobs"

# A sha must be a full or abbreviated hex object name. Interpolated into a shell
# `git checkout`, so anything else is rejected rather than quoted-and-hoped.
SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")
# Line, as 0.NN. Interpolated into the version gate.
LINE_RE = re.compile(r"^\d+\.\d+$")


def repo_root() -> Path:
    for d in [Path.cwd(), *Path(__file__).resolve().parents]:
        p = d / "pyproject.toml"
        if p.is_file() and 'name = "granite-switch"' in p.read_text():
            return d
    sys.exit("could not locate the granite-switch repo root")


def load() -> dict[str, Any]:
    if not CONFIG.is_file():
        sys.exit(f"missing {CONFIG}")
    return yaml.safe_load(CONFIG.read_text())


def job_script(d: dict, spec: dict, markers: list[str]) -> str:
    """The container's shell script.

    Written as one string rather than assembled from fragments because the
    original jobs' step comments explain WHY each gate exists, and those are the
    most valuable part of the file. The gates run cheapest-first: artifacts, then
    the tree, then the environment, then the render, then a 100-row smoke, and
    only then the 3565-row sweep.
    """
    experts_assert = (
        "assert n_exp, ('this checkpoint reports num_local_experts=%r -- it is "
        "NOT the sparse MoE model this job is meant to cover' % n_exp)"
        if spec.get("expect_experts")
        else "assert not n_exp, ('this checkpoint has %s experts -- it is NOT the "
        "dense model this job is meant to cover' % n_exp)"
    )
    experts_echo = (
        "print('  confirmed SPARSE MoE (%s routed experts)' % n_exp)"
        if spec.get("expect_experts")
        else "print('  confirmed DENSE (no routed experts)')"
    )
    marker_checks = "\n".join(
        f'grep -q "{m}" ${{HARNESS_DIR}}/answerability_eval.py || '
        f'{{ echo "FATAL: harness lacks {m}"; sleep 300; exit 1; }}'
        for m in markers
    )
    secondary = spec.get("secondary_bar")
    secondary_py = (
        f"""
        sec = {secondary}
        print('  secondary bar %.4f -> delta %+.4f  '
              '(a move TOWARD this is convergence, not regression)'
              % (sec, acc - sec))"""
        if secondary
        else ""
    )
    smoke = d["smoke_rows_per_file"]

    return f"""set -u
mkdir -p ${{LOG_DIR}}
nvidia-smi 2>&1 | tee ${{LOG_DIR}}/00-nvidia-smi.log

# -------------------------------------------------------- #
# STEP 0: cheap gates first. An absent harness or checkpoint
# must cost seconds, not a model load.
# -------------------------------------------------------- #
echo ""
echo "=== STEP 0: artifacts present? ==="
[ -f ${{HARNESS_DIR}}/answerability_eval.py ] || {{ echo "FATAL: no harness at ${{HARNESS_DIR}}"; sleep 300; exit 1; }}
for d in "${{CKPT}}" "${{DATASET_PATH}}"; do
  [ -d "${{d}}" ] || {{ echo "FATAL: missing ${{d}}"; sleep 300; exit 1; }}
  echo "  ok ${{d}}"
done
echo "  harness md5: $(md5sum ${{HARNESS_DIR}}/answerability_eval.py | cut -d' ' -f1)"
# The harness lives on the PVC, NOT in this repo, so a stale copy is possible
# and both of these failures are silent: wrong precision, or 4.2 rows rendered
# without their documents.
{marker_checks}
echo "  harness markers present"

# -------------------------------------------------------- #
# STEP 1: clone the repo at an exact sha.
# -------------------------------------------------------- #
echo ""
echo "=== STEP 1: clone ${{GS_REPO}} @ ${{GS_SHA}} ==="
git clone https://${{GITHUB_COM_TOKEN}}@${{GS_REPO}} /workspace/gs
cd /workspace/gs
git checkout ${{GS_SHA}} || {{ echo "FATAL: sha not found"; sleep 300; exit 1; }}
echo "HEAD: $(git log --oneline -1)"

# -------------------------------------------------------- #
# STEP 2: install into a --system-site-packages venv. The
# image ships torch/triton but not vllm/transformers, and
# site-packages is read-only for this non-root container.
# -------------------------------------------------------- #
echo ""
echo "=== STEP 2: install .[${{DEP_EXTRA}}] with vllm pinned to ${{VLLM_EXPECT}} ==="
python3 -m venv .venv --system-site-packages
. .venv/bin/activate
pip install --upgrade pip
# The `vllm` extra is a RANGE (it claims the whole supported span), so installing
# the extra alone gets the newest line in that range -- not the line this job is
# named for. Naming `vllm==${{VLLM_EXPECT}}.*` in the SAME pip invocation makes the
# resolver satisfy both at once; the line gate below then only has to confirm it.
pip install -e ".[${{DEP_EXTRA}}]" "vllm==${{VLLM_EXPECT}}.*" pandas numpy scikit-learn > ${{LOG_DIR}}/02-install.log 2>&1
RC=$?
tail -5 ${{LOG_DIR}}/02-install.log
if [ ${{RC}} -ne 0 ]; then echo "INSTALL FAILED"; sleep 600; exit 1; fi
# pip WARNS and exits 0 on an unrecognised extra, so a renamed extra installs
# nothing and reads as success.
if grep -qi "does not provide the extra" ${{LOG_DIR}}/02-install.log; then
  echo "FATAL: pip ignored an extra:"; grep -i "does not provide the extra" ${{LOG_DIR}}/02-install.log
  sleep 600; exit 1
fi

echo ""
echo "=== environment under test ==="
python -c "import torch; print(f'torch={{torch.__version__}} cuda_build={{torch.version.cuda}} available={{torch.cuda.is_available()}} gpus={{torch.cuda.device_count()}}')"
python -c "import transformers; print(f'transformers={{transformers.__version__}}')"
python -c "import vllm; print(f'vllm={{vllm.__version__}}')" || {{ echo "FATAL: vLLM did not install"; sleep 600; exit 1; }}
python -c "import pandas, numpy, sklearn; print('eval deps OK')"

# Without this gate a green is a lie about WHICH vLLM ran -- the whole point of
# the job is the line under test.
echo ""
echo "=== vLLM line gate: expect ${{VLLM_EXPECT}}.x ==="
python - <<PYEOF || {{ echo "VLLM LINE GATE FAILED"; sleep 600; exit 1; }}
import vllm, os
got, want = vllm.__version__, os.environ['VLLM_EXPECT']
assert got.startswith(want + '.'), 'expected %s.x, got %s' % (want, got)
print('vLLM line OK: %s' % got)
PYEOF

# -------------------------------------------------------- #
# STEP 3: RENDER GATE. The most important step. If the
# documents are dropped from the prompt, every row loses its
# evidence and the score lands at the unanswerable base rate
# -- indistinguishable from a bad adapter unless asserted.
# -------------------------------------------------------- #
echo ""
echo "=== STEP 3: render gate ==="
python - <<'PYEOF' || {{ echo "RENDER GATE FAILED"; sleep 600; exit 1; }}
import json, os, sys
sys.path.insert(0, os.environ['HARNESS_DIR'])
from transformers import AutoTokenizer
from answerability_eval import load_dataset, render_chat

name = os.environ['ADAPTER_NAME']
cap = int(os.environ['MAX_MODEL_LEN'])
d = os.environ['CKPT']
msgs, docs, labels, coll, setting = load_dataset(os.environ['DATASET_PATH'])
print('loaded rows: %d' % len(msgs))
ctl = '<|%s|>' % name

cfg = json.load(open(os.path.join(d, 'config.json')))
print('--- %s ---' % d)
n_exp = cfg.get('num_local_experts')
print('  num_local_experts=%s shared_intermediate_size=%s dual_stream=%s'
      % (n_exp, cfg.get('shared_intermediate_size'), cfg.get('dual_stream')))
# Mounting the wrong checkpoint produces a plausible score for the wrong model,
# i.e. a duplicate run masquerading as independent coverage.
{experts_assert}
{experts_echo}
tok = AutoTokenizer.from_pretrained(d)
# Detect via get_vocab, never convert_tokens_to_ids -- the latter returns unk
# for unknown tokens and so false-positives.
print('  chatml=%s' % ('<|im_start|>' in tok.get_vocab()))

# A row that actually HAS documents; no-document rows cannot detect a drop.
i = next(i for i, x in enumerate(docs) if x)
dd = [{{'text': x}} for x in docs[i]]
p = render_chat(tok, msgs[i], dd, adapter_name=name)
assert p.count(ctl) == 1, '%s appears %d times' % (ctl, p.count(ctl))
# The template JSON-encodes documents and these rows are CSV-derived with
# DOUBLED quotes mid-line, so a raw needle can be absent from a good render.
# Escape the needle the same way the template does.
frag = dd[0]['text'].split('\\n')[0].strip()[:40]
assert len(frag) >= 12, 'no usable needle: %r' % frag
esc = json.dumps(frag)[1:-1]
assert frag in p or esc in p, (
    'document text is NOT in the prompt -- the template dropped the '
    'documents: %r' % frag)
print('  row %d: %s once, %d docs, document text present' % (i, ctl, len(dd)))
# Thinking mode would spend the 6 generated tokens on reasoning instead of the
# verdict. The harness defaults enable_thinking=False, so this should be False.
print('  think marker in prompt: %s'
      % any(m in p for m in ('<think>', '<|think|>')))

missing = 0
longest = 0
longest_row = -1
for j in range(len(msgs)):
    dj = [{{'text': x}} for x in docs[j]] if docs[j] else []
    pj = render_chat(tok, msgs[j], dj, adapter_name=name)
    if ctl not in pj:
        missing += 1
    n = len(tok(pj, add_special_tokens=False)['input_ids'])
    if n > longest:
        longest, longest_row = n, j
print('  rows missing %s: %d of %d' % (ctl, missing, len(msgs)))
assert missing == 0
print('  longest prompt: %d tokens (row %d); cap %d' % (longest, longest_row, cap))
assert longest + 8 < cap, (
    'MAX_MODEL_LEN=%d does not fit the longest prompt (%d tokens + 6 '
    'generated); vLLM would truncate it and the only symptom would be a '
    'lower score.' % (cap, longest))
print('RENDER GATE OK')
PYEOF

# -------------------------------------------------------- #
# STEP 4: eval. --dataset-size is PER FILE and there are 5
# files, so `--dataset-size {smoke}` is {smoke * 5} rows.
# -------------------------------------------------------- #
run_eval() {{
  OUT="$1"; TAG="$2"; EXTRA="$3"
  mkdir -p "${{OUT}}"
  cd /workspace/gs
  MAX_MODEL_LEN=${{MAX_MODEL_LEN}} PYTHONPATH=${{HARNESS_DIR}} \\
  python ${{HARNESS_DIR}}/answerability_eval.py \\
    --mode ${{EVAL_MODE}} \\
    --model-path "${{CKPT}}" \\
    --intrinsic-name ${{ADAPTER_NAME}} \\
    --dataset-path ${{DATASET_PATH}} \\
    --output-path "${{OUT}}" \\
    --dtype ${{DTYPE}} \\
    ${{MODE_ARGS}} \\
    ${{EXTRA}} > ${{LOG_DIR}}/10-${{TAG}}.log 2>&1
  RC=$?
  tail -25 ${{LOG_DIR}}/10-${{TAG}}.log
  echo "--- exit ${{RC}} for ${{TAG}} -> ${{OUT}}"
  cat "${{OUT}}/eval_summary.md" 2>/dev/null || echo "NO SUMMARY"
  return ${{RC}}
}}

echo ""
echo "=== STEP 4: smoke ({smoke * 5} rows) ==="
run_eval "${{RESULTS}}-smoke" "smoke" "--dataset-size {smoke}"
if [ $? -ne 0 ]; then echo "SMOKE FAILED -- not spending GPU on the sweep"; sleep 600; exit 1; fi

echo ""
echo "=== STEP 5: full sweep, 3565 rows (bar ${{BAR}}) ==="
run_eval "${{RESULTS}}" "full" ""

# -------------------------------------------------------- #
# STEP 6: verdict.
# -------------------------------------------------------- #
echo ""
echo "=== SUMMARY: ${{EVAL_MODE}} on the ${{VLLM_EXPECT}} stack vs bar ${{BAR}} ==="
python - <<'PYEOF'
import json, os
d = os.environ['RESULTS']
bar = float(os.environ['BAR'])
tol = float(os.environ['TOLERANCE'])
floor = float(os.environ['STACK_BUG_FLOOR'])
p = os.path.join(d, 'eval_summary.jsonl')
if not os.path.exists(p):
    print('NO RESULTS at %s' % p)
else:
    last = [json.loads(l) for l in open(p) if l.strip()][-1]
    acc = last.get('accuracy')
    if acc is None:
        print('no accuracy key: %s' % sorted(last))
    else:
        delta = acc - bar
        if acc < floor:
            v = ('STACK BUG -- ~0.40 is the unanswerable base rate: dropped '
                 'documents, a dead adapter, or thinking mode')
        elif abs(delta) <= tol:
            v = 'MATCH (within +/-%g)' % tol
        else:
            v = 'DRIFT -- faithful band but off the bar; investigate'
        print('%-8s %-10s %-11s %s' % (os.environ['EVAL_MODE'], 'bar', 'delta', 'verdict'))
        print('%-8.4f %-10.4f %+-11.4f %s' % (acc, bar, delta, v)){secondary_py}
        for k in ('answerability_f1', 'weighted_f1'):
            if k in last:
                print('  %s=%.4f' % (k, last[k]))
PYEOF

echo ""
echo "results: ${{RESULTS}}"
echo "logs:    ${{LOG_DIR}}"
echo ""
echo "NOTE: {spec["bar_note"].strip()}"
echo "=== DONE ==="
# Keep the pod alive so the logs stay fetchable after a pass, too.
sleep 30000
"""


def render(name: str, spec: dict, data: dict, line: str, sha: str, tag: str) -> str:
    c, p, d = data["cluster"], data["paths"], data["defaults"]
    merged = {**d, **spec}
    job = f"{c['job_prefix']}-{tag}-{name}".replace("_", "-")

    env = {
        "PYTHONUNBUFFERED": "1",
        "HF_HOME": p["hf_home"],
        # Render-time injection. Empty is tolerated so the shape can be reviewed
        # without secrets on the box; submitting with an empty token fails at the
        # clone or the base-model fetch, loudly.
        "HF_TOKEN": os.environ.get("HF_TOKEN", "${HF_TOKEN}"),
        "GITHUB_COM_TOKEN": os.environ.get("GITHUB_COM_TOKEN", "${GITHUB_COM_TOKEN}"),
        "GS_REPO": "github.com/generative-computing/granite-switch.git",
        "GS_SHA": sha,
        "DEP_EXTRA": merged["dep_extra"],
        "VLLM_EXPECT": line,
        "VLLM_USE_FLASHINFER_SAMPLER": "1" if merged["flashinfer_sampler"] else "0",
        "HARNESS_DIR": p["harness"],
        "CKPT": f"{p['checkpoints']}/{spec['checkpoint']}",
        "DATASET_PATH": p["dataset"],
        "RESULTS": f"{p['results']}/{tag}-{name}",
        "LOG_DIR": f"{p['logs']}/{tag}-{name}",
        "ADAPTER_NAME": merged["adapter"],
        "DTYPE": merged["dtype"],
        "MAX_MODEL_LEN": merged["max_model_len"],
        "BAR": merged["bar"],
        "TOLERANCE": merged["tolerance"],
        "STACK_BUG_FLOOR": merged["stack_bug_floor"],
        "EVAL_MODE": spec["mode"],
        "MODE_ARGS": spec.get("mode_args", ""),
    }

    res = {
        "cpu": c["resources"]["cpu"],
        "memory": c["resources"]["memory"],
        "nvidia.com/gpu": c["resources"]["gpu"],
    }
    ann = "workload.codeflare.dev.appwrapper"
    doc = {
        "apiVersion": "workload.codeflare.dev/v1beta2",
        "kind": "AppWrapper",
        "metadata": {
            "name": job,
            "namespace": c["namespace"],
            "annotations": {
                f"{ann}/failureGracePeriodDuration": c["failure_grace"],
                f"{ann}/retryLimit": str(c["retry_limit"]),
                f"{ann}/deletionOnFailureGracePeriodDuration": c["failure_grace"],
            },
            "labels": {"kueue.x-k8s.io/queue-name": c["queue"]},
        },
        "spec": {
            "components": [
                {
                    "template": {
                        "apiVersion": "kubeflow.org/v1",
                        "kind": "PyTorchJob",
                        "metadata": {"name": job},
                        "spec": {
                            "pytorchReplicaSpecs": {
                                "Master": {
                                    "replicas": 1,
                                    "restartPolicy": "Never",
                                    "template": {
                                        "spec": {
                                            "priorityClassName": c["priority_class"],
                                            "imagePullSecrets": [
                                                {"name": c["image_pull_secret"]}
                                            ],
                                            "containers": [
                                                {
                                                    "name": "pytorch",
                                                    "image": c["image"],
                                                    "imagePullPolicy": "IfNotPresent",
                                                    "command": [
                                                        "sh",
                                                        "-c",
                                                        job_script(
                                                            merged,
                                                            spec,
                                                            data["harness_markers"],
                                                        ),
                                                    ],
                                                    "env": [
                                                        {"name": k, "value": str(v)}
                                                        for k, v in env.items()
                                                    ],
                                                    "resources": {
                                                        "requests": res,
                                                        "limits": res,
                                                    },
                                                    "volumeMounts": [
                                                        {
                                                            "name": c["pvc"],
                                                            "mountPath": c["mount"],
                                                        },
                                                        {
                                                            "name": "dshm",
                                                            "mountPath": "/dev/shm",
                                                        },
                                                    ],
                                                }
                                            ],
                                            "volumes": [
                                                {
                                                    "name": c["pvc"],
                                                    "persistentVolumeClaim": {
                                                        "claimName": c["pvc"]
                                                    },
                                                },
                                                {
                                                    "name": "dshm",
                                                    "emptyDir": {
                                                        "medium": "Memory",
                                                        "sizeLimit": c["shm_size"],
                                                    },
                                                },
                                            ],
                                        }
                                    },
                                }
                            }
                        },
                    }
                }
            ]
        },
    }

    header = (
        "# GENERATED by .claude/skills/upgrade-vllm/scripts/render_eval_jobs.py\n"
        f"# config: {name}  vLLM line: {line}  sha: {sha}\n"
        f"# {spec['description']}\n"
        "# Edit evals.yaml and re-render; do not hand-edit this file.\n"
        "# MAY CONTAIN LIVE TOKENS -- do not commit.\n"
    )
    return header + yaml.safe_dump(doc, sort_keys=False, width=10000)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm", help="the line under test, e.g. 0.31")
    ap.add_argument("--sha", help="repo sha the jobs clone and check out")
    ap.add_argument("--config", default="all", help="one config name, or 'all'")
    ap.add_argument("--out", help=f"output dir (default {DEFAULT_OUT})")
    ap.add_argument("--tag", help="results/log tag (default v<line>)")
    ap.add_argument("--list", action="store_true", help="list configs and exit")
    args = ap.parse_args()

    data = load()

    if args.list:
        print(f"{'config':<18} {'mode':<12} {'bar':<8} {'tol':<7} checkpoint")
        for n, s in data["configs"].items():
            tol = s.get("tolerance", data["defaults"]["tolerance"])
            print(f"{n:<18} {s['mode']:<12} {s['bar']:<8} {tol:<7} {s['checkpoint']}")
        return 0

    if not args.vllm or not args.sha:
        sys.exit("--vllm and --sha are required (or use --list)")
    if not LINE_RE.match(args.vllm):
        sys.exit(f"--vllm must look like 0.31, got {args.vllm!r}")
    # Interpolated into a shell command inside the job, so validate rather than
    # trusting the caller.
    if not SHA_RE.match(args.sha):
        sys.exit(f"--sha must be a hex git object name (7-40 chars), got {args.sha!r}")

    names = list(data["configs"]) if args.config == "all" else [args.config]
    for n in names:
        if n not in data["configs"]:
            sys.exit(f"unknown config {n!r}; choose from {', '.join(data['configs'])}")

    tag = args.tag or f"v{args.vllm.replace('.', '')}"
    out = Path(args.out) if args.out else repo_root() / DEFAULT_OUT
    out.mkdir(parents=True, exist_ok=True)

    missing = [v for v in ("HF_TOKEN", "GITHUB_COM_TOKEN") if not os.environ.get(v)]
    if missing:
        print(f"WARNING: {', '.join(missing)} unset -- rendering placeholders.")
        print("         The jobs will FAIL if submitted as-is. Export and re-render.")
        print()

    for n in names:
        text = render(n, data["configs"][n], data, args.vllm, args.sha, tag)
        p = out / f"job-eval-{tag}-{n.replace('_', '-')}.yaml"
        p.write_text(text)
        print(f"wrote {p}")

    ns = data["cluster"]["namespace"]
    print()
    print(f"submit: kubectl apply -f {out}/")
    print(f"watch : kubectl -n {ns} get appwrappers | grep {tag}")
    print(f"logs  : kubectl -n {ns} logs -f <pod>")
    if not missing:
        print()
        print("These files contain live tokens. Confirm they are gitignored:")
        print(f"  git check-ignore {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
