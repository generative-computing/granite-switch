# SPDX-License-Identifier: Apache-2.0
"""Does PRESERVE_MIXED_HISTORY actually save work? Real model, real server.

The routing claim is covered on CPU (``tests/unit/test_conversation_policy.py``,
``tests/hf/test_conversation_routing.py``). The *economic* claim is not, and it is
the reason the policy exists: keeping turn 1's ids should let the prefix cache
serve turn 1's blocks -- including the ones computed under adapter 1 -- so turn 2
only prefills the new turn. Nothing off-GPU can show that.

Two independent measurements, because either alone can mislead:

  ROWS   how many positions the switch was actually asked to route for turn 2,
         captured inside the server process. Positions served from cache never
         reach the model, so this is a direct count of recomputation. Under
         PRESERVE it should be roughly the delta; under RE_PREFILL, far more.
  HITS   vLLM's own ``vllm:prefix_cache_hits_total`` / ``_queries_total``,
         scraped from /metrics around each request. Guards against ROWS being
         low for some unrelated reason (e.g. the request never ran).

Prefix caching is ENABLED here -- reuse is the thing under test.

Commands (JSON to stdout; diagnostics to stderr):
  build --output-dir <dir>                       compose a multi aLoRA checkpoint
  serve --model-path <dir> --output-path <json>  run both policies, measure
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

BASE_MODEL = "ibm-granite/granite-4.1-3b"
# granitelib-rag carries aLoRA intrinsics. PRESERVE needs the turn's control
# token to land inside the turn being generated, which is what aLoRA placement
# does; a LoRA adapter's token goes to position 0, inside the already-sent
# region, and Conversation refuses that case loudly rather than mis-slicing.
ADAPTER_REPOS = ["ibm-granite/granitelib-rag-r1.0"]
DEFAULT_PORT = 8137
MAX_TOKENS = 24

_ADAPTER_MAP_RE = re.compile(
    r"'(?P<name>[^']+)':\s*\{'token':\s*'(?P<token>[^']+)',\s*"
    r"'type':\s*'(?P<type>[^']+)'(?:,\s*'invocation_text':\s*'(?P<inv>[^']*)')?\}"
)


def _parse_adapter_map(chat_template):
    """Read name -> (token, type, invocation_text) out of the composed template.

    The composer bakes this map into the template, so it is the checkpoint's own
    account of where each adapter's control token belongs -- no second source of
    truth to drift.
    """
    return {
        m.group("name"): (m.group("token"), m.group("type"), m.group("inv"))
        for m in _ADAPTER_MAP_RE.finditer(chat_template)
    }


def cmd_build(args):
    if os.path.exists(os.path.join(args.output_dir, "config.json")):
        print(f"warm-reuse {args.output_dir}", file=sys.stderr)
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "granite_switch.composer.compose_granite_switch",
            "--base-model",
            BASE_MODEL,
            *[arg for r in ADAPTER_REPOS for arg in ("--adapters", r)],
            "--technology-filter",
            "alora",
            "--output",
            args.output_dir,
        ]
        print("composing:", " ".join(cmd), file=sys.stderr)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-3000:], file=sys.stderr)
            print("STDERR:", r.stderr[-3000:], file=sys.stderr)
            raise RuntimeError(f"compose failed ({r.returncode})")

    with open(os.path.join(args.output_dir, "config.json")) as f:
        cfg = json.load(f)
    assert cfg.get("switch_type") == "multi", f"switch_type={cfg.get('switch_type')!r}"
    print(
        f"num_adapters={cfg.get('num_adapters')} names={cfg.get('adapter_names')}",
        file=sys.stderr,
    )
    print("BUILD_OK")
    return 0


# ── in-server trace: how many positions did the switch actually route? ────────
_SITECUSTOMIZE = r"""
import importlib.abc, importlib.util, json, os, sys

_T = os.environ.get("GS_CONV_TRACE")
if _T:
    _TARGET = "granite_switch.vllm.switch.multi"

    def _patch(mod):
        cls = getattr(mod, "MultiSwitch", None)
        if cls is None or getattr(cls, "_gs_conv_patched", False):
            return
        orig = cls.forward

        def traced(self, input_ids, adapter_token_ids, positions=None):
            ai, modified = orig(self, input_ids=input_ids,
                                adapter_token_ids=adapter_token_ids,
                                positions=positions)
            try:
                rec = {
                    "tag": os.environ.get("GS_CONV_TAG", ""),
                    "input_ids": [int(x) for x in input_ids.detach().cpu().tolist()],
                    "positions": (
                        None if positions is None
                        else [int(x) for x in positions.detach().cpu().tolist()]
                    ),
                    "adapter_indices": [int(x) for x in ai.detach().cpu().tolist()],
                    # The coded engine's recovered write address per row. Set only
                    # outside torch.compile, which holds here because the server
                    # runs with --enforce-eager. Under PRESERVE turn 2 carries two
                    # control tokens where RE_PREFILL carries one, so the tail
                    # addresses differ while the retrieved adapter must not.
                    "write_addresses": (
                        None
                        if getattr(self, "_debug_write_addresses", None) is None
                        else [
                            int(x)
                            for x in self._debug_write_addresses.detach()
                            .cpu()
                            .tolist()
                        ]
                    ),
                }
                with open(_T, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            except Exception as e:
                sys.stderr.write("gs-conv trace error: %r\n" % (e,))
            return ai, modified

        cls.forward = traced
        cls._gs_conv_patched = True

    if _TARGET in sys.modules:
        _patch(sys.modules[_TARGET])
    else:
        class _F(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name != _TARGET:
                    return None
                sys.meta_path.remove(self)
                try:
                    real = importlib.util.find_spec(name)
                finally:
                    sys.meta_path.insert(0, self)
                if real is None:
                    return None
                re_ = real.loader.exec_module

                def exec_module(mod, _r=re_):
                    _r(mod)
                    _patch(mod)

                real.loader.exec_module = exec_module
                return real

        sys.meta_path.insert(0, _F())
"""


def _read_trace(path, start=0):
    out = []
    if not os.path.exists(path):
        return out, start
    with open(path, "rb") as f:
        f.seek(start)
        raw = f.read()
        end = f.tell()
    for line in raw.split(b"\n"):
        line = line.strip().strip(b"\x00")
        if not line:
            continue
        try:
            out.append(json.loads(line.decode()))
        except Exception:
            continue
    return out, end


def _supported_flags(candidates):
    """Drop flags this vLLM build rejects (api_server exits on unknown flags)."""
    try:
        r = subprocess.run(
            [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--help"],
            capture_output=True,
            text=True,
            timeout=180,
        )
        help_text = (r.stdout or "") + (r.stderr or "")
    except Exception as e:
        print(f"flag probe failed ({e!r}); passing all", file=sys.stderr)
        return list(candidates)
    kept = [f for f in candidates if not f.startswith("--") or f in help_text]
    dropped = [f for f in candidates if f.startswith("--") and f not in help_text]
    if dropped:
        print(f"WARNING: vLLM build lacks {dropped}", file=sys.stderr)
    return kept


_TOKEN_ID_RE = re.compile(r"^token_id:(\d+)$")


def _generated_ids(choice):
    """Exact generated token ids, or None if this build cannot report them.

    Re-encoding the returned TEXT is not safe: ``encode(detokenize(ids))`` does
    not always reproduce ``ids`` (leading spaces, partial words), and a single
    wrong id in the answer makes the block that straddles the prompt/answer seam
    miss -- which caps reuse at the last boundary before the answer and destroys
    exactly the saving PRESERVE exists to produce. Measured: 64 hits where 96
    were available.

    With ``--return-tokens-as-token-ids`` the server reports each token as the
    literal string ``token_id:NNNN``, so the ids come back exactly.
    """
    lp = (choice or {}).get("logprobs") or {}
    toks = lp.get("tokens") or []
    ids = []
    for t in toks:
        m = _TOKEN_ID_RE.match(str(t))
        if not m:
            return None
        ids.append(int(m.group(1)))
    return ids or None


def _post(port, payload, timeout=240):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _metrics(port):
    """Scrape the two prefix-cache counters vLLM exposes."""
    out = {}
    try:
        url = f"http://127.0.0.1:{port}/metrics"
        with urllib.request.urlopen(url, timeout=20) as r:
            for line in r.read().decode().splitlines():
                for key in (
                    "vllm:prefix_cache_queries_total",
                    "vllm:prefix_cache_hits_total",
                ):
                    if line.startswith(key):
                        try:
                            out[key] = out.get(key, 0.0) + float(line.rsplit(" ", 1)[1])
                        except (IndexError, ValueError):
                            pass
    except Exception as e:
        out["error"] = repr(e)
    return out


def _wait_ready(port, proc, log_file, timeout=900):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            tail = ""
            if os.path.exists(log_file):
                with open(log_file) as f:
                    tail = f.read()[-3000:]
            raise RuntimeError(f"server exited early (rc={proc.returncode}):\n{tail}")
        try:
            health = f"http://127.0.0.1:{port}/health"
            with urllib.request.urlopen(health, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError(f"server not ready within {timeout}s")


HISTORY_FILLER = (
    "The incident report describes a phishing campaign that delivered a macro "
    "document, followed by a scripted download of a second-stage payload and "
    "persistence via a scheduled task. Analysts also observed credential access "
    "against a domain controller and lateral movement over SMB to two file "
    "servers before exfiltration to an external host. "
)

# Keyed by policy VALUE, not the enum: KVHistoryPolicy is imported lazily inside
# cmd_serve so the CLI does not pay for transformers on `build`.
CASE_MARKER = {
    "re_prefill": "Case A. ",
    "preserve_mixed_history": "Case B. ",
}


def cmd_serve(args):
    import tempfile

    from transformers import AutoTokenizer

    from granite_switch import Conversation, GraniteSwitchConfig, KVHistoryPolicy
    from granite_switch.tutorials.vllm_server import (
        kill_stale_vllm_processes,
        launch_vllm,
    )

    port = args.port
    workdir = tempfile.mkdtemp(prefix="gs_conv_")
    with open(os.path.join(workdir, "sitecustomize.py"), "w") as f:
        f.write(_SITECUSTOMIZE)
    trace = os.path.join(workdir, "trace.jsonl")
    log_file = os.path.join(workdir, "server.log")

    tok = AutoTokenizer.from_pretrained(args.model_path)
    config = GraniteSwitchConfig.from_pretrained(args.model_path)
    adapter_map = _parse_adapter_map(tok.chat_template or "")
    aloras = [n for n, (_t, tech, _i) in adapter_map.items() if tech == "alora"]

    result = {
        "num_adapters": config.num_adapters,
        "switch_type": config.switch_type,
        "adapter_map": {
            n: {"type": t, "invocation": i} for n, (_tok, t, i) in adapter_map.items()
        },
        "alora_names": aloras,
        "policies": {},
    }
    if len(aloras) < 2:
        result["error"] = f"need >=2 alora adapters, got {aloras}"
        with open(args.output_path, "w") as f:
            json.dump(result, f)
        print("SERVE_OK")
        return 0

    a_name, b_name = aloras[0], aloras[1]
    a_inv, b_inv = adapter_map[a_name][2], adapter_map[b_name][2]
    a_token, b_token = adapter_map[a_name][0], adapter_map[b_name][0]
    result["chosen"] = {"A": a_name, "B": b_name}
    control_ids = set(config.adapter_token_ids or [])

    os.environ["GS_CONV_TRACE"] = trace
    os.environ["PYTHONPATH"] = workdir + os.pathsep + os.environ.get("PYTHONPATH", "")

    kill_stale_vllm_processes()
    proc = launch_vllm(
        model=args.model_path,
        port=port,
        log_file=log_file,
        max_num_seqs=4,
        enforce_eager=True,
        max_model_len=4096,
        # Reuse is the measurement, so it must be on. It is vLLM V1's default,
        # but say it explicitly: a future default flip would silently zero the
        # whole experiment.
        extra_args=_supported_flags(
            [
                "--enable-prefix-caching",
                # Makes the completions response report token ids verbatim, so the
                # transcript can be extended with the ids the model really emitted
                # rather than a re-encoding of its detokenized text.
                "--return-tokens-as-token-ids",
            ]
        ),
    )

    def _user_text(base, invocation):
        """Real aLoRA usage: the invocation text goes in the turn it activates."""
        if invocation and "start_of_role" not in invocation:
            return f"{base} {invocation}"
        return base

    # Adapter name -> the integer index the switch should emit for it. The engine
    # reads adapter_token_ids[i] as expert i + offset, where offset is 0 only for
    # the num_adapters + 1 base-reset layout (composed with --base-reset-token;
    # the checkpoints under test here are composed without it, so offset is 1).
    ordered_ids = list(config.adapter_token_ids or [])
    index_offset = 0 if len(ordered_ids) == (config.num_adapters or 0) + 1 else 1

    def _expected_index(name):
        token_id = tok.convert_tokens_to_ids(adapter_map[name][0])
        return ordered_ids.index(token_id) + index_offset

    try:
        _wait_ready(port, proc, log_file)
        print("server ready (prefix caching ENABLED)", file=sys.stderr)
        offset = os.path.getsize(trace) if os.path.exists(trace) else 0

        for policy in (
            KVHistoryPolicy.RE_PREFILL,
            KVHistoryPolicy.PRESERVE_MIXED_HISTORY,
        ):
            conv = Conversation(tok, policy=policy, config=config)
            per_turn = []
            # Disjoint opening per policy. With shared content the policy that
            # runs second finds the first one's turn-1 blocks already cached
            # (observed: turn 1 routing only 10 of 74 tokens), which confounds the
            # comparison the test then draws from turn 2.
            #
            # SAME LENGTH in both arms, deliberately. Interpolating policy.value
            # made the openings 17 and 29 characters, so the two arms' prompts
            # differed in length for a reason that has nothing to do with the
            # policy, and their absolute hit counts were not comparable. "A"/"B"
            # keeps them disjoint at equal cost.
            marker = CASE_MARKER[policy.value]

            for turn, (question, adapter, invocation) in enumerate(
                [
                    (
                        marker + HISTORY_FILLER + "Which techniques does this map to?",
                        a_name,
                        a_inv,
                    ),
                    ("Now give me that as JSON.", b_name, b_inv),
                ],
                start=1,
            ):
                conv.user(_user_text(question, invocation))
                prompt = conv.build_prompt(adapter=adapter)

                os.environ["GS_CONV_TAG"] = f"{policy.value}:turn{turn}"
                before = _metrics(port)
                choice = _post(
                    port,
                    {
                        "model": args.model_path,
                        "prompt": prompt,
                        "max_tokens": MAX_TOKENS,
                        "temperature": 0.0,
                        "logprobs": 1,  # needed for tokens to be reported
                    },
                )["choices"][0]
                after = _metrics(port)
                records, offset = _read_trace(trace, offset)

                answer_ids = _generated_ids(choice)
                reply = choice["text"]
                # Prefer exact ids; fall back to text only if this build cannot
                # report them, and say so in the result so a weak measurement is
                # never mistaken for a real one.
                conv.record_answer(answer_ids or reply, adapter=adapter)
                sent = conv.sent_token_ids
                extends = sent[: len(prompt)] == list(prompt) if sent else None

                # Rows the switch was asked to route == positions NOT served from
                # cache. Prompt positions only: decode rows are single-token.
                prompt_rows = sum(
                    len(r["input_ids"]) for r in records if len(r["input_ids"]) > 1
                )
                # Decode rows: one row per generated token. Requests here are
                # sequential, so every single-token record inside this turn's trace
                # window belongs to this turn -- no seq_len disambiguation needed,
                # unlike the concurrent serving suite.
                #
                # Worth asserting on its own: measuring only prompt positions
                # leaves "which adapter actually generated the answer" to the
                # engine's separate decode-carry coverage, on a different harness.
                # Under PRESERVE it is also the riskiest row, because two control
                # tokens in the stream put the tail at a different codebook address
                # than RE_PREFILL's one -- so the address must move while the
                # adapter retrieved from it must not.
                decode_recs = [r for r in records if len(r["input_ids"]) == 1]
                decode_indices = sorted(
                    {int(r["adapter_indices"][0]) for r in decode_recs}
                )
                decode_addresses = sorted(
                    {
                        int(r["write_addresses"][0])
                        for r in decode_recs
                        if r.get("write_addresses")
                    }
                )
                per_turn.append(
                    {
                        "turn": turn,
                        "adapter": adapter,
                        "expected_adapter_index": _expected_index(adapter),
                        "decode_rows": len(decode_recs),
                        "decode_adapter_indices": decode_indices,
                        "decode_write_addresses": decode_addresses,
                        "prompt_len": len(prompt),
                        "prompt_rows_routed": prompt_rows,
                        "control_token_positions": [
                            i for i, t in enumerate(prompt) if t in control_ids
                        ],
                        "hits_delta": (
                            after.get("vllm:prefix_cache_hits_total", 0.0)
                            - before.get("vllm:prefix_cache_hits_total", 0.0)
                        ),
                        "queries_delta": (
                            after.get("vllm:prefix_cache_queries_total", 0.0)
                            - before.get("vllm:prefix_cache_queries_total", 0.0)
                        ),
                        "text": reply,
                        "answer_ids_source": "logprobs" if answer_ids else "text",
                        "answer_len": len(answer_ids) if answer_ids else None,
                        "sent_len": len(sent),
                        # The transcript must literally begin with what was sent, or
                        # the cache cannot match it.
                        "sent_extends_prompt": extends,
                    }
                )

            # Routing over turn 2's history region, from the trace.
            turn2 = per_turn[-1]
            history_routing = sorted(
                {
                    int(a)
                    for r in records
                    if len(r["input_ids"]) > 1
                    for a in r["adapter_indices"]
                }
            )
            turn2["routed_adapter_indices_seen"] = history_routing
            result["policies"][policy.value] = per_turn

        result["a_token"], result["b_token"] = a_token, b_token
        if os.path.exists(log_file):
            with open(log_file) as f:
                log = f.read()
            result["prefix_cache_log_lines"] = [
                ln for ln in log.splitlines() if "prefix cache" in ln.lower()
            ][-6:]
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=60)
        except Exception:
            proc.kill()
        if os.path.exists(log_file):
            with open(log_file) as f:
                print("--- server log tail ---\n" + f.read()[-3000:], file=sys.stderr)

    with open(args.output_path, "w") as f:
        json.dump(result, f)
    print(json.dumps(result.get("policies", {}), indent=1)[:3000], file=sys.stderr)
    print("SERVE_OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--output-dir", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--model-path", required=True)
    s.add_argument("--output-path", required=True)
    s.add_argument("--port", type=int, default=DEFAULT_PORT)
    a = ap.parse_args()
    return {"build": cmd_build, "serve": cmd_serve}[a.command](a)


if __name__ == "__main__":
    sys.exit(main())
